import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
from utils.memory import MEMORY_BANK, clear_mem
from utils.util import track_hessian_hook_to_cpu, track_hessian_hook_to_cuda
from ..base import BaseQuantizer
from ..__init__ import Precision, PRECISION_TO_BIT


class LinearSPQRQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, blocksize=128, groupsize=-1, percdamp=0.01, actorder=True,
                 wbit=Precision.FP16, abit=Precision.FP16, w_qscheme=torch.per_channel_affine, w_qtype='per_channel',
                 offload='cpu', device='cuda', outlier_relative_threshold=0.00001, simplified_outliers=False, 
                 save_quantization=False, **kwargs) -> None:
        super().__init__(quant_hub_linear, wbit, abit, offload, device)
        self.blocksize = blocksize
        self.groupsize = groupsize if groupsize != -1 else quant_hub_linear.core.in_features  
        self.w_groupsize = self.groupsize
        self.percdamp = percdamp
        self.actorder = actorder
        self.w_qtype = w_qtype
        self.w_qscheme = w_qscheme
        self.outlier_relative_threshold = outlier_relative_threshold  # SPQR 异常值阈值
        self.simplified_outliers = simplified_outliers  # 是否使用简化异常值检测
        self.save_quantization = save_quantization  

        self.rows = self.quant_hub_linear.core.out_features
        self.columns = self.quant_hub_linear.core.in_features
        self.nsamples = 0
        self.ready = False

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_hessian_hook_to_cuda not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_hessian_hook_to_cuda)

    @torch.no_grad()
    def tensor_quant(self, x, scale, zero_point, bits, qscheme=torch.per_channel_symmetric):
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            x = torch.fake_quantize_per_channel_affine(
                x,
                scale.to(x.device),
                zero_point.to(x.device),
                0, 0, 2 ** PRECISION_TO_BIT[bits] - 1
            ).to(x)
        else:
            x = torch.fake_quantize_per_tensor_affine(
                x,
                scale.to(x.device),
                zero_point.to(x.device),
                0, 2 ** PRECISION_TO_BIT[bits] - 1
            ).to(x)
        return x

    def find_params(self, w, bits, qscheme=torch.per_channel_symmetric):
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            observer = PerChannelMinMaxObserver(
                qscheme=qscheme,
                quant_min=0,
                quant_max=2 ** PRECISION_TO_BIT[bits] - 1
            )
        else:
            observer = MinMaxObserver(
                qscheme=qscheme,
                quant_min=0,
                quant_max=2 ** PRECISION_TO_BIT[bits] - 1
            )
        for i in w:
            observer(i)
        scale, zero_point = observer.calculate_qparams()
        return scale, zero_point

    @torch.no_grad()
    def get_leave_one_out_error(self, group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, bits, sym):
        assert group_weight.ndim == 2
        loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
        loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
        groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, groupsize, groupsize-1]

        fast_quantizer = PerChannelMinMaxObserver(
            qscheme=self.w_qscheme,
            quant_min=0,
            quant_max=2 ** PRECISION_TO_BIT[bits] - 1
        )
        fast_quantizer(groupwise_loo_data.flatten(0, 1))  
        scale, zero_point = fast_quantizer.calculate_qparams()

        #LOO
        loo_groupwise_reconstructed_weights = self.tensor_quant(
            groupwise_loo_data.flatten(0, 1), scale, zero_point, bits, qscheme=self.w_qscheme
        ).reshape_as(groupwise_loo_data)

        loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [groupsize, groupsize-1]
        loo_errors_sq = (
            ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
        )

        # 计算基线量化误差
        base_quantizer = PerChannelMinMaxObserver(
            qscheme=self.w_qscheme,
            quant_min=0,
            quant_max=2 ** PRECISION_TO_BIT[bits] - 1
        )
        base_quantizer(group_weight)
        scale, zero_point = base_quantizer.calculate_qparams()
        baseline_reconstructed_weights = self.tensor_quant(
            group_weight, scale, zero_point, bits, qscheme=self.w_qscheme
        )
        baseline_errors_sq = (
            ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
        )

        #误差减少量
        reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
        return reduction_in_squared_error

    @torch.no_grad()
    def quantize(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            W = self.quant_hub_linear.core.weight.data.detach().to(self.device)
            H = self.quant_hub_linear.core.H.to(self.device)
            W = W.float()

            if self.actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
            else:
                perm = torch.arange(self.columns, device=self.device)

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.device)
            H[diag, diag] += damp

            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            Hinv_diag = torch.diag(Hinv)

            Q = torch.zeros_like(W)
            quantization_errors = torch.zeros_like(W)
            unstructured_outlier_mask = torch.zeros_like(W, dtype=torch.bool)
            save_quant_dict = {} if self.save_quantization else None
            if self.save_quantization:
                save_quant_dict["quant_weights"] = []
                save_quant_dict["quant_layer_scale"] = []
                save_quant_dict["quant_layer_zeros"] = []
                save_quant_dict["outliers_matrix"] = torch.zeros_like(W, dtype=W.dtype, device=W.device)

            #异常值阈值
            outlier_scale = (W.var(dim=0) / Hinv_diag.square()).mean().item()
            unstructured_outlier_threshold = self.outlier_relative_threshold * outlier_scale

            in_group_index = -1  # 分组索引
            for i1 in range(0, self.columns, self.blocksize):
                i2 = min(i1 + self.blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.groupsize != -1 and (i1 + i) % self.groupsize == 0:
                        in_group_index += 1
                        group_weight = W[:, (i1 + i):(i1 + i + self.groupsize)]

                        if not self.simplified_outliers and self.outlier_relative_threshold != float("inf"):
                            loo_quantization_error_sq = self.get_leave_one_out_error(
                                group_weight, Hinv_diag[i1 + i:i1 + i + self.groupsize], self.wbit, self.w_qscheme == torch.per_channel_symmetric
                            )
                            #print(loo_quantization_error_sq)
                            #print("1% threshold of loo_quantization_error_sq:", torch.quantile(loo_quantization_error_sq.flatten(), 0.99).item())
                            likely_unstructured_outlier_mask = (
                                loo_quantization_error_sq > unstructured_outlier_threshold
                            ).float()
                            non_outlier_mask = 1 - likely_unstructured_outlier_mask
                            mean_over_non_outliers = torch.sum(
                                group_weight * non_outlier_mask, dim=1, keepdim=True
                            ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                            group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (1 - non_outlier_mask)
                            scale, zero_point = self.find_params([group_weight_without_outliers], self.wbit, self.w_qscheme)
                        else:
                            scale, zero_point = self.find_params([group_weight], self.wbit, self.w_qscheme)

                        if self.save_quantization:
                            save_quant_dict["quant_layer_scale"].append(scale.to(W.dtype))
                            save_quant_dict["quant_layer_zeros"].append(zero_point.to(W.dtype))

                    # 量化当前列
                    q = self.tensor_quant(w.unsqueeze(1), scale, zero_point, self.wbit, qscheme=self.w_qscheme).flatten()
                    Q1[:, i] = q
                    err1 = (w - q) / d
                    quantization_errors[:, i1 + i] = err1

                    if self.outlier_relative_threshold != float("inf"):
                        unstructured_outlier_mask[:, i1 + i] = (err1.square() > unstructured_outlier_threshold)
                        is_outlier = unstructured_outlier_mask[:, i1 + i].float()
                        q_no_outlier = self.tensor_quant(
                            (w * (1 - is_outlier)).unsqueeze(1), scale, zero_point, self.wbit, qscheme=self.w_qscheme
                        ).flatten()
                        Q1[:, i] = q_no_outlier * (1 - is_outlier) + w * is_outlier
                        if self.save_quantization:
                            save_quant_dict["outliers_matrix"][:, i1 + i] = w * is_outlier
                        err1 = (w - Q1[:, i]) / d
                        quantization_errors[:, i1 + i] = err1


                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                    if self.save_quantization:
                        save_quant_dict["quant_weights"].append(q.unsqueeze(1).to(torch.int8))

                Q[:, i1:i2] = Q1
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()

            if self.actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
                quantization_errors = quantization_errors[:, invperm]
                unstructured_outlier_mask = unstructured_outlier_mask[:, invperm]
                if self.save_quantization:
                    save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"][:, invperm].to_sparse()

            # 保存量化参数
            if self.groupsize == -1:
                scale, zero_point = self.find_params([W], self.wbit, self.w_qscheme)
                self.w_scale = scale.unsqueeze(1)
                self.w_zero_point = zero_point.unsqueeze(1)
            else:
                self.w_scale = torch.cat(save_quant_dict["quant_layer_scale"], dim=1) if self.save_quantization else scale
                self.w_zero_point = torch.cat(save_quant_dict["quant_layer_zeros"], dim=1) if self.save_quantization else zero_point

            if self.save_quantization:
                save_quant_dict["perm"] = perm.to(torch.int32)
                save_quant_dict["weight_shape"] = W.shape
                save_quant_dict["groupsize"] = self.groupsize
                save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)

            Q = Q.reshape(self.quant_hub_linear.core.weight.shape).to(self.quant_hub_linear.core.weight.data.dtype)
            self.fake_w = Q
            self.quantization_errors = quantization_errors
            self.unstructured_outlier_mask = unstructured_outlier_mask
            self.save_quant_dict = save_quant_dict

            
            del W, H, damp, diag, Hinv, W1, Q1, Err1, Hinv1, self.quant_hub_linear.core.H
            clear_mem()

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            raise RuntimeError('SPQR quantizer cannot support quantization of activations to {} bit'.format(
                PRECISION_TO_BIT[self.abit]))

        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)

    def to(self, desc):
        if hasattr(self, 'w_scale'):
            self.w_scale = self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point = self.w_zero_point.to(desc)
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        if hasattr(self, 'quantization_errors'):
            self.quantization_errors = self.quantization_errors.to(desc)
        if hasattr(self, 'unstructured_outlier_mask'):
            self.unstructured_outlier_mask = self.unstructured_outlier_mask.to(desc)
        if hasattr(self, 'save_quant_dict') and self.save_quant_dict:
            for key in self.save_quant_dict:
                if isinstance(self.save_quant_dict[key], torch.Tensor):
                    self.save_quant_dict[key] = self.save_quant_dict[key].to(desc)
        return self