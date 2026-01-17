import torch
import torch.nn.functional as F
from torch.ao.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver

from utils.memory import MEMORY_BANK, clear_mem
from utils.util import track_hessian_hook_to_cuda
from ..base import BaseQuantizer
from .. import Precision, PRECISION_TO_BIT


class LinearOWQQuantizer(BaseQuantizer):
    
    def __init__(self, quant_hub_linear, blocksize=128, groupsize=-1, percdamp=.01, actorder=True,
                 wbit=Precision.FP16, abit=Precision.FP16,
                 w_qscheme=torch.per_channel_affine, w_qtype='per_channel',
                 offload='cpu', device='cuda',
                 owq_ratio=0.1, **kwarg) -> None:
        super().__init__(quant_hub_linear, wbit, abit, offload, device)
        self.blocksize = blocksize
        self.groupsize = groupsize
        self.w_groupsize = groupsize
        self.nsamples = 0
        self.ready = False
        self.percdamp = percdamp
        self.actorder = actorder
        self.w_qtype = w_qtype
        self.rows = quant_hub_linear.core.out_features
        self.columns = quant_hub_linear.core.in_features
        self.w_qscheme = w_qscheme

        # >>> OWQ
        owq = True if owq_ratio > 0 else False
        self.owq = owq
        self.owq_ratio = owq_ratio
        self.out_ids = None  
        self.non_ids = None  

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_hessian_hook_to_cuda not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_hessian_hook_to_cuda)


    @torch.no_grad()
    def tensor_quant(self, x, scale, zero_point, bits, qscheme=torch.per_channel_symmetric):
        maxq = 2 ** PRECISION_TO_BIT[bits] - 1
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            return torch.fake_quantize_per_channel_affine(
                x, scale.to(x.device), zero_point.to(x.device),
                axis=0, quant_min=0, quant_max=maxq)
        else:
            return torch.fake_quantize_per_tensor_affine(
                x, scale.to(x.device), zero_point.to(x.device),
                quant_min=0, quant_max=maxq)

    def find_params(self, w, bits, qscheme=torch.per_channel_symmetric):
        if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
            obs = PerChannelMinMaxObserver(qscheme=qscheme, quant_min=0,
                                           quant_max=2 ** PRECISION_TO_BIT[bits] - 1)
        else:
            obs = MinMaxObserver(qscheme=qscheme, quant_min=0,
                                 quant_max=2 ** PRECISION_TO_BIT[bits] - 1)
        for wi in w:
            obs(wi)
        return obs.calculate_qparams()

    @torch.no_grad()
    def quantize(self):
        if self.wbit in [Precision.FP16, Precision.FP32]:
            return

        W = self.quant_hub_linear.core.weight.data.detach().to(self.device).float()
        H = self.quant_hub_linear.core.H.to(self.device)

        if self.owq:
            diag_H = torch.diag(H)
            n_out = int(self.columns * self.owq_ratio)
            self.out_ids = torch.topk(diag_H, n_out, largest=True)[1]
            self.out_ids = torch.sort(self.out_ids)[0]          # 升序
            mask = torch.ones(self.columns, dtype=torch.bool, device=W.device)
            mask[self.out_ids] = False
            self.non_ids = torch.where(mask)[0]
        else:
            self.out_ids = torch.tensor([], device=W.device, dtype=torch.long)
            self.non_ids = torch.arange(self.columns, device=W.device)

        if self.owq or self.actorder:
            perm = torch.cat([self.non_ids, self.out_ids]) if self.owq else \
                   torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.perm = perm   

        if self.owq:
            scale, zero_point = self.find_params([W[:, :len(self.non_ids)]],
                                                 self.wbit, self.w_qscheme)
        else:
            scale, zero_point = self.find_params([W], self.wbit, self.w_qscheme)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        w_scale = torch.empty(0)
        w_zero_point = torch.empty(0)

        loop_cols = len(self.non_ids) if self.owq else self.columns
        for i1 in range(0, loop_cols, self.blocksize):
            i2 = min(i1 + self.blocksize, loop_cols)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for j in range(count):
                w = W1[:, j]
                d = Hinv1[j, j]

                if self.groupsize != -1 and (i1 + j) % self.groupsize == 0:
                    end = min(i1 + j + self.groupsize, loop_cols)
                    scale, zero_point = self.find_params(
                        [W[:, (i1 + j):end]], self.wbit, self.w_qscheme)
                    w_scale = torch.cat((w_scale, scale.unsqueeze(0)), dim=1)
                    w_zero_point = torch.cat((w_zero_point, zero_point.unsqueeze(0)), dim=1)

                q = self.tensor_quant(w, scale, zero_point, self.wbit,
                                      qscheme=self.w_qscheme).flatten()
                Q1[:, j] = q
                Losses1[:, j] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d
                W1[:, j:] -= err1.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
                Err1[:, j] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        #outlier列保留 FP16
        if self.owq:
            Q[:, len(self.non_ids):] = W[:, len(self.non_ids):]

        if (self.owq or self.actorder) and hasattr(self, 'perm'):
            invperm = torch.argsort(self.perm)
            Q = Q[:, invperm]

        #保存
        self.fake_w = Q.reshape(self.quant_hub_linear.core.weight.shape).to(
            self.quant_hub_linear.core.weight.data.dtype)
        if self.groupsize == -1 and not self.owq:
            w_scale = scale.unsqueeze(1)
            w_zero_point = zero_point.unsqueeze(1)
        self.w_scale, self.w_zero_point = w_scale, w_zero_point
        del W, H, Losses, damp, diag, Hinv, W1, Q1, Err1, Losses1, Hinv1, self.quant_hub_linear.core.H
        clear_mem()

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            raise RuntimeError(f'gptq quantizer cannot quantize activation to {PRECISION_TO_BIT[self.abit]} bit')

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
        return self