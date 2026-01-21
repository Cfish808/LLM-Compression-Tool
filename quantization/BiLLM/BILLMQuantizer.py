import torch
import torch.nn.functional as F
import math
import time
from ..base import BaseQuantizer
from ..__init__ import Precision, PRECISION_TO_BIT
from utils.util import  track_hessian_hook_to_cpu, track_hessian_hook_to_cuda
from torch.ao.quantization.observer import PerChannelMinMaxObserver, MinMaxObserver
from utils.memory import MEMORY_BANK,clear_mem
import numpy as np


# ==================== 移植的BiLLM核心函数 ====================

@torch.no_grad()
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone() * mask
    for od in range(order):
        residual = new_matrix - sum_order
        masked_residual = torch.where(mask, residual, torch.tensor(float('nan'), device=x.device))

        mean_tensor = torch.nanmean(masked_residual, dim=1, keepdim=True)
        mean_tensor = torch.where(torch.isnan(mean_tensor), torch.zeros_like(mean_tensor), mean_tensor)

        centered = masked_residual - mean_tensor
        scale_tensor = torch.nanmean(torch.abs(centered), dim=1, keepdim=True)
        scale_tensor = torch.where(torch.isnan(scale_tensor), torch.zeros_like(scale_tensor), scale_tensor)

        binary = torch.sign(centered) * scale_tensor + mean_tensor
        sum_order = sum_order + binary * mask
    return sum_order


@torch.no_grad()
def generate_structural_mask(origin_matrix, mask3, braq1_border):
    mask1_2 = ~mask3
    binary_group = torch.abs(origin_matrix * mask1_2)

    mask2 = binary_group >= braq1_border  # 大权重1-bit组
    mask1 = binary_group < braq1_border   # 小权重1-bit组

    mask1 = mask1 & mask1_2
    mask2 = mask2 & mask1_2
    return mask1, mask2


@torch.no_grad()
def structural_searching(origin_matrix, up_lim=50):
    #print(f"[Matrix Debug]: {origin_matrix}")
    true_counts = origin_matrix.abs().sum(dim=0)  # 每列重要性

    # 搜索显著列
    minimal_value_0 = float('inf')
    optimal_split_0 = 0
    _, top_columns = torch.topk(true_counts, up_lim)

    for i in range(1, up_lim):
        mask3 = torch.zeros_like(origin_matrix, dtype=torch.bool)
        mask3[:, top_columns[:i]] = True

        group3 = high_order_residual(origin_matrix, mask3, order=2)
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)
        error = torch.mean((origin_matrix - (group3 + group4)) ** 2)

        if error < minimal_value_0:
            minimal_value_0 = error
            optimal_split_0 = i

    #mask3
    _, top_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.zeros_like(origin_matrix, dtype=torch.bool)
    mask3[:, top_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)


    search_matrix = origin_matrix * (~mask3)
    flat_abs_tensor = torch.abs(search_matrix).view(-1)

    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy())
    ).to(origin_matrix.device)

    minimal_value = float('inf')
    optimal_split = 0.0

    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        total_error = torch.mean((origin_matrix - (group1 + group2 + group3)) ** 2)
        if total_error < minimal_value:
            minimal_value = total_error
            optimal_split = split_value

    #print(f"Debug: optimal_significant_cols = {optimal_split_0}")
    #print(f"Debug: optimal_threshold = {optimal_split.item():.8f}")

    return optimal_split, mask3


@torch.no_grad()
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=50):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape(1, -1) + 1e-6) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    #print(mask1.sum().item() / mask1.numel(), mask2.sum().item() / mask2.numel(), mask3.sum().item() / mask3.numel(),"\n")

    return mask1, mask2, mask3



class LinearBiLLMQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear,
                 blocksize=128,
                 percdamp=0.01,
                 actorder=False, 
                 wbit=Precision.FP16,
                 abit=Precision.FP16,
                 offload='cpu',
                 device='cuda',
                 salient_metric="magnitude",
                 disable_gptq=False,
                 up_lim=50,
                 orders=(1, 1, 2),
                 **kwargs):
        super().__init__(quant_hub_linear, wbit, abit, offload, device)

        self.blocksize = blocksize
        self.percdamp = percdamp
        self.actorder = actorder
        self.salient_metric = salient_metric
        self.disable_gptq = disable_gptq
        self.up_lim = up_lim
        self.orders = orders

        self.rows = self.quant_hub_linear.core.out_features
        self.columns = self.quant_hub_linear.core.in_features

    def add_hook(self):
        if track_hessian_hook_to_cuda not in self.quant_hub_linear.hook_func:
            self.quant_hub_linear.hook_func.append(track_hessian_hook_to_cuda)

    @torch.no_grad()
    def quantize(self):
        if self.wbit in [Precision.FP16, Precision.FP32]:
            return

        W = self.quant_hub_linear.core.weight.data.detach().clone().float().to(self.device)
        H = self.quant_hub_linear.core.H.to(self.device).float()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        #print(H)


        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.device)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        #print("H after inv:", H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        #print("Hinv:", Hinv)

        Q = torch.zeros_like(W)
        Losses = torch.zeros(self.rows, device=self.device)

        tick = time.time()

        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)

            W1 = W[:, i1:i2].clone()
            #print("W1:", i1, " ", i2)
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            mask1, mask2, mask3 = structural_guassian_distribution(
                W1, H=H[i1:i2, i1:i2] if self.salient_metric == "hessian" else None,  
                metric=self.salient_metric, up_lim=self.up_lim
            )

            q1 = high_order_residual(W1, mask1, order=self.orders[0])
            q2 = high_order_residual(W1, mask2, order=self.orders[1])
            q3 = high_order_residual(W1, mask3, order=self.orders[2])
            q_part = q1 + q2 + q3

            if self.disable_gptq:
                Q[:, i1:i2] = q_part
            else:
                for i in range(i2 - i1):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    q = q_part[:, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, dim=1) / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if self.actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        torch.cuda.synchronize()
        #print(f"BiLLM quant time: {time.time() - tick:.2f}s, total loss: {torch.sum(Losses).item():.4f}")

        self.fake_w = Q.reshape(self.quant_hub_linear.core.weight.shape).to(self.quant_hub_linear.core.weight.dtype)

        del W, H, Q, Losses, Hinv, W1, Q1, Err1, Losses1, Hinv1
        del self.quant_hub_linear.core.H
        clear_mem()

    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
            raise RuntimeError('BiLLM quantizer does not support activation quantization below FP16')

        if self.wbit in [Precision.FP16, Precision.FP32]:
            w = self.quant_hub_linear.core.weight.to(x.dtype).to(x.device)
        else:
            w = self.fake_w.to(x.device)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x.device)
        out = F.linear(x, w, bias)
        return out.to(origin_dtype)

    def to(self, desc):
        if hasattr(self, 'fake_w'):
            self.fake_w = self.fake_w.to(desc)
        return self