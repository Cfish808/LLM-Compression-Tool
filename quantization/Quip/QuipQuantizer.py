import torch
import primefac
import scipy
import math
import torch.nn.functional as F
from utils.memory  import MEMORY_BANK, clear_mem
from ..__init__ import Precision, PRECISION_TO_BIT
from ..base import BaseQuantizer
from utils.util import track_quip_hessian_hook_to_cpu

class LinearQuipQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_linear, incoh_processing=True, wbit=Precision.FP16, abit=Precision.FP16, offload='cpu', device='cuda', qmethod='ldlq', npasses=10, qfn='b', preproc_gptqH=True, preproc_rescale=True, preproc_proj=True, preproc_proj_extra=0, alpha=0.01, percdamp=0.01, **kwarg):
        super().__init__(quant_hub_linear, wbit, abit, offload, device)
        self.npasses = npasses
        self.qfn = qfn
        self.qmethod = qmethod
        self.unbiased = False
        self.lazy_batch = False
        self.incoh_processing = incoh_processing
        
        # QuIP论文推荐的关键参数
        self.preproc_gptqH = preproc_gptqH
        self.preproc_rescale = preproc_rescale
        self.preproc_proj = preproc_proj
        self.preproc_proj_extra = preproc_proj_extra
        self.alpha = alpha
        self.percdamp = percdamp
        
        # 预处理状态标记
        self.preproc_done = False

    def add_hook(self):
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            if track_quip_hessian_hook_to_cpu not in self.quant_hub_linear.hook_func:
                self.quant_hub_linear.hook_func.append(track_quip_hessian_hook_to_cpu)
    
    def gen_rand_orthos(self, m,p):
        if (p != 2):
            return torch.tensor(scipy.stats.special_ortho_group.rvs(p, size=m)).to(torch.float32)
        X = torch.zeros(m,2,2)
        t = torch.rand(m) * (2 * math.pi) 
        sin_t = torch.sin(t)
        cos_t = torch.cos(t)
        X[:,0,0] = cos_t
        X[:,1,1] = cos_t
        X[:,0,1] = sin_t
        X[:,1,0] = -sin_t
        return X

    def butterfly_factors(self, n):
        pf = list(primefac.primefac(n))
        return (math.prod(pf[0::2]), math.prod(pf[1::2]))

    def gen_rand_ortho_butterfly(self, n):
        return ([self.gen_rand_orthos(n//p, p) for p in self.butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

    def gen_rand_ortho_butterfly_noblock(self, n):
        return ([self.gen_rand_orthos(1, p) for p in self.butterfly_factors(n)], torch.randperm(n), torch.randperm(n))

    def gen_rand_ortho_butterfly_nopermute(self, n):
        return ([self.gen_rand_orthos(n//p, p) for p in self.butterfly_factors(n)], torch.arange(n), torch.arange(n))

    def mul_ortho_butterfly(self, Bpp, x):
        (B, p_in, p_out) = Bpp
        assert((len(x.shape) == 1) or (len(x.shape) == 2))
        orig_dim = 2
        if (len(x.shape) == 1):
            (n,) = x.shape
            x = x.reshape(n,1)
            orig_dim = 1
        (n,q) = x.shape
        x = x[p_in,:]
        pfn = tuple(self.butterfly_factors(n))
        for i in range(len(pfn)):
            mpfx = math.prod(pfn[0:i])
            p = pfn[i]
            msfx = math.prod(pfn[(i+1):])
            x = x.reshape(mpfx, p, msfx, q).permute(0,2,1,3).reshape(mpfx * msfx, p, q)
            x = B[i] @ x
            x = x.reshape(mpfx, msfx, p, q).permute(0,2,1,3).reshape(n,q)
        x = x[p_out,:]
        if (orig_dim == 1):
            x = x.reshape(n)
        return x

    def rand_ortho_butterfly(self, n):
        return self.mul_ortho_butterfly(self.gen_rand_ortho_butterfly(n), torch.eye(n))

    def rand_ortho_butterfly_noblock(self, n):
        return self.mul_ortho_butterfly(self.gen_rand_ortho_butterfly_noblock(n), torch.eye(n))

    def rand_ortho_butterfly_nopermute(self, n):
        return self.mul_ortho_butterfly(self.gen_rand_ortho_butterfly_nopermute(n), torch.eye(n))

    def preproc(self, w, H, preproc_gptqH=True, percdamp=0.01,
                preproc_rescale=True, preproc_proj=True, preproc_proj_extra=0, alpha=0.01):
        """
        QuIP Incoherence Pre-Processing (Algorithm 1 from paper)
        """
        origin_w_dtype = w.dtype
        origin_H_dtype = H.dtype
        
        # 保存原始参数用于后处理
        U, V, scaleWH = None, None, None
        
        # 1. GPTQ-style H modification (from OPTQ)
        if preproc_gptqH:
            w = w.to(torch.float32)
            H = H.to(torch.float32)
            
            # 处理dead neurons
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            w[:, dead] = 0
            
            # 添加阻尼项 (论文中的α * mean(diag(H)) * I)
            damp = alpha * torch.mean(torch.diag(H))
            diag = torch.arange(w.size(1), device=self.device)
            H[diag, diag] += damp
            
            w = w.to(origin_w_dtype)
            H = H.to(origin_H_dtype)
        
        # 2. Diagonal rescaling (论文中的关键步骤)
        if preproc_rescale:
            w = w.to(torch.float32)
            H = H.to(torch.float32)
            
            # 论文公式: ˜D ← 4√(diag(H)/diag(W^T W))
            diagH = torch.diag(H)
            diagW2 = torch.diag(w.T @ w)
            
            # 避免除零和数值不稳定
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            
            # 正确的缩放因子计算 (四次方根)
            scaleWH = (diagH / diagW2).sqrt().sqrt()  # 相当于四次方根
            scaleWH = scaleWH.clamp(min=1e-8)
            
            # 应用缩放: W ← W˜D, H ← ˜D^{-1} H ˜D^{-1}
            w = w * scaleWH[None,:]
            H = H / scaleWH[None,:]
            H = H / scaleWH[:,None]
            
            w = w.to(origin_w_dtype)
            H = H.to(origin_H_dtype)
            scaleWH = scaleWH.to(torch.float32)
        
        # 3. Random orthogonal projection (核心的incoherence processing)
        if preproc_proj:
            w = w.to(torch.float32)
            H = H.to(torch.float32)
            
            # 生成随机正交矩阵 (论文中的U和V)
            if preproc_proj_extra == 0:
                U = self.rand_ortho_butterfly(w.shape[0]).to(torch.float32).to(w.device)
                V = self.rand_ortho_butterfly(w.shape[1]).to(torch.float32).to(w.device)
            elif preproc_proj_extra == 1:
                U = self.rand_ortho_butterfly_noblock(w.shape[0]).to(torch.float32).to(w.device)
                V = self.rand_ortho_butterfly_noblock(w.shape[1]).to(torch.float32).to(w.device)
            elif preproc_proj_extra == 2:
                U = self.rand_ortho_butterfly_nopermute(w.shape[0]).to(torch.float32).to(w.device)
                V = self.rand_ortho_butterfly_nopermute(w.shape[1]).to(torch.float32).to(w.device)

            # 应用投影: W ← U W V^T, H ← V H V^T
            w = U @ w @ V.T
            H = V @ H @ V.T
            
            w = w.to(origin_w_dtype)
            H = H.to(origin_H_dtype)

        self.preproc_done = True  
        return w, H, U, V, scaleWH 
    
    def postproc(self, w, H, U, V, scaleWH, preproc_proj=True, preproc_rescale=True):
        assert self.preproc_done is True
        origin_w_dtype = w.dtype
        origin_H_dtype = H.dtype
        if preproc_proj:
            w = w.to(torch.float32)
            H = H.to(torch.float32)
            U = U.to(w.device)
            V = V.to(w.device)
            w = (U.T @ w @ V)
            H = (V.T @ H @ V)
            w = w.to(origin_w_dtype)
            H = H.to(origin_H_dtype)
        if preproc_rescale:
            scaleWH = scaleWH.to(w.device)
            w = w / scaleWH[None,:]
            H = H * scaleWH[:,None]
            H = H * scaleWH[None,:]
        return w, H

    def find_params(self, x, bits, perchannel, sym, weight=False):
        maxq = torch.tensor(2**bits - 1)
        shape = x.shape
        if perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1).values
        xmax = x.max(1).values

        if sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = xmin == xmax
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if sym:
            zero = torch.full_like(scale, (maxq + 1) / 2)
        else:
            zero = -xmin / scale

        if not perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            scale = scale.repeat(tmp)
            zero = zero.repeat(tmp)
        
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            scale = scale.reshape(shape)
            zero = zero.reshape(shape)
            return scale, zero
        if len(shape) == 4:
            scale = scale.reshape((1, -1, 1, 1))
            zero = zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            scale = scale.reshape((1, 1, -1))
            zero = zero.reshape((1, 1, -1))
        if len(shape) == 2:
            scale = scale.unsqueeze(0)
            zero = zero.unsqueeze(0)
        return scale, zero, maxq

    def check_nbits(self, wr, nbits):
        (wr_vals, wr_counts) = torch.unique(wr, sorted=True, return_counts=True)
        assert (len(wr_vals) <= 2**nbits)
        return wr_counts
    
    def round_ldl(self, w, H, nbits, n_greedy_passes=9, unbiased=False):
        assert (not unbiased) or (n_greedy_passes == 0), "greedy passes are incompatible with unbiased LDL rounding"
        (d, d_) = H.shape
        assert (d == d_)
        (m, d) = w.shape
        L = torch.linalg.cholesky(H)
        L = L @ torch.diag(1/torch.diag(L))
        L = L - torch.eye(d, device=L.device)
        if unbiased:
            eta = torch.rand(w.shape).to(w.device)
        else:
            eta = 0.5 * torch.ones(w.shape).to(w.device)
        w_hat = w.clone()
        for i in reversed(range(d)):
            w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w[:,i:] - w_hat[:,i:]) @ L[i:,i] + eta[:,i]), min=0, max=2**nbits-1)
        
        wr = w_hat.clone()
        s = w_hat - w
        # H = H / H.diag().max() 
        for igp in range(n_greedy_passes):
            for i in reversed(range(d)):
                Hs = s @ H[:, i]
                epsXTsj = wr[:, i] - torch.round(wr[:, i] - Hs / H[i,i])
                wr[:, i] -= epsXTsj
                s[:, i] -= epsXTsj
            wr = torch.clamp(wr, min=0, max=2**nbits - 1)
            if ((w_hat == wr).all()):
                break
            w_hat.copy_(wr)
        
        wr_counts = self.check_nbits(wr, nbits)
        return wr
    
    def round_ldl_block(self, w, H, nbits, blocksize=128, n_greedy_passes=9, unbiased=False):
        assert (not unbiased) or (n_greedy_passes == 0), "greedy passes are incompatible with unbiased LDL rounding"
        (d, d_) = H.shape
        assert (d == d_)
        (m, d) = w.shape
        L = torch.linalg.cholesky(H)
        L = L @ torch.diag(1/torch.diag(L))
        L = L - torch.eye(d, device=L.device)
        if unbiased:
            eta = torch.rand(w.shape).to(w.device)
        else:
            eta = 0.5 * torch.ones(w.shape).to(w.device)
        w_hat = w.clone()
        for i2 in range(d, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            W1 = w[:, i1:i2]
            W2Hdiff = w[:, i2:] - w_hat[:, i2:]
            WHat1 = w_hat[:, i1:i2].clone()
            L1 = L[:, i1:i2]
            Eta1 = eta[:, i1:i2]

            for i in reversed(range(count)):
                WHat1[:,i] = torch.clamp(
                    torch.floor(W1[:,i] + (W1 - WHat1) @ L1[i1:i2,i] + W2Hdiff @ L1[i2:,i] + Eta1[:,i]), 
                    min=0, max=2**nbits-1)

            w_hat[:, i1:i2] = WHat1

        wr = w_hat.clone()
        s = w_hat - w
        H = H / H.diag().max()
        initial_err = torch.nn.functional.mse_loss(w, w_hat).item()
        for igp in range(n_greedy_passes):
            for i2 in range(d, 0, -blocksize):
                i1 = max(i2 - blocksize, 0)
                count = i2 - i1
                W1 = wr[:, i1:i2].clone()
                S0 = s[:, :i1]
                S1 = s[:, i1:i2].clone()
                S2 = s[:, i2:]
                H0 = H[:i1, i1:i2]
                H1 = H[i1:i2, i1:i2]
                H2 = H[i2:, i1:i2]

                for i in reversed(range(count)):
                    Hs = S0 @ H0[:, i] + S1 @ H1[:, i] + S2 @ H2[:, i]
                    epsXTsj = W1[:, i] - torch.round(W1[:, i] - Hs / H1[i,i])
                    W1[:, i] -= epsXTsj
                    S1[:, i] -= epsXTsj

                wr[:, i1:i2] = W1
                s[:, i1:i2] = S1

            wr = torch.clamp(wr, min=0, max=2**nbits - 1)
            initial_err = torch.nn.functional.mse_loss(w, w_hat).item()
            if ((w_hat == wr).all()):
                break
            w_hat.copy_(wr)
        
        wr_counts = self.check_nbits(wr, nbits)
        return wr

    def round_ldl_gptqequiv(self, w, H, nbits, unbiased=False):
        (d, d_) = H.shape
        assert (d == d_)
        (m, d) = w.shape
        H = torch.flip(H, [0,1])
        L = torch.linalg.cholesky(H)
        L = torch.flip(L,[0,1])
        L = L @ torch.diag(1/torch.diag(L))
        L = L - torch.eye(d, device=L.device)
        if unbiased:
            eta = torch.rand(w.shape).to(w.device)
        else:
            eta = 0.5 * torch.ones(w.shape).to(w.device)
        w_hat = w.clone()
        for i in range(d):
            w_hat[:,i] = torch.clamp(torch.floor(w[:,i] + (w[:,:i+1] - w_hat[:,:i+1]) @ L[:i+1,i] + eta[:,i]), min=0, max=2**nbits-1)
        wr = w_hat
        wr_counts = self.check_nbits(wr, nbits)
        return wr

    @torch.no_grad()
    def quantize_weight_vecbal(self, w, H, nbits, npasses, scale, zero, maxq, unbiased=False, qfn='b', qmethod='ldlq'):
        """
        QuIP量化函数 - 根据论文使用正确的qfn='b'和qmethod='ldlq'
        """
        if qfn == 'a':
            # 基础量化方法
            wr = torch.clamp((w/scale) + zero, 0, maxq)
            if qmethod == 'ldl_gptqequiv':
                wr = self.round_ldl_gptqequiv(wr, H, nbits=nbits, unbiased=unbiased)
            elif qmethod == 'ldlq':
                if self.lazy_batch is False:
                    wr = self.round_ldl(wr.float(), H, nbits=nbits, n_greedy_passes=npasses, unbiased=unbiased)
                else:
                    wr = self.round_ldl_block(wr.float(), H, nbits=nbits, n_greedy_passes=npasses, unbiased=unbiased)
            else:
                raise RuntimeError('not support {} qmethod'.format(qmethod))
                
            wr = scale * (wr - zero)
            return wr
            
        elif qfn == 'b':
            # QuIP论文中的对称不连贯量化函数
            # 论文公式：scale = ρ * ||W||_F / sqrt(m * n)，其中ρ=2.4
            m, n = w.shape
            w_fro = torch.norm(w, p='fro')  # 计算Frobenius范数
            
            # 正确的缩放因子计算
            scale = 2.4 * w_fro / math.sqrt(m * n) + 1e-16
            
            # 权重归一化到[-1,1]范围
            wr = w / scale
            
            # 映射到量化范围[0, maxq]
            wr = torch.clamp(((wr + 1) / 2) * maxq, 0, maxq)
            
            # 应用LDLQ舍入
            if qmethod == 'ldl_gptqequiv':
                wr = self.round_ldl_gptqequiv(wr, H, nbits=nbits, unbiased=unbiased)
            elif qmethod == 'ldlq':
                if self.lazy_batch is False:
                    wr = self.round_ldl(wr.float(), H, nbits=nbits, n_greedy_passes=npasses, unbiased=unbiased)
                else:
                    wr = self.round_ldl_block(wr.float(), H, nbits=nbits, n_greedy_passes=npasses, unbiased=unbiased)
            else:
                raise RuntimeError('not support {} qmethod'.format(qmethod))
            
            # 映射回原始范围
            wr = scale * (2 * wr / maxq - 1)
            return wr
        else:
            raise RuntimeError('not support {} qfn'.format(qfn))

    @torch.no_grad()
    def quantize(self):
        #quantize weight
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            H = self.quant_hub_linear.core.H.to(self.device)
            H = (H / self.quant_hub_linear.core.nsamples).to(torch.float32)
            w = self.quant_hub_linear.core.weight.clone().float().to(self.device)
            if self.incoh_processing:
                # 使用从配置文件传递的参数，而不是硬编码值
                w, H, U, V, scaleWH = self.preproc(
                    w, H, 
                    preproc_gptqH=self.preproc_gptqH,
                    preproc_rescale=self.preproc_rescale,
                    preproc_proj=self.preproc_proj,
                    preproc_proj_extra=self.preproc_proj_extra,
                    alpha=self.alpha,
                    percdamp=self.percdamp
                )
            scale, zero, maxq = self.find_params(w, bits=PRECISION_TO_BIT[self.wbit], perchannel=True, sym=False, weight=False)
            quant_w = self.quantize_weight_vecbal(
                w=w, H=H,
                nbits=PRECISION_TO_BIT[self.wbit],
                npasses=self.npasses,
                scale=scale,
                zero=zero,
                maxq=maxq,
                unbiased=self.unbiased,
                qfn=self.qfn,
                qmethod=self.qmethod
            )
            if self.incoh_processing:
                quant_w, H = self.postproc(quant_w, H, U, V, scaleWH, 
                                         preproc_proj=self.preproc_proj, 
                                         preproc_rescale=self.preproc_rescale)
            err = torch.nn.functional.l1_loss(w, quant_w)
            quant_w = quant_w.to(self.quant_hub_linear.core.weight.dtype)
            self.w_scale = MEMORY_BANK.add_value('{id}_w_scale'.format(id=id(self)), scale, self.offload)
            self.w_zero_point = MEMORY_BANK.add_value('{id}_w_zero_point'.format(id=id(self)), zero, self.offload)
            self.fake_w = quant_w
            
            del w, H, U, V, scaleWH, quant_w, err, self.quant_hub_linear.core.H, self.quant_hub_linear.core.nsamples
            clear_mem()
    
    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32:
            x = x.float()
        else:
             raise RuntimeError('QuIP quantizer cannot support quantization of activations to {} bit'.format(self.abit))
        
        if self.wbit == Precision.FP16:
            w = self.quant_hub_linear.core.weight.half()
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_linear.core.weight.float()
            x = x.float()
        else:
            w = self.fake_w.to(x)

        bias = None if self.quant_hub_linear.core.bias is None else self.quant_hub_linear.core.bias.to(x)
        return F.linear(x, w, bias).to(origin_dtype)
    
    def to(self, desc):
        if hasattr(self, 'fake_w'):
            self.fake_w.to(desc)
        if hasattr(self, 'w_scale'):
            self.w_scale.to(desc)
        if hasattr(self, 'w_zero_point'):
            self.w_zero_point.to(desc)
        return self
