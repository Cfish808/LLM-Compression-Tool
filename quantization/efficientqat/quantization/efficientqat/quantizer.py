import torch
import torch.nn as nn

import pdb

CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class UniformAffineQuantizer_(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        mask=False
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        self.mask = mask
        assert (weight.shape[-1] - mask.sum()) % group_size == 0
        
        # init scale and zero point through Max-Min quantization
        #import pdb;pdb.set_trace()
        with torch.no_grad():
            if weight is not None:
                if isinstance(mask, bool):
                    x = weight.reshape(-1,self.group_size)
                    xmin = x.amin([-1], keepdim=True)
                    xmax =  x.amax([-1], keepdim=True)
                    range = xmax - xmin
                    scale = range / (2**self.n_bits-1)
                    scale = scale.clamp(min=1e-4, max=1e4)
                    zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                    self.scale = nn.Parameter(scale)
                    self.zero_point = nn.Parameter(zero_point.round())
                else:
                    weight_ = weight[:, ~mask]
                    x = weight_.reshape(-1,self.group_size)
                    xmin = x.amin([-1], keepdim=True)
                    xmax =  x.amax([-1], keepdim=True)
                    range = xmax - xmin
                    scale = range / (2**self.n_bits-1)
                    scale = scale.clamp(min=1e-4, max=1e4)
                    zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                    self.scale = nn.Parameter(scale)
                    self.zero_point = nn.Parameter(zero_point.round())
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    def merge_tensors(self, a, b, c):
        assert a.shape[-1] == b.shape[-1] + c.shape[-1]
        dd = torch.zeros_like(a, dtype=b.dtype)
        dd[:, a[0]] = b
        dd[:, ~a[0]] = c
        return dd
        
    def fake_quant(self, x_):
        #import pdb;pdb.set_trace()
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        if isinstance(self.mask, bool):
            x = x_
        else:
            x = x_[:, ~self.mask]
        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if not isinstance(self.mask, bool):
            mask_tensor = self.mask.unsqueeze(0).repeat(dim1, 1)
            x_dequant = self.merge_tensors(mask_tensor, x_[:, self.mask], x_dequant)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        #import pdb;pdb.set_trace()
        return x_dequant

class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
        mask=False
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.qmax2 = 2 ** 3 - 1
        self.qmax1 = 2 ** 2 - 1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True
        self.mask = mask
        # assert (weight.shape[-1] - mask.sum()) % group_size == 0
        
        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                if isinstance(mask, bool):
                    x = weight.reshape(-1,self.group_size)
                    xmin = x.amin([-1], keepdim=True)
                    xmax =  x.amax([-1], keepdim=True)
                    range = xmax - xmin
                    scale = range / (2**self.n_bits-1)
                    scale = scale.clamp(min=1e-4, max=1e4)
                    zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                    self.scale = nn.Parameter(scale)
                    self.zero_point = nn.Parameter(zero_point.round())
                elif isinstance(mask, torch.Tensor):
                    weight_ = weight[:, ~mask]
                    x = weight_.reshape(-1,self.group_size)
                    xmin = x.amin([-1], keepdim=True)
                    xmax =  x.amax([-1], keepdim=True)
                    range = xmax - xmin
                    scale = range / (2**self.n_bits-1)
                    scale = scale.clamp(min=1e-4, max=1e4)
                    zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4)
                    self.scale = nn.Parameter(scale)
                    self.zero_point = nn.Parameter(zero_point.round())
                elif isinstance(mask, dict):
                    mask2, mask1 = self.mask["mask2"], self.mask["mask1"]
                    masked2_tensor, masked1_tensor = weight.clone(), weight.clone()
                    masked2_tensor[:, ~mask2] = torch.tensor(-float('inf'), dtype=torch.float16)
                    masked1_tensor[:, ~mask1] = torch.tensor(-float('inf'), dtype=torch.float16)
                    x2 = masked2_tensor.reshape(-1, self.group_size)
                    x1 = masked1_tensor.reshape(-1, self.group_size)
                    xmax2 = x2.amax([-1], keepdim=True)
                    xmax1 = x1.amax([-1], keepdim=True)

                    masked2_tensor, masked1_tensor = weight.clone(), weight.clone()
                    masked2_tensor[:, ~mask2] = torch.tensor(float('inf'), dtype=torch.float16)
                    masked1_tensor[:, ~mask1] = torch.tensor(float('inf'), dtype=torch.float16)
                    x2 = masked2_tensor.reshape(-1, self.group_size)
                    x1 = masked1_tensor.reshape(-1, self.group_size)
                    xmin2 = x2.amin([-1], keepdim=True)
                    xmin1 = x1.amin([-1], keepdim=True)

                    range2 = xmax2 - xmin2
                    range1 = xmax1 - xmin1
                    scale2 = range2 / (2 ** 3 - 1)
                    scale1 = range1 / (2 ** 2 - 1)
                    scale2 = scale2.clamp(min=1e-4, max=1e4)
                    zero_point2 = -(xmin2 / scale2).clamp(min=-1e4, max=1e4)
                    scale1 = scale1.clamp(min=1e-4, max=1e4)
                    zero_point1 = -(xmin1 / scale1).clamp(min=-1e4, max=1e4)
                    self.scale1 = nn.Parameter(scale1)
                    self.zero_point1 = nn.Parameter(zero_point1.round())
                    self.scale2 = nn.Parameter(scale2)
                    self.zero_point2 = nn.Parameter(zero_point2.round())
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)

    def merge_tensors(self, a, b, c):
        assert a.shape[-1] == b.shape[-1] + c.shape[-1]
        dd = torch.zeros_like(a, dtype=b.dtype)
        dd[:, a[0]] = b
        dd[:, ~a[0]] = c
        return dd
        
    def fake_quant(self, x_):
        #import pdb;pdb.set_trace()
        if isinstance(self.mask, torch.Tensor) or isinstance(self.mask, bool):
            scale = clamp_ste(self.scale,1e-4, 1e4)
            round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

            if isinstance(self.mask, bool):
                x = x_
            else:
                x = x_[:, ~self.mask]
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            x_int = round_ste(x / scale)
            if round_zero_point is not None:
                x_int = x_int.add(round_zero_point)
            x_int = x_int.clamp(self.qmin, self.qmax)
            x_dequant = x_int
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
            if self.group_size:
                x_dequant = x_dequant.reshape(dim1, dim2)
            if not isinstance(self.mask, bool):
                mask_tensor = self.mask.unsqueeze(0).repeat(dim1, 1)
                x_dequant = self.merge_tensors(mask_tensor, x_[:, self.mask], x_dequant)
        else:
            # import pdb;pdb.set_trace()
            scale2 = clamp_ste(self.scale2, 1e-4, 1e4)
            round_zero_point2 = clamp_ste(round_ste(self.zero_point2), self.qmin, self.qmax2)
            scale1 = clamp_ste(self.scale1, 1e-4, 1e4)
            round_zero_point1 = clamp_ste(round_ste(self.zero_point1), self.qmin, self.qmax1)

            dim1, dim2 = x_.shape
            '''
            mask_x = torch.zeros_like(x_)
            x_2 = x_ * self.mask["mask2"].unsqueeze(0).repeat(dim1, 1).to(x_.device)
            x_1 = x_ * self.mask["mask1"].unsqueeze(0).repeat(dim1, 1).to(x_.device)
            x_3 = x_ * self.mask["mask3"].unsqueeze(0).repeat(dim1, 1).to(x_.device)
            x_2 = x_2.reshape(-1, self.group_size)
            x_1 = x_1.reshape(-1, self.group_size)
            x_2_int = round_ste(x_2 / scale2)
            x_1_int = round_ste(x_1 / scale1)
            if round_zero_point2 is not None:
                x_2_int = x_2_int.add(round_zero_point2)
            if round_zero_point1 is not None:
                x_1_int = x_1_int.add(round_zero_point1)
            x_2_int = x_2_int.clamp(self.qmin, self.qmax2)
            x_1_int = x_1_int.clamp(self.qmin, self.qmax1)

            x_2_dequant, x_1_dequant = x_2_int, x_1_int
            if round_zero_point2 is not None:
                x_2_dequant = x_2_dequant.sub(round_zero_point2)
            if round_zero_point1 is not None:
                x_1_dequant = x_1_dequant.sub(round_zero_point1)
            x_2_dequant = x_2_dequant.mul(scale2)
            x_1_dequant = x_1_dequant.mul(scale1)
            if self.group_size:
                x_2_dequant = x_2_dequant.reshape(dim1, dim2)
                x_1_dequant = x_1_dequant.reshape(dim1, dim2)

            x_dequant_ = x_3 + x_2_dequant + x_1_dequant
            '''

            x_3 = x_ * self.mask["mask3"].unsqueeze(0).repeat(dim1, 1).to(x_.device)

            x_2 = x_.reshape(-1, self.group_size)
            x_2_int = round_ste(x_2 / scale2)
            if round_zero_point2 is not None:
                x_2_int = x_2_int.add(round_zero_point2)
            x_2_dequant = x_2_int.clamp(self.qmin, self.qmax2)
            if round_zero_point2 is not None:
                x_2_dequant = x_2_dequant.sub(round_zero_point2)
            x_2_dequant = x_2_dequant.mul(scale2)
            x_2_dequant = x_2_dequant.reshape(dim1, dim2)

            x_1_int = round_ste(x_2 / scale1)
            if round_zero_point1 is not None:
                x_1_int = x_1_int.add(round_zero_point1)
            x_1_dequant = x_1_int.clamp(self.qmin, self.qmax1)
            if round_zero_point1 is not None:
                x_1_dequant = x_1_dequant.sub(round_zero_point1)
            x_1_dequant = x_1_dequant.mul(scale1)
            x_1_dequant = x_1_dequant.reshape(dim1, dim2)
            x_dequant = x_3 + x_2_dequant * self.mask["mask2"].unsqueeze(0).repeat(dim1, 1).to(x_.device) + x_1_dequant * self.mask["mask1"].unsqueeze(0).repeat(dim1, 1).to(x_.device)

        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        #import pdb;pdb.set_trace()
        return x_dequant        

    
