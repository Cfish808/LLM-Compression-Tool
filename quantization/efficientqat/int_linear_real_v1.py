import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from quantization.efficientqat.triton_utils.kernels import dequant_dim0, dequant_dim1
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
from quantization.efficientqat.utils import get_named_linears,set_op_by_name
from quantization.efficientqat.quantizer import UniformAffineQuantizer

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        # if bits not in [2, 4, 8]:
        #     raise NotImplementedError("Only 2,4,8 bits are supported.")
        # if infeatures % 32 != 0 or outfeatures % 32 != 0:
        #     raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.group_size), math.ceil(outfeatures / (32 // self.bits))), dtype=torch.int32)
        )
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )   # not used, just for consistent with GPTQ models
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False

    def post_init(self):
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        # use fake quantization for faster training but consume more memory
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        scale_zeros = zeros * scales
        self.scales = nn.Parameter(scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros = zeros.numpy().astype(np.uint32)
        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        qzeros = np.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), zeros.shape[1])):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
            weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        # out = torch.matmul(x, weight)
        out = torch.matmul(x, weight.to(x.dtype))
        out = out + self.bias if self.bias is not None else out
        return out

class QuantLinear_fake_v1(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight',org_module.weight) # trainable
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant


class QuantLinear_fake(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(
            self,
            org_module: nn.Linear,
            wbits=4,
            group_size=64,
            mask=False,
            mask_training_only=False
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_parameter('weight', org_module.weight)  # trainable
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.mask = mask
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.mask_training_only = mask_training_only
        # initialize quantizer
        #import pdb;pdb.set_trace()
        if mask_training_only:
            with torch.no_grad():
                self.mask_weight = nn.Parameter(torch.zeros(self.out_features, mask["mask3"].sum(), dtype=torch.float16).to(self.weight.device))
        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight, mask=mask)
        self.use_temporary_parameter = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            if self.mask_training_only:
                ''' 
                mask_x = torch.zeros_like(self.weight)
                mask_x[:, self.mask["mask3"]] = self.mask_weight
                weight = self.weight + mask_x
                '''
                bias = self.bias
                out = self.fwd_func(input, self.weight, bias, **self.fwd_kwargs) + self.fwd_func(input[:, :, self.mask["mask3"]], self.mask_weight, bias, **self.fwd_kwargs)
                #out = self.fwd_func(input, self.weight, bias, **self.fwd_kwargs)

                '''
                #import pdb;pdb.set_trace()
                weight = self.weight.clone()
                indices = torch.where(self.mask["mask3"])[0]
                for mask_idx, indice in enumerate(indices):
                    weight[:, indice] = weight[:, indice] + self.mask_weight[:, mask_idx]
                '''
                #weight = self.weight  # test
            else:
                weight = self.weight_quantizer(self.weight)
                bias = self.bias
                out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        else:
            weight = self.weight
            bias = self.bias
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

def load_quantized_model(model_path, wbits, group_size, real_quant=True, mixed_precision=False, mask_training_only=False, maskfile_dir="salient_columns.json"):
    print(f"Loading quantized model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    '''
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    '''
    if mixed_precision:
        import json
        salient_columns = json.loads(open(maskfile_dir).read())
        print("Train with a mixed-precision strategy and obtain salient columns.")
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16,trust_remote_code=True)
        # 在原模型结构中增加teacher_model模块，如果非distil的测试，需要把这部分注释
        # model.teacher_model = AutoModelForCausalLM.from_config(config=config,trust_remote_code=True)
    layers = model.model.layers
    import pdb;pdb.set_trace()
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            if real_quant: q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            else:
                if not mixed_precision:
                    q_linear = QuantLinear_fake(module, wbits, group_size)
                else:
                    '''
                    columns = salient_columns[f"{i}_{name}_salient_cols"]
                    columns = columns[: (len(columns) // group_size) * group_size]
                    dim1, dim2 = module.weight.shape
                    columns_mask = torch.full((dim2,), False, device='cuda')
                    columns_mask[columns] = True
                    '''

                    columns = salient_columns[f"{i}_{name}_salient_cols"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    dim1, dim2 = module.weight.shape
                    mask3 = torch.full((dim2,), False)
                    mask3[columns] = True

                    columns = salient_columns[f"{i}_{name}_Non-salient_cols1"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    mask2 = torch.full((dim2,), False)
                    mask2[columns] = True

                    columns = salient_columns[f"{i}_{name}_Non-salient_cols0"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    mask1 = torch.full((dim2,), False)
                    mask1[columns] = True

                    columns_mask = {"mask3": mask3, "mask2": mask2, "mask1": mask1}
                    q_linear = QuantLinear_fake(module, wbits, group_size, columns_mask, mask_training_only=mask_training_only)
                q_linear.use_weight_quant = True
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    if hasattr(model, "teacher_model"):
        del model.teacher_model
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

def trans2mixprecison_model(model_path, wbits, group_size, real_quant=True, mixed_precision=False, mask_training_only=False, maskfile_dir="salient_columns.json"):
    model, tokenizer = load_quantized_model(model_path, wbits, group_size, real_quant=real_quant, mixed_precision=mixed_precision,
                         mask_training_only=mask_training_only, maskfile_dir=maskfile_dir)
    if mixed_precision:
        import json
        salient_columns = json.loads(open(maskfile_dir).read())
    for name, module in model.model.named_modules():
        # if isinstance(module, LoraLayer):
        if mixed_precision and isinstance(module, QuantLinear_fake) and not 'head' in name:
            layer_num = name.split(".")[1]
            tensor_name = ".".join(name.split(".")[2 :])
            salient_cols = salient_columns[f"{layer_num}_{tensor_name}_salient_cols"]
            with torch.no_grad():
                # module.weight = module.weight_quantizer(module.weight)
                module.mask_weight = nn.Parameter(module.weight[:, salient_cols].clone())
                module.weight[:, salient_cols] = 0.0
    print('saving...')
    save_path = model_path + "_mix_precision_init"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def trans_blockwise2llama_model(model, config, tokenizer, model_path):
    print("translate the model from mix-precision blockwise to llama.")
    layers = model.model.layers
    for qlayer in layers:
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear_fake) and not 'head' in name:
                Linear_layer = torch.nn.Linear(module.weight.shape[0], module.weight.shape[1], bias=None)
                with torch.no_grad():
                    Linear_layer.weight = module.weight
                set_op_by_name(qlayer, name, Linear_layer)
                del module

    return model

def trans_e2e2llama_model(model, config, tokenizer, model_path, maskfile_dir):
    print("translate the model from mix-precision e2e mask training only to llama.")
    layers = model.model.layers
    import json
    salient_columns = json.loads(open(maskfile_dir).read())
    for layer_num, qlayer in enumerate(layers):
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear_fake) and not 'head' in name:
                tensor_name = name
                salient_cols = salient_columns[f"{layer_num}_{tensor_name}_salient_cols"]

                Linear_layer = torch.nn.Linear(module.weight.shape[0], module.weight.shape[1], bias=None)
                with torch.no_grad():
                    Linear_layer.weight = module.weight
                    Linear_layer.weight[:, salient_cols] = module.mask_weight
                set_op_by_name(qlayer, name, Linear_layer)
                del module

    return model
__all__ = ["QuantLinear","load_omniq_quantized"]
