import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.efficientqat.quantizer import UniformAffineQuantizer

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc
#from quantize.utils import get_named_linears,set_op_by_name



class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64,
        mask=False
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
        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight, mask=mask)
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        #import pdb;pdb.set_trace()
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


def load_quantized_model(model_path, wbits, group_size, mixed_precision=False):
    print(f"Loading quantized model from {model_path}")

    #import pdb;pdb.set_trace()
    from quantization.efficientqat.utils import get_named_linears, set_op_by_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    layers = model.model.layers
    if mixed_precision:
        import json
        salient_columns = json.loads(open("salient_columns_v2.json").read())
        print("Train with a mixed-precision strategy and obtain salient columns.")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            if not mixed_precision:
                q_linear = QuantLinear(module, wbits, group_size)
            else:
                columns = salient_columns[f"{i}_{name}_salient_cols"]
                dim1, dim2 = module.shape
                columns_mask = torch.full((dim2,), False, device='cuda')
                columns_mask[columns] = True
                q_linear = QuantLinear(module, wbits, group_size, columns_mask)

            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer
