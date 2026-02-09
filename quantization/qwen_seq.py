import torch
from torch import nn
from tqdm import tqdm

# from transformers.quantizers.quantizer_awq import AwqQuantizer

from quantization.AWQ.AWQQuantizer import LinearAwqQuantizer
from quantization.gptq.GPTQQuantizer import LinearGPTQQuantizer
from quantization.smoothquant.SmoothQuantizer import LinearSmoothQuantizer
from quantization.rtn.RTNQuantizer import LinearRTNQuantizer
from quantization.omniquant.generate_act_scale_shift import generate_act_scale_shift
from quantization.omniquant.OmniQuantizer import omni_quantize
from quantization.Quip.QuipQuantizer import LinearQuipQuantizer
from quantization.owq.OWQQuantizer import LinearOWQQuantizer
from quantization.BiLLM.BILLMQuantizer import LinearBiLLMQuantizer
from quantization.spqr.SPQRQuantizer import LinearSPQRQuantizer
from quantization.layers import LinearQuantHub
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding # for transformers>=4.51
from packaging import version
import transformers

from utils.load_model import find_layers
from utils.memory import clear_mem


@torch.no_grad()
def qwen_sequential(model, method, calibrate_data, **kwargs):
    device = kwargs.get('device', 'cuda')
    offload = kwargs["weight"].get('offload', 'cpu')
    block_sequential = kwargs["weight"].get('block_sequential', False)
    layer_sequential = kwargs["weight"].get('layer_sequential', False)
    with torch.no_grad():
        use_cache = model.config.use_cache
        model_device = model.device
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        layers[0] = layers[0].to(device)
        
        if version.parse(transformers.__version__) >= version.parse('4.51.0'):
            rotary_emb = Qwen2RotaryEmbedding(model.config)
            embed_layer = model.get_input_embeddings()   # nn.Embedding

        dtype = next(iter(model.parameters())).dtype

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.inputs = []
                self.attention_mask = []
                self.position_ids = []

            def forward(self, input, **kwargs):
                self.inputs.append(input)
                self.attention_mask.append(kwargs['attention_mask'])
                self.position_ids.append(kwargs['position_ids'])
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in calibrate_data:
            try:
                model(batch.to(device))
            except ValueError:
                pass

        inputs = layers[0].inputs
        attention_mask = layers[0].attention_mask
        position_ids = layers[0].position_ids
        layers[0] = layers[0].module

        layers[0] = layers[0].to(offload)
        model.model.embed_tokens = model.model.embed_tokens.to(offload)
        model.model.norm = model.model.norm.to(offload)
        torch.cuda.empty_cache()

        quant_outputs = [None] * len(inputs)
        fp_outputs = [None] * len(inputs)

        for i in range(len(layers)):
            if hasattr(layers[i], "_hf_hook"):
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(layers[i], recurse=True)
                
            block = layers[i].to(device)
            if not block_sequential:
                for j in range(len(calibrate_data)):
                    if version.parse(transformers.__version__) < version.parse('4.51.0'):
                        fp_outputs[j] = block(inputs[j].to(device),
                                            attention_mask=attention_mask[j] if attention_mask[j] == None else
                                            attention_mask[j].to(device), position_ids=position_ids[j].to(device))[0].to(
                            offload)
                    else:
                        inputs_embeds = embed_layer(calibrate_data[j]).to(device)  
                        position_embeddings = rotary_emb(inputs_embeds, position_ids[j].to(device))
                        fp_outputs[j] = block(inputs[j].to(device),
                                            attention_mask=attention_mask[j] if attention_mask[j] == None else
                                            attention_mask[j].to(device), position_ids=position_ids[j].to(device), position_embeddings=position_embeddings)[0].to(
                            offload)
                        
            layer_linear = find_layers(block, (LinearQuantHub))
            if layer_sequential:
                sequential = [
                    ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                    ['self_attn.o_proj'],
                    ['mlp.up_proj', 'mlp.gate_proj'],
                    ['mlp.down_proj']
                ]
            else:
                sequential = [list(layer_linear.keys())]

            for names in sequential:
                subset = {n: layer_linear[n] for n in names}
                for name, layer in subset.items():
                    layer: LinearQuantHub
                    if method == 'gptq':
                        layer.register_quantizer(LinearGPTQQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method =='smoothquant':
                        layer.register_quantizer(LinearSmoothQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method == 'awq':
                        layer.register_quantizer(LinearAwqQuantizer(layer,  device=device, **kwargs["weight"]))
                    elif method == 'rtn':
                        layer.register_quantizer(LinearRTNQuantizer(layer,  device=device, **kwargs["weight"]))
                    elif method == 'quip':
                        layer.register_quantizer(LinearQuipQuantizer(layer,  device=device, **kwargs["weight"]))
                    elif method =='owq':
                        layer.register_quantizer(LinearOWQQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method =='spqr':
                        layer.register_quantizer(LinearSPQRQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method =='billm':
                        layer.register_quantizer(LinearBiLLMQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method=='awq+gptq':
                        layer.register_quantizer(LinearAwqQuantizer(layer,  device=device, **kwargs["weight"]))
                        layer.register_quantizer(LinearGPTQQuantizer(layer, device=device, **kwargs["weight"]))
                    elif method=='smoothquant+gptq':
                        layer.register_quantizer(LinearSmoothQuantizer(layer, device=device, **kwargs["weight"]))
                        layer.register_quantizer(LinearGPTQQuantizer(layer, device=device, **kwargs["weight"]))
                    else:
                        raise RuntimeError(f'No {method} Quantizer!')
                    layer.prepare_hook()

                for j in range(len(calibrate_data)):
                    if version.parse(transformers.__version__) < version.parse('4.51.0'):
                        _ = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device) if attention_mask[j] != None else None,
                                position_ids=position_ids[j].to(device))[0].to(offload)
                    else:
                        inputs_embeds = embed_layer(calibrate_data[j]).to(device)  
                        position_embeddings = rotary_emb(inputs_embeds, position_ids[j].to(device))
                        _ = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device) if attention_mask[j] != None else None,
                                position_ids=position_ids[j].to(device), position_embeddings=position_embeddings)[0].to(offload)
                        
                for name, layer in tqdm(subset.items()):
                    layer.remove_hook()
                    layer.quantize()
                    layer.set_default_quantizer(0)
                    # del layer.core.weight
                    if method != "awq": layer.core.weight.data = layer.quantizer[0].fake_w
                    else: layer.core.weight.data = layer.quantizer[0].fake_w.div(layer.quantizer[0].smooth_factor.view(1, -1))
                    layer.to(offload)
                    clear_mem()
                del subset

            if block_sequential:
                for j in range(len(calibrate_data)):
                    if version.parse(transformers.__version__) < version.parse('4.51.0'):
                        quant_outputs[j] = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device) if attention_mask[j] != None else None,
                                                position_ids=position_ids[j].to(device))[0].to(offload)
                    else:
                        inputs_embeds = embed_layer(calibrate_data[j]).to(device)  
                        position_embeddings = rotary_emb(inputs_embeds, position_ids[j].to(device))
                        quant_outputs[j] = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device) if attention_mask[j] != None else None,
                                                position_ids=position_ids[j].to(device), position_embeddings=position_embeddings)[0].to(offload)

            layers[i] = block.to(offload)
            del block
            clear_mem()
            if block_sequential:
                inputs, quant_outputs = quant_outputs, inputs
            else:
                inputs, fp_outputs = fp_outputs, inputs
        clear_mem()
    model.config.use_cache = use_cache
    model = model.to(model_device)
    return model


def qwen_quipsharp(calibrate,kwargs):
    from quantization.quip_sharp.quantize_qwen.complete_quantize_finetune_qwen import quip_sharp_main
    return quip_sharp_main(calibrate,kwargs)

