import torch
from torch import nn
from tqdm import tqdm


from quantization.AWQ.AWQQuantizer import LinearAwqQuantizer
from quantization.gptq.GPTQQuantizer import LinearGPTQQuantizer
from quantization.layers import LinearQuantHub

from utils.load_model import find_layers
from utils.memory import clear_mem


@torch.no_grad()
def llama_sequential(model, method, calibrate_data, **kwargs):
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
            block = layers[i].to(device)
            if not block_sequential:
                for j in range(len(calibrate_data)):
                    fp_outputs[j] = block(inputs[j].to(device),
                                          attention_mask=attention_mask[j] if attention_mask[j] == None else
                                          attention_mask[j].to(device), position_ids=position_ids[j].to(device))[0].to(
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
                    elif method == 'awq':
                        layer.register_quantizer(LinearAwqQuantizer(layer,  device=device, **kwargs["weight"]))
                    else:
                        raise RuntimeError(f'No {method} Quantizer!')
                    layer.prepare_hook()

                for j in range(len(calibrate_data)):
                    _ = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device),
                              position_ids=position_ids[j].to(device))[0].to(offload)

                for name, layer in tqdm(subset.items()):
                    layer.remove_hook()
                    layer.quantize()
                    layer.set_default_quantizer(0)
                    # del layer.core.weight
                    layer.core.weight.data = layer.quantizer[0].fake_w
                    layer.to(offload)
                    clear_mem()
                del subset

            if block_sequential:
                for j in range(len(calibrate_data)):
                    quant_outputs[j] = block(inputs[j].to(device), attention_mask=attention_mask[j].to(device),
                                             position_ids=position_ids[j].to(device))[0].to(offload)

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
