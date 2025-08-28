import torch

from export import QLinear
from quantization.gptq.GPTQQuantizer import LinearGPTQQuantizer
from quantization.__init__ import QuantizedModule


def transform_layers(module):
    if isinstance(module, QuantizedModule):

        if isinstance(module.default_quantizer, LinearGPTQQuantizer):
            return QLinear.pack_from_gptq_quantizer(module.default_quantizer)

    return module


    

