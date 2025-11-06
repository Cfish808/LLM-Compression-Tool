import os
import sys
import random
import numpy as np
from .models.LMClass import LMClass
import torch
import time
from lm_eval import evaluator
from pprint import pprint
from .parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from .quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from .categories import subcategories, categories
import pdb
from easydict import EasyDict


torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]


def omni_quantize(
    model_name_or_path,
    model,
    seed=42,
    epochs=10,
    wbit=4,
    abit=16,
    w_groupsize=None,
    deactive_amp=True,
    let=False,
    lwc=False,
    alpha=0.5,
    let_lr=5e-3,
    lwc_lr=1e-2,
    wd=0,
    symmetric=False,
    disable_zero_point=False,
    aug_loss=False,
    a_qtype="per_token",
    w_qtype="per_channel",
    seqlen=2048,
    act_scales=None,
    act_shifts=None,
    dataloader=None,
    batch_size=1,
    attn_implementation="eager",
    resume=None,
    real_quant=False,
    nsamples=128,
    limit=-1,
    logger=None,
    **kwargs
):    
    args = EasyDict()
    args.net = model_name_or_path
    args.model = model_name_or_path
    args.resume = resume
    args.real_quant = real_quant
    args.nsamples = nsamples
    args.batch_size = batch_size
    args.seed = seed
    args.wbits = wbit
    args.abits = abit
    args.group_size = w_groupsize
    args.deactive_amp = deactive_amp
    args.let = let
    args.lwc = lwc
    args.alpha = alpha
    args.let_lr = let_lr
    args.lwc_lr = lwc_lr
    args.wd = wd
    args.symmetric = symmetric
    args.disable_zero_point = disable_zero_point
    args.aug_loss = aug_loss
    args.a_dynamic_method = a_qtype
    args.w_dynamic_method = w_qtype
    args.seqlen = seqlen
    args.act_scales = act_scales
    args.act_shifts = act_shifts
    args.dataloader = dataloader
    args.attn_implementation = attn_implementation
    args.epochs = epochs
    args.logger = logger
    args.limit = limit
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # check
    if epochs > 0:
        assert lwc or let
        
    if (wbit<16 and wbit>=8) or (abit<16 and abit>=8):
        deactive_amp = True
    
    logger.info(args)
    lm = LMClass(model, args)
    lm.seqlen = seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": wbit,
        "per_channel_axes": [0],
        "symmetric": symmetric,
        "dynamic_method": w_qtype,
        "group_size": w_groupsize,
        "lwc":lwc,
        "disable_zero_point": disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  abit,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": a_qtype,
    }
    args.q_quant_params = {
        "n_bits": abit,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": a_qtype,
    }
    args.k_quant_params = {
        "n_bits": abit,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": a_qtype,
    }
    args.v_quant_params = {
        "n_bits": abit,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": a_qtype,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    # gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
    # lm._device = f"cuda:{gpu_id}"
    # logger.info(f"set quantization in gpu {gpu_id}")

    # quantization
    if wbit < 16 or abit <16:
        logger.info("=== start quantization ===")
        tick = time.time()        
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    
    # if args.save_dir:
    #     # delete omni parameters
    #     for name, module in lm.model.named_modules():
    #         if isinstance(module, QuantLinear):
    #             del module.weight_quantizer.lowbound_factor
    #             del module.weight_quantizer.upbound_factor
    #         if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
    #             if args.let:
    #                 del module.qkv_smooth_scale
    #                 del module.qkv_smooth_shift
    #                 del module.out_smooth_scale
    #                 del module.out_smooth_shift
    #                 del module.fc1_smooth_scale
    #                 del module.fc1_smooth_shift           
        # lm.model.save_pretrained(args.save_dir)  
        # lm.tokenizer.save_pretrained(args.save_dir) 
    return lm.model

