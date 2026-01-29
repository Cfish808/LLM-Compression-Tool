#!/usr/bin/env python3
"""
完整的 Qwen 模型量化微调脚本
完整合并以下四个文件：
1. quantize_finetune_qwen.py - 量化微调主流程
2. hessian_offline_qwen.py - Hessian 矩阵计算
3. finetune_e2e_qwen.py - 端到端微调
4. hfize_qwen.py - HuggingFace 模型转换
"""

import os
from tqdm import tqdm
from types import SimpleNamespace
import datetime
import random
from copy import deepcopy

import argparse
import copy
import gc
import math
import time

import glog
import numpy
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerFast)
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from quantization.quip_sharp.lib import codebook, utils
from quantization.quip_sharp.lib.algo import finetune, quip
from quantization.quip_sharp.lib.linear import FusedLinear
from quantization.quip_sharp.lib.utils.unsafe_import import model_from_hf_path
from quantization.quip_sharp.model.qwen import Qwen2ForCausalLM, Qwen2DecoderLayer

def check_exist(idx, args):
    """检查量化文件是否已存在"""
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True

def quantize_qwen_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                         model_config):
    """量化单个 Qwen 层"""
    if check_exist(idx, args):
        return

    mixed_layer = Qwen2DecoderLayer(model_config, idx).cpu()
    with torch.no_grad():
        weights = [
            layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ]

        fused_qkv_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                     weights[0].shape[1],
                                     sum([_.shape[0] for _ in weights]),
                                     bias=False)
        cur = 0
        for w in weights:
            fused_qkv_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]

        mixed_layer.self_attn.qkv_proj = fused_qkv_proj

        mixed_layer.self_attn.o_proj = layer.self_attn.o_proj

        weights = [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight]
        fused_upgate_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                        weights[0].shape[1],
                                        sum([_.shape[0] for _ in weights]),
                                        bias=False)
        cur = 0
        for w in weights:
            fused_upgate_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.mlp.upgate_proj = fused_upgate_proj

        mixed_layer.mlp.down_proj = layer.mlp.down_proj

        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)

    finetune.quantize_finetune_decoder_layer(mixed_layer,
                                             [('self_attn.qkv_proj', 'qkv'),
                                              ('self_attn.o_proj', 'o'),
                                              ('mlp.upgate_proj', 'up'),
                                              ('mlp.down_proj', 'down')], idx,
                                             cb, args, device, pre_orig_emb,
                                             orig_emb)

    torch.save(
        {
            'input_layernorm':
            mixed_layer.input_layernorm.weight,
            'post_attention_layernorm':
            mixed_layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')
    del mixed_layer


def move_fn(in_q, async_copy_speed):
    """异步文件传输函数"""
    # async copy to avoid slow disk
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')


def forward_layer(layer, position_ids, attention_mask, bs, device, in_q,
                  out_q):
    """前向传播层函数"""
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)
    done_qkv = utils.register_H_hook(layer.self_attn.q_proj, device)
    done_o = utils.register_H_hook(layer.self_attn.o_proj, device)
    done_up = utils.register_H_hook(layer.mlp.up_proj, device)
    done_down = utils.register_H_hook(layer.mlp.down_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            out_q.put({
                'qkv': done_qkv(),
                'o': done_o(),
                'up': done_up(),
                'down': done_down()
            })
            return

        assert len(dev_emb) % bs == 0
        for i in range(len(dev_emb) // bs):
            dev_emb[i * bs:(i + 1) * bs] = layer(
                dev_emb[i * bs:(i + 1) * bs].to(device),
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False)[0].cpu()


def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
    """累积 Hessian 矩阵"""
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape,
                                      dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape,
                                       dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    keys = list(Hs.keys())

    for key in Hs:
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" if args.scratch_path is not None else f"{args.hessian_path}/{transformer_layer_index}_{key}.pt"
        torch.save(
            {
                'flatH': utils.sym_to_flat(Hs[key].to(torch.float32)),
                'mu': mus[key].to(torch.float32),
                'n': Hs[key].shape[0],
                'ct': cts[key]
            }, save_path)
        if args.scratch_path is not None:
            move_q.put(
                (f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
                 f"{args.hessian_path}/{transformer_layer_index}_{key}.pt"))

    del Hs, mus, cts, out


def compute_hessian_offline(calibrate,args):  
    """离线计算 Hessian 矩阵"""
    torch.set_grad_enabled(False)
    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype="auto",
                                                 low_cpu_mem_usage=True)
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.isfile(f"{args.hessian_path}/dev_activations.pt"):
        print("loading cached dataset...")
        loaded_dev_activations = torch.load(
            f"{args.hessian_path}/dev_activations.pt")  
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(
            f"loaded cached dataset from {loaded_dev_activations['timestamp']}"
        )
    elif calibrate is not None and len(calibrate) > 0:
        devset = torch.cat(calibrate, dim=0)
        dev_emb = model.model.embed_tokens(devset)
        after_layer = -1
        print("loaded calibrate data!")
    else:
        print("loading dataset...")
        devset = utils.sample_rp1t(tokenizer,
                                   args.devset_size,
                                   args.ctx_size,
                                   nproc=args.sample_proc)
        dev_emb = model.model.embed_tokens(devset)
        after_layer = -1
        print("loaded dataset!")

    print(f"dev_emb dtype: {dev_emb.dtype}")
    dev_emb.share_memory_()

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
    if hasattr(model.config, 'sliding_window'):
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size],
            0,
            sliding_window=model.config.sliding_window)
    else:
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),
            dev_emb[0:args.batch_size], 0)

    if args.scratch_path is not None:
        move_q = mp.Queue()
        move_p = mp.Process(target=move_fn,
                            args=(move_q, args.async_copy_speed))
        move_p.start()
    else:
        move_q = None

    for transformer_layer_index in range(len(model.model.layers)):
        if (transformer_layer_index <= after_layer):
            print(
                f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
            )
            continue

        transformer_layer = model.model.layers[transformer_layer_index]
        # check that there are four layers, as expected
        assert (len([
            m for m in transformer_layer.modules()
            if isinstance(m, torch.nn.Linear)
        ]) == 7)

        chunk_size = min(args.chunk_size, len(dev_emb))
        ngpus = min(torch.cuda.device_count(), len(dev_emb) // chunk_size)

        manager = mp.get_context('spawn').Manager()
        in_q = manager.Queue()
        out_q = manager.Queue()

        accumulate_proc = mp.Process(target=accumulate,
                                     args=(out_q, move_q, ngpus, args,
                                           transformer_layer_index))
        accumulate_proc.start()

        forward_procs = []
        for i in range(ngpus):
            p = mp.Process(target=forward_layer,
                           args=(transformer_layer, position_ids,
                                 attention_mask, args.batch_size, i, in_q,
                                 out_q))
            p.start()
            forward_procs.append(p)

        assert len(
            dev_emb
        ) % args.batch_size == 0 and chunk_size % args.batch_size == 0
        i = 0
        while i < len(dev_emb):
            next = min(i + chunk_size, len(dev_emb))
            in_q.put(dev_emb[i:next])
            i = next

        for i in range(ngpus):
            in_q.put(None)

        for p in forward_procs:
            p.join()

        accumulate_proc.join()

        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        if args.save_activations and (
                transformer_layer_index % args.act_save_rate == 0 or \
                transformer_layer_index == len(model.model.layers) - 1):
            if args.scratch_path is not None:
                if os.path.exists(f'{args.scratch_path}/dev_activations.pt'):
                    print('not saving layer since disk is too slow')
                else:
                    torch.save(
                        {
                            'dev_emb': dev_emb,
                            'after_layer': transformer_layer_index,
                            'timestamp': str(datetime.datetime.now())
                        }, f'{args.scratch_path}/dev_activations.pt')
                    move_q.put((f'{args.scratch_path}/dev_activations.pt',
                                f'{args.hessian_path}/dev_activations.pt'))
            else:
                torch.save(
                    {
                        'dev_emb': dev_emb,
                        'after_layer': transformer_layer_index,
                        'timestamp': str(datetime.datetime.now())
                    }, f'{args.hessian_path}/dev_activations.pt')  



        print(f"done processing layer {transformer_layer_index}")

    if args.scratch_path is not None:
        move_q.put(None)
        move_p.join()


def get_qwen_save_fn(args):
    """获取 Qwen 模型保存函数"""

    def save_fn(shard_model):
        ct = 0
        for i in range(len(shard_model.shards)):
            for j in range(len(shard_model.shards[i].layers)):
                shard = shard_model.shards[i].layers[j]
                utils.save_susv(shard.self_attn.qkv_proj,
                                f'{args.ckpt_path}/{ct}_qkv.pt')
                utils.save_susv(shard.self_attn.o_proj,
                                f'{args.ckpt_path}/{ct}_o.pt')
                utils.save_susv(shard.mlp.upgate_proj,
                                f'{args.ckpt_path}/{ct}_up.pt')
                utils.save_susv(shard.mlp.down_proj,
                                f'{args.ckpt_path}/{ct}_down.pt')
                torch.save(
                    {
                        'input_layernorm':
                        shard.input_layernorm.weight,
                        'post_attention_layernorm':
                        shard.post_attention_layernorm.weight,
                    }, f'{args.ckpt_path}/{ct}_layernorm.pt')
                glog.info(f'wrote layer {ct}')
                ct += 1
        torch.save(
            {
                'lm_head': shard_model.output_layer[1].weight,
                'norm': shard_model.output_layer[0].weight,
            }, f'{args.ckpt_path}/lmhead.pt')

    return save_fn


def qwen_arg_fn(output, args, kwargs):
    """Qwen 参数函数"""
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    """获取嵌入函数"""
    return args[0]


def finetune_e2e(calibrate,args):
    """端到端微调"""
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    devset = torch.cat(calibrate, dim=0)
    orig_model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                      torch_dtype='auto',
                                                      device_map='auto',
                                                      low_cpu_mem_usage=True)
    orig_logits = utils.calculate_logits(orig_model, devset, args.batch_size)
    orig_logits = orig_logits[:, :-1].contiguous().softmax(dim=-1).float()

    del orig_model
    utils.clean()

    quant_model = model_from_hf_path(args.hf_path,
                                     use_cuda_graph=False,
                                     use_flash_attn=False,
                                     device_map=None)[0].cpu()
    emb = quant_model.model.embed_tokens(devset)
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.ft_bs, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.ft_bs, args.ctx_size), emb[:args.ft_bs], 0)

    # construct shards
    nshards = torch.cuda.device_count(
    ) if args.ft_nshards < 0 else args.ft_nshards
    nlayers = len(quant_model.model.layers)
    shards = [nn.ModuleList([]) for _ in range(nshards)]
    for i in range(nshards):
        for j in range(int(nlayers * i / nshards),
                       int(nlayers * (i + 1) / nshards)):
            shards[i].append(quant_model.model.layers[j])
        shards[i] = {'device': i, 'arg_fn': qwen_arg_fn, 'shard': shards[i]}
    output_layer = {
        'layer': nn.Sequential(quant_model.model.norm, quant_model.lm_head),
        'fn': get_emb
    }

    shard_model = utils.ShardTransformer(shards, output_layer,
                                         args.ft_grad_ckpt, args.ft_train_mode)
    shard_model.manifest(emb[:args.ft_bs],
                         position_ids=position_ids,
                         attention_mask=attention_mask)
    utils.clean()

    torch.set_grad_enabled(True)
    finetune.finetune_susv_e2e(shard_model, orig_logits, emb, position_ids,
                               attention_mask, get_qwen_save_fn(args), args)


def hfize_model(quantized_path,hf_output_path):
    """转换为 HuggingFace 格式"""
    torch.set_grad_enabled(False)
    assert os.path.exists(quantized_path)
    saved_config = torch.load(os.path.join(quantized_path, 'config.pt'))
    model_config = saved_config['model_config']

    codebook_id = codebook.get_id(model_config.quip_params['codebook'])
    codesz = model_config.quip_params['codesz']

    tokenizer = AutoTokenizer.from_pretrained(model_config._name_or_path)

    model = Qwen2ForCausalLM.from_pretrained(model_config._name_or_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config).half()
    cpu = torch.device('cpu')

    if os.path.exists(f'{quantized_path}/lmhead.pt'):
        lmhead_data = torch.load(f'{quantized_path}/lmhead.pt',
                                    map_location=cpu)
        model.lm_head.weight.copy_(lmhead_data['lm_head'])
        model.model.norm.weight.copy_(lmhead_data['norm'])

    for ii in range(len(model.model.layers)):
        layer = model.model.layers[ii]

        ln_data = torch.load(f'{quantized_path}/{ii}_layernorm.pt',
                                map_location=cpu)
        layer.input_layernorm.weight.copy_(ln_data['input_layernorm'])
        layer.post_attention_layernorm.weight.copy_(
            ln_data['post_attention_layernorm'])

        saved_layer = torch.load(f'{quantized_path}/{ii}_qkv.pt',
                                map_location=cpu)
        for i in range(len(saved_layer['scales'])):
            layer.self_attn.qkv_proj.fuse_scales[i].copy_(
                saved_layer['scales'][i])
        utils.unpack_quip(layer.self_attn.qkv_proj, saved_layer, codebook_id,
                          codesz)

        saved_layer = torch.load(f'{quantized_path}/{ii}_o.pt',
                                map_location=cpu)
        utils.unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id,
                          codesz)

        saved_layer = torch.load(f'{quantized_path}/{ii}_up.pt',
                                map_location=cpu)
        for i in range(len(saved_layer['scales'])):
            layer.mlp.upgate_proj.fuse_scales[i].copy_(
                saved_layer['scales'][i])
        utils.unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id,
                          codesz)

        saved_layer = torch.load(f'{quantized_path}/{ii}_down.pt',
                                map_location=cpu)
        utils.unpack_quip(layer.mlp.down_proj, saved_layer, codebook_id,
                          codesz)
        glog.info(f'loaded layer {ii} down')

    glog.info(f'saving model...')
    model.save_pretrained(hf_output_path, safe_serialization=True)

    del model

    model, _ = model_from_hf_path(hf_output_path, use_cuda_graph=False)

    glog.info('successfully loaded hfized model')

    glog.info('generating some text...')

    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                             attention_mask=inputs['attention_mask'].cuda(),
                             max_new_tokens=64,
                             return_dict_in_generate=True)
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)
    glog.info(f'elapsed: {time.time() - start}')
    return model


def quantize_main(calibrate,args):
    """量化主函数"""
    torch.set_grad_enabled(False)

    cb = codebook.get_codebook(args.codebook)

    model = AutoModelForCausalLM.from_pretrained(args.base_model,   
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True)

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'resid_scale_override': args.resid_scale_override,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

    devset = torch.cat(calibrate, dim=0)
    print("loaded calibrate data!")
    glog.info('loaded dataset and devset')

    nproc = torch.cuda.device_count()
    orig_emb_cache = [model.model.embed_tokens(devset)]
    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0)

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.model.layers)):
        glog.info(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device].join()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1].join()
        utils.clean()

        # 检查是否需要进行微调
        if args.ft_epochs > 0:
            st = time.time()
            position_ids = position_ids.to(cur_device)
            attention_mask = attention_mask.to(cur_device)
            model.model.layers[i].to(cur_device)
            for j in range(args.devset_size // args.batch_size):
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](
                        orig_emb_cache[cur_device][
                            args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu()
            model.model.layers[i].cpu()
            orig_msv = orig_emb_cache[cur_device].float().norm(
            )**2 / orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm(
            )**2 / orig_emb_cache[cur_device + 1].numel()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()
            glog.info(
                'computed original embedding for layer {} in {}s, pre msv {}, post msv {}'
                .format(i,
                        time.time() - st, orig_msv, target_msv))

        proc_list[cur_device] = mp.Process(target=quantize_qwen_layer,
                                           args=(
                                               model.model.layers[i],
                                               i,
                                               cb,
                                               args,
                                               cur_device,
                                               orig_emb_cache[cur_device],
                                               orig_emb_cache[cur_device + 1],
                                               all_config['model_config'],
                                           ))
        proc_list[cur_device].start()

        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        p.join()


def quip_sharp_main(calibrate, kwargs):
    """主函数 - 完整的量化微调流程"""
    glog.info("开始完整的 Qwen 模型量化微调流程")
    base_model_config,quant_config = kwargs['base_model'],kwargs['quant']
    main_args, hessian_args, quantize_args, finetune_args, data_args = quant_config['main'], quant_config['hessian'], quant_config['quantize'], quant_config['finetune'],quant_config['data']
    main_args['save_path'],main_args['ckpt_path'],main_args['base_model'],main_args['ctx_size'],main_args['devset_size'] = main_args['quantized_path'],main_args['quantized_path'],base_model_config['path'],data_args['seqlen'],data_args['nsamples']
    # 设置随机种子
    torch.manual_seed(main_args.seed)
    random.seed(main_args.seed)
    numpy.random.seed(main_args.seed)
    
    # 创建保存目录
    os.makedirs(main_args.save_path, exist_ok=True)
    os.makedirs(main_args.hessian_path, exist_ok=True)
    os.makedirs(main_args.quantized_path, exist_ok=True)
    os.makedirs(main_args.ckpt_path, exist_ok=True)
    os.makedirs(main_args.hf_path, exist_ok=True)

    # 步骤1: Hessian 计算（如果启用）
    if main_args.enable_hessian:
        glog.info("=== 步骤1: Hessian 计算 ===")
        args = SimpleNamespace(**{**vars(main_args),**vars(hessian_args)})
        compute_hessian_offline(calibrate,args)

    # 步骤2: 量化（如果启用）
    if main_args.enable_quantize:
        glog.info("=== 步骤2: 模型量化 ===")
        args = SimpleNamespace(**{**vars(main_args),**vars(quantize_args)})
        quantize_main(calibrate,args)
        
    # 步骤3: 端到端微调（如果启用）
    if main_args.enable_finetune:
        glog.info("=== 步骤3: 端到端微调 ===")
        hfize_model(main_args.quantized_path, main_args.hf_path)
        args = SimpleNamespace(**{**vars(main_args),**vars(finetune_args)})
        finetune_e2e(calibrate,args)  

    
    # 步骤4: HuggingFace 格式转换（如果启用）
    if main_args.enable_hfize:
        glog.info("=== 步骤4: HuggingFace 格式转换 ===")
        quantmodel = hfize_model(main_args.ckpt_path, main_args.hf_ft_path)
    
    glog.info("完整的量化微调流程完成!")
    
    # 返回最终的模型
    return quantmodel


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    # 主参数解析器
    main_parser = argparse.ArgumentParser(description='完整的 Qwen 模型量化微调脚本')
    # 基础参数
    main_parser.add_argument('--seed', default=0, type=int)
    main_parser.add_argument('--num_cpu_threads', default=8, type=int)
    # 路径参数
    main_parser.add_argument('--base_model', type=str, default='/home/xzy/Llama-2-7b-hf')
    main_parser.add_argument('--save_path', type=str, default='/home/xzy/quip-sharp/outputs_q')
    main_parser.add_argument('--hessian_path', type=str, default='/home/xzy/quip-sharp/hessians/llama2_7b')
    main_parser.add_argument('--quantized_path', type=str, default='/home/xzy/quip-sharp/outputs_q')
    main_parser.add_argument('--hf_path', type=str, default='/home/xzy/quip-sharp/outputs_q_hf')
    main_parser.add_argument('--ckpt_path', type=str, default='/home/xzy/quip-sharp/outputs_q')
    main_parser.add_argument('--hf_ft_path', type=str, default='/home/xzy/quip-sharp/outputs_q_hf_ft_hf')

    # 功能开关
    main_parser.add_argument('--enable_hessian', action='store_true', help='启用 Hessian 计算')
    main_parser.add_argument('--enable_quantize', action='store_true', help='启用量化')
    main_parser.add_argument('--enable_finetune', action='store_true', help='启用微调')
    main_parser.add_argument('--enable_hfize', action='store_true', help='启用 HuggingFace 转换')

    # Hessian 计算参数解析器
    hessian_parser = argparse.ArgumentParser(add_help=False)
    hessian_parser.add_argument('--batch_size', default=4, type=int)
    hessian_parser.add_argument('--devset_size', default=6144, type=int)  # 增加到2048以支持8个GPU 
    hessian_parser.add_argument('--ctx_size', default=2048, type=int)
    hessian_parser.add_argument('--scratch_path', default=None, type=str)
    hessian_parser.add_argument('--chunk_size', default=256, type=int)  # 每个GPU处理256个样本
    hessian_parser.add_argument('--async_copy_speed', default=-1, type=int)
    hessian_parser.add_argument('--act_save_rate', default=4, type=int)
    hessian_parser.add_argument('--save_activations', action='store_true')
    hessian_parser.add_argument('--sample_proc', default=8, type=int)  # 增加到8个进程以匹配GPU数量

    # 量化参数解析器
    quantize_parser = argparse.ArgumentParser(add_help=False)
    quantize_parser.add_argument('--batch_size', default=16, type=int)  # 减小batch_size以降低内存使用
    quantize_parser.add_argument('--devset_size', default=384, type=int)  # 减小devset_size以降低内存使用
    quantize_parser.add_argument('--ctx_size', default=4096, type=int)  # 减小上下文长度以降低内存使用
    quantize_parser.add_argument('--sigma_reg', default=1e-2, type=float)
    quantize_parser.add_argument('--sigma_reg2', default=1e-2, type=float)
    quantize_parser.add_argument('--incoh_mode', default='had', type=str, choices=['had', 'kron'])
    quantize_parser.add_argument('--lora_rank', default=0, type=int, help='if <=0 then turned off')
    quantize_parser.add_argument('--scale_override', default=0.9, type=float)
    quantize_parser.add_argument('--resid_scale_override', default=-1, type=float)
    quantize_parser.add_argument('--codebook', type=str, default='E8P12')
    quantize_parser.add_argument('--quip_tune_iters', default=10, type=int)
    quantize_parser.add_argument('--use_fp64', action='store_true')
    quantize_parser.add_argument('--full_svd', action='store_true')
    quantize_parser.add_argument('--no_use_buffered', action='store_true')
    quantize_parser.add_argument('--rescale_WH', action='store_true')
    quantize_parser.add_argument('--sample_proc', default=1, type=int)
    quantize_parser.add_argument('--lowmem_ldlq', action='store_true')
    quantize_parser.add_argument('--ft_lr', default=5e-5, type=float)
    quantize_parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
    quantize_parser.add_argument('--ft_bs', default=4, type=int)
    quantize_parser.add_argument('--ft_update_freq', default=2, type=int)
    quantize_parser.add_argument('--ft_epochs', default=5, type=int)
    quantize_parser.add_argument('--ft_valid_freq', default=1, type=int)
    quantize_parser.add_argument('--ft_valid_size', default=128, type=float)
    quantize_parser.add_argument('--ft_early_stop', default=3, type=int)
    quantize_parser.add_argument('--ft_train_mode', action='store_true')
    quantize_parser.add_argument('--ft_grad_ckpt', action='store_true')

    # 微调参数解析器
    finetune_parser = argparse.ArgumentParser(add_help=False)
    finetune_parser.add_argument('--batch_size', default=16, type=int)
    finetune_parser.add_argument('--devset_size', default=384, type=int)
    finetune_parser.add_argument('--ctx_size', default=4096, type=int)
    finetune_parser.add_argument('--sample_proc', default=1, type=int)
    finetune_parser.add_argument('--ft_lr', default=1e-5, type=float)
    finetune_parser.add_argument('--ft_susv_lr', default=1e-4, type=float)
    finetune_parser.add_argument('--ft_bs', default=1, type=int)
    finetune_parser.add_argument('--ft_update_freq', default=2, type=int)
    finetune_parser.add_argument('--ft_epochs', default=8, type=int)
    finetune_parser.add_argument('--ft_valid_freq', default=1, type=int)
    finetune_parser.add_argument('--ft_valid_size', default=128, type=float)
    finetune_parser.add_argument('--ft_early_stop', default=3, type=int)
    finetune_parser.add_argument('--ft_train_mode', action='store_true')
    finetune_parser.add_argument('--ft_grad_ckpt', action='store_true')
    finetune_parser.add_argument('--ft_nshards', default=-1, type=int)


    main_args = main_parser.parse_args()
    hessian_args = hessian_parser.parse_args()
    quantize_args = quantize_parser.parse_args()
    finetune_args = finetune_parser.parse_args()
    
    # 如果没有指定任何功能，默认启用所有功能
    if not any([main_args.enable_hessian, main_args.enable_quantize, 
                main_args.enable_finetune, main_args.enable_hfize]):
        main_args.enable_hessian = True
        main_args.enable_quantize = True
        main_args.enable_finetune = True
        main_args.enable_hfize = True
    
    final_model = quip_sharp_main(main_args, hessian_args, quantize_args, finetune_args)
    
    # 输出最终模型信息
    if final_model is not None:
        glog.info(f"最终模型类型: {type(final_model)}")
        glog.info("量化微调流程成功完成!")
