#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model-quantification-tool 
@File    ：load_model.py
@Date    ：2025/8/20 15:12
'''
import inspect
import json
import logging
import os
import re
from typing import Dict, List, Optional, Union
from collections import defaultdict
from packaging import version

import accelerate
import torch
import transformers
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,LlamaTokenizer,LlamaTokenizerFast,PretrainedConfig,PreTrainedModel,PreTrainedTokenizerBase
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers, cached_file

from quantization.__init__ import QuantizedModule
from quantization.layers import LinearQuantHub


from os.path import exists, join, isdir
from peft.tuners.lora import LoraLayer
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)



import importlib
def get_checkpoints(model_name_or_path: str, extensions: List[str], possible_model_basenames: List[str], **cached_file_kwargs):
    """
    Retrives (and if necessary downloads from Hugging Face Hub) the model checkpoint. Sharding is supported. All the `possible_model_basenames` (e.g. `["model", "model-4bit-gptq"]`) will be explored over all `extensions` (e.g. `[".bin", ".safetensors"]`).
    """
    searched_files = []
    resolved_archive_file = None
    true_model_basename = None

    if os.path.isdir(model_name_or_path):
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                searched_files.append(shard_index_name)
                possible_index_file = os.path.join(model_name_or_path, shard_index_name)
                if os.path.isfile(possible_index_file):
                    # The model is sharded over several checkpoints.
                    possible_model_basename = possible_index_file.replace(ext + ".index.json", "")
                    return True, possible_index_file, possible_model_basename
                else:
                    model_save_name = os.path.join(model_name_or_path, possible_model_basename)
                    searched_files.append(possible_model_basename + ext)
                    if os.path.isfile(model_save_name + ext):
                        resolved_archive_file = model_save_name + ext
                        return False, resolved_archive_file, possible_model_basename
    else:
        temp = None
        for ext in extensions:
            for possible_model_basename in possible_model_basenames:
                shard_index_name = possible_model_basename + ext + ".index.json"
                shard_index = cached_file(
                    model_name_or_path,
                    shard_index_name,
                    **cached_file_kwargs,
                )
                searched_files.append(shard_index_name)
                if shard_index is not None:
                    # The model is sharded over several checkpoints.
                    with open(str(shard_index)) as f:
                        index_json = json.load(f)
                        # Download the shards from the index.json.
                        shards = list(set(index_json["weight_map"].values()))
                        for shard in shards:
                            resolved_archive_file = cached_file(
                                model_name_or_path,
                                shard,
                                **cached_file_kwargs,
                            )
                        return True, shard_index, possible_model_basename
                else:
                    resolved_archive_file = cached_file(
                        model_name_or_path,
                        possible_model_basename + ext,
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is None:
                        resolved_archive_file = temp
                    searched_files.append(possible_model_basename + ext)
                    if resolved_archive_file is not None:
                        temp = resolved_archive_file
                        return False, resolved_archive_file, possible_model_basename

    if resolved_archive_file is None:
        raise FileNotFoundError(
            f"Could not find a model in {model_name_or_path} with a name in {', '.join(searched_files)}. Please specify the argument model_basename to use a custom file name."
        )

    return False, resolved_archive_file, true_model_basename


class BaseModel():
    def __init__(self, config,device_map=None,use_cache=False):
        self.vision_model = None
        self.model_config = None
        self.config = config

        self.model_type = self.config.base_model.type
        self.model_path = self.config.base_model.path
        self.torch_dtype = self.config.base_model.torch_dtype
        self.tokenizer_mode = self.config.base_model.get('tokenizer_mode', 'fast')
        self.model = None  # 子类加载具体模型
        self.tokenizer = None  # 子类加载具体tokenizer
        self.use_cache = use_cache
        self.device_map = device_map

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False

        if self.model_type=="auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model_config,
                device_map=self.device_map,
                trust_remote_code=True,
                torch_dtype=self.model_config.torch_dtype,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.model_config,
                device_map=self.device_map,
                trust_remote_code=True,
                # torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        self.model.eval()
        return self.model


    def build_tokenizer(self):
        if self.model_type not in ['Vit', 'WanT2V', 'WanI2V']:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=True
            )
            if 'Intern' in self.model_type:
                self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None
        return  self.tokenizer

    def replace_module(self,model, module_type=torch.nn.Linear, new_module_type=LinearQuantHub, exclude_layers=[],
                       include_layers=['.*'], display=False):
        if display:
            def count_children(module, name=''):
                count = 0
                for child_name, mod in list(module.named_children()):
                    if any(re.fullmatch(pat, name + child_name) for pat in include_layers):
                        if any(re.fullmatch(pat, name + child_name) for pat in exclude_layers):
                            continue
                        if isinstance(mod, module_type):
                            count += 1
                        else:
                            count += count_children(mod, name + child_name + '.')
                return count

            count = count_children(model, name='')
            bar = tqdm(total=count)

        # transform in-place
        def transform_children(module, name=''):
            for child_name, mod in list(module.named_children()):
                if any(re.fullmatch(pat, name + child_name) for pat in include_layers):
                    if any(re.fullmatch(pat, name + child_name) for pat in exclude_layers):
                        continue
                if isinstance(mod, module_type):
                    if display:
                        bar.update(1)
                    try:
                        setattr(module, child_name, new_module_type(mod, name=child_name))
                    except:
                        if new_module_type == "":
                            setattr(module, child_name, mod.core)
                        else:
                            setattr(module, child_name, new_module_type(mod))
                else:
                    transform_children(mod, name + child_name + '.')

        transform_children(model, name='')
        return model

def find_layers(module, layers, name=''):
    if isinstance(layers, list):
        layers = tuple(layers)
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir
    return None

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def find_all_linear_names(args, model):
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True

def prepare_model_for_int8_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        
    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    model.lm_head = model.lm_head.float()
    for _, param in model.named_parameters():
        if param.dtype == torch.float16:
            param = param.float()

    return model

def get_accelerate_model(args,method=''):
    args.model_name_or_path = args.path
    checkpoint_dir = get_last_checkpoint(args.output_dir)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'loading base model {args.model_name_or_path}...')
    if method in ['qlora','irlora']:
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token)

        if compute_dtype == torch.float16 and args.bits == 4:
            if torch.cuda.is_bf16_supported():
                print('='*80)
                print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
                print('='*80)
            
        if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
            compute_dtype = torch.bfloat16
            print('Intel XPU does not support float16 yet, so switching to bfloat16')
    
    elif method == "qalora":
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name_or_path,
            device_map='balanced',
            max_memory=max_memory,
            trust_remote_code=args.trust_remote_code,
            inject_fused_attention = False,
            inject_fused_mlp = False,
            use_triton=False,
            warmup_triton=False,
            trainable=True,
            model_basename="model",
            use_safetensors=True
        )
        model.model.quantize_config = model.quantize_config
        model.train()
    else:
        raise ValueError(f"Unknown method {method}")

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=True if method == "qalora" else False, 
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )


    DEFAULT_PAD_TOKEN = "[PAD]"

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                # "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id),
        })
    
    if method in ["qlora","irlora"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif method == "qalora":
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        raise ValueError(f"Unknown method {method}")


    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        print(f'adding LoRA modules...')
        if method == "qlora":
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
        elif method == "qalora":
            from auto_gptq.utils.peft_utils import get_gptq_peft_model, GPTQLoraConfig
            config = GPTQLoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                #target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_gptq_peft_model(model, config, auto_find_all_linears=True, train_mode=True)
        elif method == "irlora":
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            from utils.irlora_utils import get_my_model
            model_fp = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code)
            model = get_my_model(model, model_fp, args.blocksize2, args.tau_lambda, args.tau_n)
        else:
            raise ValueError(f"Unknown method {method}")

    if method == "qalora" and args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def prepare_model_for_training(
    model: "PreTrainedModel",
    use_gradient_checkpointing: Optional[bool] = True,
) -> "PreTrainedModel":
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/v0.2.0/src/peft/utils/other.py#L33
    """
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled
        print("Gradient checkpointing enabled.")
    return model


def load_model_and_tokenizer(model_args,finetuning_args,is_trainable: Optional[bool] = False):
    r"""
    Loads pretrained model and tokenizer.
    Support both training and inference.
    """
    # model_args.model_name_or_path = model_args.path
    # is_trainable = True if model_args.do_train else False
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **config_kwargs
    )

    model_to_load = model_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    # Fix tokenizer (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # Set model dtype
    if model_args.compute_dtype is not None: # for training
        setattr(config, "torch_dtype", model_args.compute_dtype)
    else: # for evaluation, priority: bf16 > fp16 > fp32
        from quantization.onebit.extras import infer_optim_dtype
        model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # Fix config (for Qwen)
    if getattr(config, "model_type", None) == "qwen":
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, getattr(config, "torch_dtype", None) == dtype)

    # Set RoPE scaling
    if model_args.rope_scaling is not None:
        if hasattr(config, "use_dynamic_ntk"): # for Qwen models
            if is_trainable:
                print("Qwen model does not support RoPE scaling in training.")
            else:
                setattr(config, "use_dynamic_ntk", True)
                setattr(config, "use_logn_attn", True)
                print("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"): # for LLaMA and Falcon models
            if is_trainable:
                if model_args.rope_scaling == "dynamic":
                    print(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )

                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    print("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            print("Using {} scaling strategy and setting scaling factor to {}".format(
                model_args.rope_scaling, scaling_factor
            ))

        else:
            print("Current model does not support RoPE scaling.")

   
    # from Xu Yuzhuang
    from transformers import BitLlamaForCausalLM
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
        from transformers.deepspeed import is_deepspeed_zero3_enabled
    model = BitLlamaForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )
    
    assert model_args.teacher_model_name_or_path is not None, "teacher model path is None!"
    teacher_config = AutoConfig.from_pretrained(model_args.teacher_model_name_or_path, **config_kwargs)
    model.teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.teacher_model_name_or_path,
        config=teacher_config,
        torch_dtype=torch.float16,
        **config_kwargs
    )

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    # Initialize adapters
    model = prepare_model_for_training(model=model) if is_trainable else model
    if finetuning_args.finetuning_type == "full" and is_trainable:
        model = model.float()
    model = model.train() if is_trainable else model.eval()
    
    model.kd_loss_scale = model_args.kd_loss_scale
    model.kd_alpha = model_args.kd_alpha
    model.kd_beta = model_args.kd_beta
    model.kd_gamma = model_args.kd_gamma
    # model.teacher_model = model.teacher_model.eval()
    for param in model.teacher_model.parameters():
        param.requires_grad = False

    # Prepare model for inference
    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.to(model_args.compute_dtype) if model_args.quantization_bit is None else model
    from quantization.onebit.extras import count_parameters
    trainable_params, all_param = count_parameters(model)
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    if not is_trainable:
        print("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer