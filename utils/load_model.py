#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model-quantification-tool 
@File    ：load_model.py
@Author  ：zxy
@Date    ：2025/8/20 15:12 
'''
import inspect
import json
import logging
import os
from typing import Dict, List, Optional, Union
from collections import defaultdict

import accelerate
import torch
import transformers
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logger  # Add required imports
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers, cached_file

from quantization.auto_gptq import BaseQuantizeConfig
from quantization.auto_gptq.modeling._utils import find_layers, make_quant, get_checkpoints, \
    make_sure_no_tensor_in_meta_device, simple_dispatch_model
from quantization.auto_gptq.utils.accelerate_utils import load_checkpoint_in_model
from quantization.auto_gptq.utils.import_utils import dynamically_import_QuantLinear
from quantization.auto_gptq.utils.marlin_utils import _validate_marlin_device_support, _validate_marlin_compatibility, \
    prepare_model_for_marlin_load




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
        self.use_cpu_to_save_cuda_mem_for_catcher = self.config.model.get('use_cpu_to_save_cuda_mem_for_catcher',
                                                                          False)  # noqa
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.torch_dtype = self.config.model.torch_dtype
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.model_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
    def from_quantized(
            cls,
            model_name_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = None,
            low_cpu_mem_usage: bool = False,
            use_triton: bool = False,
            use_qigen: bool = False,
            use_marlin: bool = False,
            torch_dtype: Optional[torch.dtype] = None,
            inject_fused_attention: bool = False,
            inject_fused_mlp: bool = False,
            use_cuda_fp16: bool = True,
            quantize_config: Optional[BaseQuantizeConfig] = None,
            model_basename: Optional[str] = None,
            use_safetensors: bool = True,
            trust_remote_code: bool = False,
            warmup_triton: bool = False,
            trainable: bool = False,
            disable_exllama: Optional[bool] = None,
            disable_exllamav2: bool = False,
            use_tritonv2: bool = False,
            checkpoint_format: Optional[str] = None,
            **kwargs,
    ):


        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        # == step1: prepare configs and file names == #
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )
        quantize_config.model_name_or_path = model_name_or_path
        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]


        if model_basename is None:
            if quantize_config.model_file_base_name:
                possible_model_basenames = [quantize_config.model_file_base_name]
            else:
                possible_model_basenames = [
                    f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
                    "model",
                ]
        else:
            possible_model_basenames = [model_basename]
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(model_name_or_path=model_name_or_path,
                                                                                 extensions=extensions,
                                                                                 possible_model_basenames=possible_model_basenames,
                                                                                 **cached_file_kwargs)
        quantize_config.model_file_base_name = true_model_basename

