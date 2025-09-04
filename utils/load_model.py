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

import accelerate
import torch
import transformers
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers, cached_file

from quantization.__init__ import QuantizedModule
from quantization.layers import LinearQuantHub


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
                torch_dtype=self.torch_dtype,
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