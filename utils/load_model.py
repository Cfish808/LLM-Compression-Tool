#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model-quantification-tool 
@File    ：load_model.py
@Author  ：zxy
@Date    ：2025/8/20 15:12 
'''
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # Add required imports

class BaseModel():
    def __init__(self, config,device_map=None,use_cache=False):
        self.model_config = None
        self.config = config
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


    def build_tokenizer(self):
        if self.model_type not in ['Vit', 'WanT2V', 'WanI2V']:
            assert self.tokenizer_mode in ['fast', 'slow']
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=True
            )
            if 'Intern' in self.model_type:
                self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None
