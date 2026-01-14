#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model-quantification-tool 
@File    ：load_dataset.py
@Date    ：2025/8/20 17:12
'''
import random
import numpy as np
from datasets import load_dataset, load_from_disk

class BaseDataset():
    def __init__(self,tokenizer,batch_process, config,seed= 42):
        """
        config 示例：
        data:
            name: c4
            download: True
            n_samples: 128
            path: calib data path
            bs: 1
            seq_len: 2048
            preproc: c4_gptq
            seed: 42
        """
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        self.config = config
        self.dataset_name = self.config.get("name")
        self.seq_len= self.config.get("seq_len")
        self.download= self.config.get("download")
        self.n_samples = self.config.get("n_samples", 128)
        self.batch_size = self.config.get("bs", 1)
        self.path = self.config.get("path", None)
        self.seed = seed
        self.dataset = None

        random.seed(self.seed)
        np.random.seed(self.seed)

    def load(self):
        if self.download:
            if self.dataset_name == 'pileval':
                self.dataset = load_dataset(
                    'mit-han-lab/pile-val-backup', split='validation'
                )
            elif self.dataset_name == 'c4':
                self.dataset = load_dataset(
                    'allenai/c4',
                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'} if self.path is None else self.path,
                    split='train',
                )
            elif self.dataset_name == 'wikitext2':
                self.dataset = load_dataset(
                    'wikitext', 'wikitext-2-raw-v1', split='train'
                )
            elif self.dataset_name == 'ptb':
                self.dataset = load_dataset(
                    'ptb_text_only', 'penn_treebank', split='train'
                )
            elif self.dataset_name == 'ultrachat':
                self.dataset = load_dataset(
                    'HuggingFaceH4/ultrachat_200k', split='train_sft'
                )
            else:
                raise Exception(f'Not support {self.dataset_name} dataset.')
        else:
            self.dataset = load_from_disk(self.path)

        return self.dataset

    def get_calib_data(self):
        """
        从数据集中采样 n_samples * seq_len token，用于量化校准
        """
        if self.dataset is None:
            self.load()

        # 简单文本提取
        if self.dataset_name in ["pileval", "c4", "wikitext2", "ptb"]:
            texts = [item['text'] for item in self.dataset]
        elif self.dataset_name == "ultrachat":
            texts = [item['content'] for item in self.dataset]
        else:
            raise Exception(f'Unsupported extraction for {self.dataset_name}')

        # 采样 n_samples
        sampled_texts = random.sample(texts, min(self.n_samples, len(texts)))
        # 对 sampled_texts 做 tokenize 并截断到 seq_len
        tokenized = self.tokenizer(
            sampled_texts,
            max_length=self.seq_len,  # 限制到 seq_len
            truncation=True,  # 太长截断
            padding="max_length",  # 太短补 pad
            return_tensors="pt"  # 返回 PyTorch tensor
        )

        # 返回 batch 形式
        for i in range(0, len(sampled_texts), self.batch_size):
            yield {k: v[i:i + self.batch_size] for k, v in tokenized.items()}


