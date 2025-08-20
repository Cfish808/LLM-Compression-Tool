import torch
from torch.utils.data import DataLoader
import lm_eval
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # Add required imports
from abc import ABCMeta, abstractmethod
from .utils.regaster import  QUANTIZATION_REGISTRY



@QUANTIZATION_REGISTRY
class QuantizationBase(metaclass=ABCMeta):
    def __init__(self, config, **kwargs):
        self.config = config
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
        self.model = None  # 子类加载具体模型
        self.tokenizer = None  # 子类加载具体tokenizer
        self.kwargs = kwargs

    def get_loader(self, dataset, batch_size=1, shuffle=False):
        # 加载数据，可以是本地加载也可以是联网下载，返回时处理好的数据
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  
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

    @abstractmethod
    def quantize(self):
        # 量化方法的实现
        # 这里以GPTQ为例，实际可根据需求替换
        # 假设self.model是transformers模型
        # 这里只做简单的8位量化
        pass

    def eval(self, task="lambada", device="cuda"):
        # 评估方法的实现
        # 评估使用lm_eval进行测试
        results = lm_eval.simple_evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task,
            device=device
        )
        return results

    def save(self, save_path="quantized_model"):
        # 保存模型的方法
        # 将量化后的模型进行保存处理，默认GPTQ，保存的模型需要是可以直接transformers加载的格式
        # self.model.save_pretrained(save_path)
        # self.tokenizer.save_pretrained(save_path)
        # print(f"模型已保存到 {save_path}")


        self.contiguous_params()
        if self.config.model.type in ['Llava', 'InternVL2', 'Mllama', 'Qwen2vl']:
            self.model.vlm_model.language_model = self.model.get_model()
            self.model.vlm_model.save_pretrained(save_path)
            self.copy_tokenizer(save_path)
        elif self.config.model.type in ['Qwen2Audio']:
            self.model.alm_model.language_model = self.model.get_model()
            self.model.alm_model.save_pretrained(save_path)
            self.copy_tokenizer(save_path)
        elif self.config.model.type in ['InternOmni']:
            self.model.avlm_model.language_model = self.model.get_model()
            self.model.avlm_model.save_pretrained(save_path)
            self.copy_tokenizer(save_path)
        else:
            self.model.get_model().save_pretrained(save_path)
            self.copy_tokenizer(save_path)
        print(f"模型已保存到 {save_path}")