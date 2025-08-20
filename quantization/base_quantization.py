import torch
from torch.utils.data import DataLoader
import lm_eval
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logger  # Add required imports
from abc import ABCMeta, abstractmethod

from utils.load_dataset import BaseDataset
from utils.regaster import  QUANTIZATION_REGISTRY
from utils.load_model import BaseModel


class QuantizationBase(metaclass=ABCMeta):
    def __init__(self, config, **kwargs):
        self.dataset = None
        self.config = config
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
        self.model = None  # 子类加载具体模型
        self.tokenizer = None  # 子类加载具体tokenizer
        self.kwargs = kwargs
        self.baseDataset=BaseDataset(self.config["data"],self.config["base"]["seed"])


    def get_loader(self):
        # 加载数据，可以是本地加载也可以是联网下载，返回时处理好的数据
        self.dataset = self.baseDataset.get_calib_data()
        return self.dataset


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
    def contiguous_params(self):
        if self.model.mm_model is not None:
            for name, param in self.model.mm_model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

            for name, param in self.model.mm_model.named_buffers():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        else:
            for name, param in self.model.model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

            for name, param in self.model.model.named_buffers():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
    def copy_tokenizer(self, path):
        if self.model.tokenizer is not None:
            self.model.tokenizer.save_pretrained(path)
            logger.info('copy tokenizer done --')
        else:
            logger.info('no tokenizer, skip --')

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