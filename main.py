import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group
import pdb
import logging

from transformers import AutoTokenizer, LlamaForCausalLM

from export import QLinear
from my_datasets import get_calibrate_loader
from quantization.gptq.GPTQQuantizer import LinearGPTQQuantizer
from quantization.layers import LinearQuantHub
from quantization.llama_seq import llama_sequential
from quantization.__init__ import QuantizedModule
from quantization.util import replace_module
from utils.benchmark import Benchmark
from utils.load_dataset import BaseDataset
from utils.load_model import BaseModel


def deploy_all_modality(blockwise_opts, quant_format):
    for blockwise_opt in blockwise_opts:
        blockwise_opt.deploy(quant_format)
def load_model(model_name_or_path):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto')
    return model



def transform_layers(module):
    if isinstance(module, QuantizedModule):

        if isinstance(module.default_quantizer, LinearGPTQQuantizer):
            return QLinear.pack_from_gptq_quantizer(module.default_quantizer)

    return module


def main(config):
    basemodel = BaseModel(config)
    tokenizer = basemodel.build_tokenizer()
    model=basemodel.build_model()

    new_config = {'model_path': '/ssd/yejinyu/llama2_7b/Llama-2-7b-ms/', 'algo': 'gptq', 'wbit': 4, 'abit': 16,
                  'w_groupsize': 128, 'w_qtype': 'per_group', 'benchmark': 'ceval', 'num_calibrate': 1, 'num_shot': 0,
                  'calibrate_name': 'c4', 'seqlen': 2048, 'device': 'cuda', 'offload': 'cpu',
                  'skip_layers': ['l', 'm', '_', 'h', 'e', 'a', 'd'], 'block_sequential': False,
                  'layer_sequential': False, 'save': '/ssd/yejinyu/llama2_7b/output/llama2_7b_miom.pt'}
    # new_model=basemodel.replace_module(model, exclude_layers=config.skip_layers, include_layers=['.*'])
    # calibrate_config = {'name': 'c4', 'nsamples': 1, 'seqlen': 2048}
    # calibrate = get_calibrate_loader(tokenizer=tokenizer, calibrate_config=calibrate_config)
    calibrate=torch.load("calibrate.pt")
    config["algo"] = "gptq"
    del config["model"]
    new_model=llama_sequential(model=model,data=calibrate,**new_config)
    new_model = new_model.to("cuda")
    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {tokenizer}')
    benchmark = Benchmark()
    # results_ceval = benchmark.eval_ppl(new_model, tokenizer, nsamples='all', test_datasets=['wikitext2'])
    # results_ceval = benchmark.eval_ceval(new_model, tokenizer,model_type="llama",subject="hm" )
    logging.info("\n转换前:")
    # logging.info(results_ceval)
    # if args.save:
    save_path = "/home/yejinyu/llama2_7b/output/llama27b_miom"
    # new_model.generation_config.do_sample = True
    # new_model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)

    save_path2 = "/ssd/yejinyu/llama2_7b/output/llama27b_model_quant_tool_4.pt"
    model = replace_module(new_model, LinearQuantHub, transform_layers, display=True)
    torch.save(model, save_path2)

    results_ceval = benchmark.eval_ppl(model, tokenizer, nsamples='all', test_datasets=['wikitext2'])
    logging.info("\n转换后的模型:")
    logging.info(results_ceval)



    # eval_list = get_eval_list(model, config)
    # eval_model(model, None, eval_list, eval_pos='pretrain')
    #
    # blockwise_opts = []
    # modalities=["language"]
    # modality_configs = config.quant
    # modality_configs.modality=["language"]
    #
    #
    # for modality, modality_config in zip(modalities, modality_configs):
    #     model.set_modality(modality)
    #
    #
    #     dataset = BaseDataset(
    #         tokenizer,  model.batch_process ,config
    #     )
    #     calib_data = dataset.get_calib_data()
    #     model.collect_first_block_input(calib_data)
    #     del calib_data
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #
    #     # blockwise_opt = ALGO_REGISTRY[modality_config.method](
    #     #     model,
    #     #     modality_config,
    #     #     model.get_first_block_input(),
    #     #     model.get_padding_mask(),
    #     #     config,
    #     # )
    #     # blockwise_opt.run_block_loop()
    #     blockwise_opts.append(blockwise_opt)
    #     dist.barrier()
    #
    # eval_model(model, blockwise_opts, eval_list, eval_pos='transformed')
    #
    # if 'save' in config and config.save.get('save_fake', False):
    #     deploy_all_modality(blockwise_opts, 'fake_quant')
    #     blockwise_opt.save_model(save_fake_path)
    #
    #
    #
    # dist.barrier()


def mkdirs(path):
    #
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    logger.add(sys.stdout, level='INFO')
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="/home/yejinyu/model-quantification-tool/config/llama_gptq.yml", type=str)
    args = parser.parse_args()
    import os

    # 设置 HuggingFace 的缓存和镜像源
    os.environ['HF_HOME'] = '/home/yejinyu/huggingface_3_copy'
    os.environ['HF_DATASETS_CACHE'] = '/home/yejinyu/huggingface_3_copy'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')


    save_fake_path = os.path.join(config.save.save_path, 'output_models')
    mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    # dist.barrier()

    main(config)


    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f'llmc_duration_time: {llmc_duration_time} s')
    logger.info('--- llmc finished ---')


# export HF_HOME=/home/yejinyu/huggingface_3_copy
#
# export HF_DATASETS_CACHE=/home/yejinyu/huggingface_3_copy
#
# export HF_ENDPOINT=https://hf-mirror.com


# export PYTHONPATH=/ssd/yejinyu/MI-optimize-main:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME=/ssd/yejinyu/huggingface_3_copy
# export HF_DATASETS_CACHE=/ssd/yejinyu/huggingface_3_copy