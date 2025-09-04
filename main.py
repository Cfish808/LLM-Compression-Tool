import argparse
import json
import sys
import time
import torch
import yaml
from easydict import EasyDict
from loguru import logger

from eval.eval_by_category import run_evaluation
from quantization.layers import LinearQuantHub
from quantization.llama_seq import llama_sequential
from utils.load_model import BaseModel


def main(config):
    basemodel = BaseModel(config)
    tokenizer = basemodel.build_tokenizer()
    model = basemodel.build_model()
    new_model = None
    if config.get("quant", False):
        new_model = basemodel.replace_module(model, exclude_layers=config.quant.skip_layers, include_layers=['.*'])
        # calibrate = get_calibrate_loader(tokenizer=tokenizer, calibrate_config=config.quant.data)
        calibrate = torch.load("calibrate.pt")

        new_model = llama_sequential(model=new_model, calibrate_data=calibrate, **config.quant)
        new_model = new_model.to("cuda")
        logger.info(f'model: {model}')
        logger.info(f'tokenizer: {tokenizer}')

    if config.get("save", False) and config.get("quant", False):
        model = basemodel.replace_module(new_model, module_type=LinearQuantHub, new_module_type="", display=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)

    if config.get("eval", False):
        eval_config = config.eval
        model = model.to(eval_config.get('device', "cpu"))
        run_evaluation(model, tokenizer, **eval_config)



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


    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')

    save_fake_path = os.path.join(config.save, 'output_models')
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
