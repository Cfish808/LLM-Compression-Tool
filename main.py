import argparse
import json
import sys
import time
import yaml
from easydict import EasyDict
from loguru import logger
from eval.eval_by_category import run_evaluation
from my_datasets import get_calibrate_loader,make_data_module
from quantization.layers import LinearQuantHub
from quantization.llama_seq import llama_sequential, llama_omniquant
from utils.load_model import BaseModel, get_accelerate_model
from quantization.efficientqat.block_ap import block_ap, get_loaders


def get_model(config):
    basemodel = BaseModel(config)
    tokenizer = basemodel.build_tokenizer()
    model = basemodel.build_model()
    return model, tokenizer , basemodel


def main(config):
    new_model = None
    if "quant" in config and config.quant.method != "efficientqat_e2e":
        model, tokenizer, basemodel = get_model(config)
    else:
        model = None
        tokenizer = None
    if config.get("quant", False):
        calibrate = None
        if config.quant.method in ["qlora", "qalora","irlora"]:
            calibrate = make_data_module(tokenizer=tokenizer, args=config.quant.data)
        elif config.quant.method not in ["efficientqat_e2e","efficientqat_block"]:
            # pass
            calibrate = get_calibrate_loader(tokenizer=tokenizer, calibrate_config=config.quant.data)
        if config.quant.method == "omniquant":
            model = llama_omniquant(config.base_model.path, model, calibrate, config.quant, logger=logger)
            new_model = model
        elif config.quant.method == "quip_sharp":
            from quantization.llama_seq import llama_quipsharp
            model = llama_quipsharp(calibrate,config)
            new_model = model
        elif config.quant.method == "fbi_llm":
            from quantization.fbi_llm.fbi_train import train_fbi
            model = train_fbi(model, calibrate, config)
            new_model = model
        elif config.quant.method == "efficientqat_block":
            trainloader, valloader = get_loaders(
                config.quant.data.name,
                tokenizer,
                config.quant.data.train_size,
                config.quant.data.val_size,
                seed=config.quant.data.seed,
                seqlen=config.quant.data.training_seqlen,
                model_type=config.base_model.type,
            )
            block_ap(
                model,
                config,
                trainloader,
                valloader,
                logger,
            )
            new_model = model
        elif config.quant.method == "efficientqat_e2e":
            from quantization.efficientqat.e2e import train
            new_model,tokenizer = train(config)
        elif config.quant.method in ["qlora", "qalora","irlora"]:
            from quantization.qlora.qlora import train
            model,tokenizer = get_accelerate_model(config.base_model,config.quant.method)
            model = train(model=model,tokenizer=tokenizer,calibrate_data=calibrate, args=config.quant.args)
            new_model = model
        else:
            new_model = basemodel.replace_module(model, exclude_layers=config.quant.skip_layers, include_layers=['.*'])
            new_model = llama_sequential(model=new_model, calibrate_data=calibrate, **config.quant)
            logger.info(f'model: {model}')
            logger.info(f'tokenizer: {tokenizer}')

        if config.get("save", False):
            model = basemodel.replace_module(new_model, module_type=LinearQuantHub, new_module_type="", display=True)
            gen_config = model.generation_config
            gen_config.do_sample = True
            model.save_pretrained(config.get("save"))
            tokenizer.save_pretrained(config.get("save"))

    if config.get("eval", False):
        if new_model is None:
            config.base_model.path = config.save
            new_model, tokenizer, _ = get_model(config)
        eval_config = config.eval
        model = new_model.to(eval_config.get('device', "cpu"))
        run_evaluation(model, tokenizer, **eval_config)



def mkdirs(path):
    #
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    logger.add(sys.stdout, level='INFO')
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/efficientqat_block.yml", type=str)
    args = parser.parse_args()
    import os

    # 设置 HuggingFace 的缓存和镜像源
    os.environ['HF_HOME'] = './.huggingface'
    os.environ['HF_DATASETS_CACHE'] = './.huggingface'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')


    save_fake_path = os.path.join(config.save, 'output_models')
    mkdirs(save_fake_path)


    main(config)


    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f'llmc_duration_time: {llmc_duration_time} s')
    logger.info('--- llmc finished ---')


