import argparse
import time
from easydict import EasyDict
from utils.regaster import  QUANTIZATION_REGISTRY
import logging
import yaml


if __name__ == "__main__":
    # 创建 logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 控制台输出 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # 添加 handler
    logger.addHandler(console_handler)
    llmc_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    blockwise_opt = QUANTIZATION_REGISTRY[config.model.type](
        config=config
    )

    blockwise_opt.build_model()


