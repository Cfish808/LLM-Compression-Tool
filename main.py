import argparse
import sys
import time
from venv import logger
from easydict import EasyDict
from utils.regaster import  QUANTIZATION_REGISTRY

import yaml
from quantization import *


if __name__ == "__main__":
    logger.add(sys.stdout, level='INFO')
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


