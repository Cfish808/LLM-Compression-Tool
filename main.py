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

    blockwise_opt = ALGO_REGISTRY[modality_config.method](
        model,
        modality_config,
        input=None,
        padding_mask=None,
        config=config,
    )

    quantizaion = blockwise_opt.quantize()


