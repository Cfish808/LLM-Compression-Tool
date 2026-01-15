# This file is modified from https://github.com/artidoro/qlora/blob/main/qlora.py
import json
import warnings
from os.path import exists, join, isdir
from typing import Optional, Dict
import numpy as np
import importlib
from packaging import version

import torch
import transformers
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    LlamaTokenizer, Trainer, TrainerCallback, DataCollatorForLanguageModeling
)

from quantization.efficientqat.datautils_e2e import make_data_module
from bitsandbytes.optim import AdamW
import os

from quantization.efficientqat.dsets import split_dataset, get_dataset, preprocess_dataset
from quantization.efficientqat.int_linear_real import trans2mixprecison_model, QuantLinear_fake
from utils.efficientqat_utils import create_logger
from quantization.efficientqat.int_linear_real import load_quantized_model, QuantLinear
from pathlib import Path

from utils.config_utils import flatten_dict, to_dotdict


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    model, tokenizer = load_quantized_model(args.path + "_mix_precision_init", args.wbits, args.group_size,
                                            real_quant=args.real_quant, mixed_precision=args.mixed_precision,
                                            mask_training_only=True, maskfile_dir=args.maskfile_dir)
    tokenizer.model_max_length = args.pt_context_len
    # import pdb;pdb.set_trace()

    compute_dtype = (torch.float16 if args.bf16  else torch.float32)
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float32 if args.bf16 else torch.bfloat16)
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.cuda()
    model.train()

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

        # TODO
    # if 'llama1' in args.model_name_or_path or 'llama2' in args.model_name_or_path or 'llama-1' in args.model_name_or_path or 'llama-2' in args.model_name_or_path:
    if isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    '''
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)
    '''
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        # if isinstance(module, QuantLinear):
        #     # transfer trainable step size into float32
        #     module.scales.data = module.scales.data.to(torch.float32)
        if 'norm' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                    # module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    # import pdb;pdb.set_trace()
    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*' * 80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print('*' * 80)
    if args.wbits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train(config):

    # model_args, data_args, training_args, generation_args, extra_args = \
    #     hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    # training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # args = argparse.Namespace(
    #     **vars(model_args), **vars(data_args), **vars(training_args)
    # )
    # args = to_dotdict(flatten_dict(config))

    # args = to_dotdict(flatten_dict(config))
    model_args = config.base_model
    data_args = config.data
    training_args = config.quant
    generation_args = config.generation_args

    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    # args = argparse.Namespace(
    #     **vars(model_args), **vars(data_args), **vars(training_args)
    # )
    args = flatten_dict(model_args)
    args.update(flatten_dict(data_args))
    args.update(flatten_dict(training_args))
    args = to_dotdict(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = create_logger(args.output_dir)
    logger.info(args)

    # checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    checkpoint_dir, completed_training = None, False
    if completed_training:
        print('Detected that training was already completed!')

    trans2mixprecison_model(args.path, args.wbits, args.group_size, real_quant=args.real_quant,
                            mixed_precision=args.mixed_precision, mask_training_only=True,
                            maskfile_dir=args.maskfile_dir)
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    cache_dataloader = f'{data_args.cache_path}/e2e_dataloader_{data_args.name}_{model_args.type}_{training_args.pt_context_len}_{data_args.max_train_samples}.cache'
    if data_args.cache_path is not None and os.path.exists(cache_dataloader):
        print("Loading dataset from disk will ignore other data arguments.")
        from datasets import load_from_disk
        dataset = load_from_disk(cache_dataloader)
    else:
        dataset = get_dataset(model_args, data_args)
        dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, model_args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    # data_module = make_data_module(tokenizer=tokenizer, args=args)

    if args.mixed_precision:
        import json
        salient_columns = json.loads(open(args.maskfile_dir).read())

    for name, module in model.model.named_modules():
        # if isinstance(module, LoraLayer):
        if args.mixed_precision and isinstance(module, QuantLinear_fake) and not 'head' in name:
            # module.weight.requires_grad = True
            # del module.weight_quantizer
            if module.bias is not None:
                module.bias.requires_grad = True
            module.mask_weight.requires_grad = True
            # module.weight_quantizer.scale.requires_grad = False
            module.use_weight_quant = True
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"可训练参数数量: {trainable_params}")
        elif isinstance(module, QuantLinear_fake) and not 'head' in name:
            module.scales.requires_grad = True
            module.use_weight_quant = True
    # if not args.mixed_precision: optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if 'scale' in n], 'weight_decay': 0.0, 'lr': args.learning_rate})
    # else:
    #     optimizer_grouped_parameters.append(
    #         {'params': [p for n, p in model.named_parameters() if 'scale' in n], 'weight_decay': 0.0,
    #          'lr': args.learning_rate})
    # optimizer = AdamW(optimizer_grouped_parameters)

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     optimizers=(optimizer, None),
    #     **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    # )
    optimizer_grouped_parameters = []
    # optimizer_grouped_parameters.append({'params': [p for n, p in model.named_parameters() if 'mask_weight' in n], 'weight_decay': 0.01, 'lr': args.learning_rate})
    # optimizer = AdamW(optimizer_grouped_parameters)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TrainerCallback()],
        # optimizers=(optimizer, None),
        **split_dataset(dataset, data_args, training_args)
    )

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    print(args.output_dir)
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

