# This file is modified from https://github.com/artidoro/qlora/blob/main/qlora.py
import json
import warnings
from dataclasses import field, dataclass
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
# from transformers import TrainingArguments
import inspect, argparse


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


@dataclass
class ModelArguments:
    path: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model by Block-AP."}
    )
    torch_dtype: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model by Block-AP."}
    )
    real_quant: bool = field(
        default=True, metadata={"help": "reload the quant model or not."}
    )
    mixed_precision: bool = field(
        default=False, metadata={"help": "reload the quant model or not."}
    )
    maskfile_dir: Optional[str] = field(
        default="./salient_columns.json",
        metadata={"help": "direction of mask file"}
    )
    type: Optional[str] = field(
        default="llama-2",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    name: str = field(
        default='redpajama',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    cache_path: str = field(
        default='./cache',
        metadata={"help": "direction of cached dataset, leading to faster debug"}
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."}
    )
    train_on_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the mask on the prompt or not."}
    )
    conv_temp: str = field(
        default='llama-2',
        metadata={"help": "Conversation template, only useful with deita datasets"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)



@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    should_save: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to save the preprocessed dataset."}
    )
    should_log: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to log the preprocessed dataset."}
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_ppl_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    pt_context_len: int = field(
        default=1024,
        metadata={"help": "language modeling length."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    wbits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    group_size: int = field(
        default=64,
        metadata={"help": "How many group size to use."}
    )
    max_memory_MB: int = field(
        default=81000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    method: str = field(
        default='efficientqat_e2e',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    resume_from_checkpoint: str = field(default=None, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=0, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.01, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-5, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=5, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

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

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
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

    # 1️⃣ 生成 HF training args
    hf_keys = inspect.signature(TrainingArguments.__init__).parameters
    training_args_dict = {k: v for k, v in vars(config.quant).items() if k in hf_keys}
    training_args = TrainingArguments(**training_args_dict)

    # 2️⃣ 生成 HF generation config
    training_args.generation_config = transformers.GenerationConfig(**vars(config.generation_args))

    # 3️⃣ 生成 model / data / generation dataclass
    model_args = ModelArguments(**vars(config.base_model))
    data_args = DataArguments(**vars(config.data))
    generation_args = GenerationArguments(**vars(config.generation_args))

    # 4️⃣ 最终 args
    tmp_args = flatten_dict(config)
    # 已有三类参数
    base_dict = {
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
    }
    extra_dict = {
        k: v
        for k, v in tmp_args.items()
        if k not in base_dict
    }

    args = argparse.Namespace(
        **base_dict,
        **extra_dict
    )

    data_args.seed = training_args.seed
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

    cache_dataloader = f'{data_args.cache_path}/e2e_dataloader_{data_args.name}_{model_args.type}_{args.pt_context_len}_{data_args.max_train_samples}.cache'
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

