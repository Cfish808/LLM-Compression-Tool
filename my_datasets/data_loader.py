import json
import pdb
import random
import logging
from collections import defaultdict

import nltk
import numpy as np
import torch

import copy
from .load_ceval import get_calibrate_ceval
from .load_cmmlu import get_calibrate_cmmlu
from .load_boss import get_calibrate_boss

import os

from datasets import load_dataset, concatenate_datasets, interleave_datasets
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union, List, Any
from torch.nn.utils.rnn import pad_sequence

def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    try:
        loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
        traindata = load_dataset(loacal_dataset, split='train')
    except:
        traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train')

    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)
    trainloader = []

    val_sample_ratio = 0.9
    print("**********start the trainloader.************")

    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    print("**********finish the trainloader.************")

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader

def get_wikitext2(tokenizer, split='test', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split == 'train':
        logging.info("get_wikitext2_train")
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    if split == 'test':
        logging.info("get_wikitext2_test")
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        if nsamples == 'all':
            nsamples = len(testenc['input_ids'][0]) // seqlen + 1
        testloader = []
        for i in range(nsamples):
            testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
        return testloader

    raise ValueError(f'not support wikitext2 {split} split')


def get_c4(tokenizer, split='validation', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split == 'train':
        logging.info("get_c4_train")
        traindata = load_dataset("allenai/c4", name='default',
                                 data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader


    if split == 'validation':
        logging.info("get_c4_validation")
        valdata = load_dataset("allenai/c4",
                            # "allenai/c4",
                               data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                               split='validation')

        testenc = tokenizer(" ".join(valdata[:1100]['text']), return_tensors='pt')
        if nsamples == 'all':
            nsamples = len(testenc['input_ids'][0]) // seqlen + 1
        testloader = []
        for i in range(nsamples):
            testloader.append(testenc['input_ids'][:, i * seqlen: (i + 1) * seqlen])
        return testloader

    raise ValueError(f'not support c4 {split} split')


def get_ptb(tokenizer, split='test', nsamples=128, seqlen=2048, seed=42, **kwargs):
    if split == 'train':
        logging.info("get_ptb_train")
        traindata = load_dataset("./my_datasets/ptb_text_only", 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append(inp)
        return trainloader

    if split == 'test':
        logging.info("get_ptb_test")
        testdata = load_dataset("./my_datasets/ptb_text_only", 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        testloader = []
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        for i in range(nsamples):
            testloader.append(testenc[:, (i * seqlen):((i + 1) * seqlen)])
        return testloader

    raise ValueError(f'not support ptb {split} split')


def get_download(tokenizer, split='train', nsamples=128, seqlen=2048, seed=42, **kwargs):
    path = kwargs.get("path")
    name = kwargs.get("name", None)
    datakey = kwargs.get("datakey", "text")
    # datakey:
    # Specifies the field name in each dataset sample that contains the raw text.
    # For example, in JSON datasets this is typically "text", but it may vary
    # depending on the dataset structure (e.g., "content", "sentence", "document").
    # This key is used to extract textual data before tokenization.
    if not kwargs.get("download",False):
        # 本地 json / json.gz 加载
        data_sets = load_dataset(
            "json",
            data_files=path,
            split="train"
        )
    else:
        data_sets = load_dataset(path, name, split=split)

    if split == 'train':
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(data_sets) - 1)
                trainenc = tokenizer(data_sets[i][datakey], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        # return trainloader
        return up_batch_size(trainloader,kwargs.get("batch_size"))

    else:
        logging.info("get_test")
        testenc = tokenizer(" ".join(data_sets[:1100][datakey]), return_tensors="pt")
        testloader = []
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        for i in range(nsamples):
            testloader.append(testenc[:, (i * seqlen):((i + 1) * seqlen)])
        return testloader

    raise ValueError(f'not support ptb {split} split')

def up_batch_size(samples,calib_bs=1):
    calib_model_inputs = []
    if calib_bs > 1:
        for i in range(0, len(samples), calib_bs):
            start = i
            end = min(i + calib_bs, len(samples))
            batch = samples[start:end]
            batch = torch.cat(batch, dim=0)
            calib_model_inputs.append(batch)
    else:
        calib_model_inputs = samples
    return calib_model_inputs



def get_calibrate_loader(tokenizer, calibrate_config: dict = {}):
    calibrate_name = calibrate_config.get("name",None)
    calibrate_down = calibrate_config.get("download",False)
    calibrate_path = calibrate_config.get("path",None)

    if calibrate_name == 'wikitext2' and calibrate_down:
        return get_wikitext2(tokenizer, **calibrate_config)

    if calibrate_name == 'c4' and calibrate_down:
        return get_c4(tokenizer, **calibrate_config)

    if calibrate_name == 'ptb' and calibrate_down:
        return get_ptb(tokenizer, **calibrate_config)

    if calibrate_name == 'ceval' and calibrate_down:
        return get_calibrate_ceval(tokenizer, **calibrate_config)

    if calibrate_name == 'cmmlu' and calibrate_down:
        return get_calibrate_cmmlu(tokenizer, **calibrate_config)

    if calibrate_name == 'boss' and calibrate_down:
        return get_calibrate_boss(tokenizer, **calibrate_config)

    if calibrate_down == True or calibrate_path != None:
        return get_download(tokenizer, **calibrate_config)

    raise ValueError(f'not support calibrate name:{calibrate_name}')


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: object
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        IGNORE_INDEX = -100
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) 
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def make_data_module(tokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'wikitext2':
            return load_dataset("wikitext", "wikitext-2-raw-v1")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset):
        if  args.dataset in ['alpaca', 'alpaca-clean']:
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif args.dataset == 'chip2':
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif args.dataset == 'self-instruct':
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif args.dataset == 'hh-rlhf':
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif args.dataset == 'oasst1':
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif args.dataset == 'wikitext2':
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
            dataset = dataset.filter(lambda x: len(x['output'].strip()) > 0)
        else:
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset)

    train_dataset = dataset['train']
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.group_by_length:
        train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len
    )
    return dict(
        train_dataset=train_dataset,
        eval_dataset= None,
        predict_dataset=None,
        data_collator=data_collator
    )

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def get_dataset(data_args):
    EXT2TYPE = {"csv": "csv","json": "json","jsonl": "json","txt": "text"}
    max_samples = data_args.max_samples
    all_datasets: List[Union["Dataset", "IterableDataset"]] = [] # support multiple datasets
    for dataset_attr in data_args.dataset_list:
        print("Loading dataset {}...".format(dataset_attr))
        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # directory
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # single file
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
        else:
            raise NotImplementedError

        dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=data_args.split,
            streaming=data_args.streaming
        )

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        # TODO: adapt to the sharegpt format

        for column_name in ["prompt", "query", "response", "history"]: # align datasets
            if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.system_prompt: # add system prompt
            if data_args.streaming:
                dataset = dataset.map(lambda _: {"system": dataset_attr.system_prompt})
            else:
                dataset = dataset.add_column("system", [dataset_attr.system_prompt] * len(dataset))

        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            print("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            print("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=data_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        )
    else:
        raise ValueError("Unknown mixing strategy.")

def preprocess_dataset(dataset,tokenizer,data_args,training_args):
    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        import tiktoken
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["prompt"], **kwargs)
        from itertools import chain
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        block_size = data_args.cutoff_len
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of cutoff_len
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
        
    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    from quantization.onebit.extras import get_logger, IGNORE_INDEX, get_template_and_fix_tokenizer
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)
    preprocess_func = preprocess_pretrain_dataset
    print_function = print_unsupervised_dataset_example
    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset"
            )

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )

        if data_args.cache_path is not None and not os.path.exists(data_args.cache_path):
            if training_args.should_save:
                dataset.save_to_disk(data_args.cache_path)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
   

def get_dataset_loader(tokenizer, data_args, training_args):
    dataset = get_dataset(data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args)
    return dataset

