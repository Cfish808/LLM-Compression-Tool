import random
import logging

import numpy as np
import torch

import copy
from .load_ceval import get_calibrate_ceval
from .load_cmmlu import get_calibrate_cmmlu
from .load_boss import get_calibrate_boss

import os

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from torch.nn.utils.rnn import pad_sequence


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
        valdata = load_dataset("./c4_local",
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
        traindata = load_dataset("mi_optimize/my_datasets/ptb_text_only", 'penn_treebank', split='train')
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
        testdata = load_dataset("mi_optimize/my_datasets/ptb_text_only", 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        testloader = []
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        for i in range(nsamples):
            testloader.append(testenc[:, (i * seqlen):((i + 1) * seqlen)])
        return testloader

    raise ValueError(f'not support ptb {split} split')


def get_test_loader(dataset_name, tokenizer, seqlen=2048, nsamples=128, seed=42, split='test'):
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Sequence length: {seqlen}")
    logging.info(f"Number of samples: {nsamples}")

    if dataset_name == 'wikitext2':
        return get_wikitext2(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed, split=split)
    if dataset_name == 'c4':
        return get_c4(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen)
    if dataset_name == 'ptb':
        return get_ptb(tokenizer, nsamples=nsamples, seed=seed, seqlen=seqlen)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_calibrate_loader(tokenizer, calibrate_config: dict = {}):
    calibrate_name = calibrate_config['name']

    if calibrate_name == 'wikitext2':
        return get_wikitext2(tokenizer, **calibrate_config)

    if calibrate_name == 'c4':
        return get_c4(tokenizer, **calibrate_config)

    if calibrate_name == 'ptb':
        return get_ptb(tokenizer, **calibrate_config)

    if calibrate_name == 'ceval':
        return get_calibrate_ceval(tokenizer, **calibrate_config)

    if calibrate_name == 'cmmlu':
        return get_calibrate_cmmlu(tokenizer, **calibrate_config)

    if calibrate_name == 'boss':
        return get_calibrate_boss(tokenizer, **calibrate_config)

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
