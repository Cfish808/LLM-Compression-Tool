import os
import hashlib
import tiktoken
from itertools import chain
from typing import TYPE_CHECKING, List, Union, Optional, Dict, Literal, Generator, Any, Tuple
from dataclasses import dataclass
from datasets import concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from datasets import Dataset, IterableDataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import Seq2SeqTrainingArguments, TrainingArguments




EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}

@dataclass
class Template:

    prefix: List[Union[str, Dict[str, str]]]
    prompt: List[Union[str, Dict[str, str]]]
    system: str
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids = prompt_ids + query_ids + resp_ids
        prompt_ids, answer_ids = prompt_ids + encoded_pairs[-1][0], encoded_pairs[-1][1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        system, history = self._format(query, resp, history, system)
        encoded_pairs = self._encode(tokenizer, system, history)
        return encoded_pairs

    def _format(
        self,
        query: str,
        resp: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str]]]:
        r"""
        Aligns inputs to the standard format.
        """
        system = system or self.system # use system if provided
        history = history if (history and self.use_history) else []
        history = history + [(query, resp)]
        return system, history

    def _get_special_ids(
        self,
        tokenizer: "PreTrainedTokenizer"
    ) -> Tuple[List[int], List[int]]:
        if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", True):
            bos_ids = [tokenizer.bos_token_id]
        else: # baichuan, qwen and gpt2 models have no bos token
            bos_ids = []

        if tokenizer.eos_token_id is None:
            raise ValueError("EOS token is required.")

        if self.efficient_eos: # used in baichuan, qwen, chatglm, etc.
            eos_ids = []
        else:
            eos_ids = [tokenizer.eos_token_id]

        return bos_ids, eos_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        system: str,
        history: List[Tuple[str, str]]
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: bos + prefix + sep + query    resp + eos
        Turn t: sep + bos + query             resp + eos
        """
        bos_ids, eos_ids = self._get_special_ids(tokenizer)
        sep_ids = self._convert_inputs_to_ids(tokenizer, context=self.sep)
        encoded_pairs = []
        for turn_idx, (query, resp) in enumerate(history):
            if turn_idx == 0:
                prefix_ids = self._convert_inputs_to_ids(tokenizer, context=self.prefix, system=system)
                if len(prefix_ids) != 0: # has prefix
                    prefix_ids = bos_ids + prefix_ids + sep_ids
                else:
                    prefix_ids = bos_ids
            else:
                prefix_ids = sep_ids + bos_ids

            query_ids = self._convert_inputs_to_ids(tokenizer, context=self.prompt, query=query, idx=str(turn_idx))
            resp_ids = self._convert_inputs_to_ids(tokenizer, context=[resp])
            encoded_pairs.append((prefix_ids + query_ids, resp_ids + eos_ids))
        return encoded_pairs

    def _convert_inputs_to_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        context: List[Union[str, Dict[str, str]]],
        system: Optional[str] = None,
        query: Optional[str] = None,
        idx: Optional[str] = None
    ) -> List[int]:
        r"""
        Converts context to token ids.
        """
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=False)

        token_ids = []
        for elem in context:
            if isinstance(elem, str):
                elem = elem.replace("{{system}}", system, 1) if system is not None else elem
                elem = elem.replace("{{query}}", query, 1) if query is not None else elem
                elem = elem.replace("{{idx}}", idx, 1) if idx is not None else elem
                if len(elem) != 0:
                    token_ids = token_ids + tokenizer.encode(elem, **kwargs)
            elif isinstance(elem, dict):
                token_ids = token_ids + [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            else:
                raise ValueError("Input must be string or dict[str, str], got {}".format(type(elem)))

        return token_ids

templates: Dict[str, Template] = {}

def get_template_and_fix_tokenizer(
    name: str,
    tokenizer: "PreTrainedTokenizer"
) -> Template:
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        print("Add eos token: {}".format(tokenizer.eos_token))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Add pad token: {}".format(tokenizer.pad_token))

    if name is None:
        return None

    template = templates.get(name, None)
    assert template is not None, "Template {} does not exist.".format(name)
    tokenizer.add_special_tokens(
        dict(additional_special_tokens=template.stop_words),
        replace_additional_special_tokens=False
    )
    return template

def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        val_size = int(data_args.max_eval_samples) if data_args.max_eval_samples > 1 else data_args.max_eval_samples
        dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
        return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}

    else: # do_eval or do_predict
        return {"eval_dataset": dataset}


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> Union["Dataset", "IterableDataset"]:
    max_samples = data_args.max_train_samples
    name = data_args.name
    if name == "redpajama":
        try:
            loacal_dataset = "/mnt/usercache/zxy/EfficientQAT/EfficientQAT/data/RedPajama-Data-1T-Sample"
            traindata = load_dataset(loacal_dataset,split='train')
        except:
            traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",split='train')
    elif name == "c4":
        traindata = load_dataset("c4_local", data_files=["c4-train.00000-of-01024.json.gz", "c4-train.00001-of-01024.json.gz", "c4-train.00002-of-01024.json.gz", "c4-train.00003-of-01024.json.gz"], split='train')
        #traindata = load_dataset("C4", data_files={"train": ["c4-train.00000-of-01024.json.gz"], "validation": "c4-validation.00000-of-00008.json.gz"})

    '''
    if max_samples is not None:
        max_samples_temp = min(len(traindata), max_samples)
        traindata = traindata.select(range(max_samples_temp))
    '''
    return traindata

def preprocess_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    model_args: "ModelArguments",
) -> Union["Dataset", "IterableDataset"]:
    template = get_template_and_fix_tokenizer(data_args.template, tokenizer)

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")


    def preprocess_pretrain_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        # build grouped texts with format `X1 X2 X3 ...`
        if isinstance(getattr(tokenizer, "tokenizer", None), tiktoken.Encoding): # for tiktoken tokenizer (Qwen)
            kwargs = dict(allowed_special="all")
        else:
            kwargs = dict(add_special_tokens=True)

        if hasattr(tokenizer, "add_eos_token"): # for LLaMA tokenizer
            setattr(tokenizer, "add_eos_token", True)

        tokenized_examples = tokenizer(examples["text"], **kwargs)
        # concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        # total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # block_size = training_args.pt_context_len
        # # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        # total_length = (total_length // block_size) * block_size
        # # split by chunks of cutoff_len
        # result = {
        #     k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        #     for k, t in concatenated_examples.items()
        # }

        input_ids, attention_masks = [], []
        import random
        block_size = training_args.pt_context_len
        for index in range(len(examples["text"])):
            if len(tokenized_examples["input_ids"][index]) > block_size:
                start = random.randint(0, len(tokenized_examples["input_ids"][index]) - block_size - 1)
                end = start + block_size
                input_id, attention_mask = tokenized_examples["input_ids"][index][start:end], tokenized_examples["attention_mask"][index][start:end]
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
        result = {"input_ids": input_ids, "attention_mask": attention_masks}
        return result

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    preprocess_func = preprocess_pretrain_dataset
    print_function = print_unsupervised_dataset_example

    cache_dataloader = f'{data_args.cache_path}/e2e_dataloader_{data_args.name}_{model_args.model_family}_{training_args.pt_context_len}_{data_args.max_train_samples}.cache'
    if data_args.cache_path is not None and os.path.exists(cache_dataloader):
        print("Loading dataset from disk will ignore other data arguments.")
        return load_from_disk(cache_dataloader)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}

        dataset = dataset.map(
            preprocess_func,
            batched=True,            
            remove_columns=column_names,
            **kwargs
        )
        max_samples = data_args.max_train_samples
        if max_samples is not None:
            max_samples_temp = min(len(dataset["input_ids"]), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        if data_args.cache_path is not None and not os.path.exists(cache_dataloader):
            if training_args.should_save:
                dataset.save_to_disk(cache_dataloader)
            raise SystemExit("Dataset saved, rerun this script with the same `--cache_file`.")

        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Empty dataset!")

        return dataset
