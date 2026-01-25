# run_eval_by_category.py
import logging
from typing import Dict, List, Optional, Iterable

import lm_eval
import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from loguru import logger
from sympy.strategies.core import switch
from tqdm import tqdm

from my_datasets import  get_calibrate_loader

# === 任务分类 ===
TASK_CATEGORIES: Dict[str, List[str]] = {
    "Language_Modeling": [
        "wikitext2",
        "c4",
        "ptb"
    ],
    "General_Knowledge": [
        "nq_open",
        "triviaqa",
        "mmlu",
    ],
    "Commonsense_Reasoning": [
        "commonsense_qa",
        "piqa",
        "social_iqa",
        "hellaswag",
        "winogrande",
        "arc_challenge",
        "arc_easy",
    ],
    "Complex_Reasoning": [
        "bigbench_strategyqa",
        "openbookqa",
        "mathqa",
        "gsm8k",
        "longbench_hotpotqa",
    ],
    "Safety_Truthfulness": [
        "truthfulqa",
        "crows_pairs_english",
    ],
    "Instruction_Following": [
        "ifeval",
    ],
}


def _unique_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out



def _wrap_hflm(model, tokenizer=None, device="cuda", **hflm_kwargs):
    """
        将已有 HF 模型对象封装为 lm_eval 的 HFLM。
        - model: transformers.PreTrainedModel
        - tokenizer: transformers.PreTrainedTokenizer (建议提供)
        其余可选参数：device, dtype, max_length 等（按需传入）
        """
    if tokenizer is not None:
        return HFLM(pretrained=model, tokenizer=tokenizer, device=device, **hflm_kwargs)
    return HFLM(pretrained=model, device=device, **hflm_kwargs)


@torch.no_grad()
def compute_ppl(model, tokenizer, loader,**kwargs):
    total_loss = 0
    total_count = 0
    for batch in tqdm(loader):
        batch = batch.clone()
        if batch.shape[1] <= 1: continue
        input_ids = batch.to(model.device)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        if tokenizer.pad_token_id is not None:

            count = input_ids.ne(tokenizer.pad_token_id).ne(-100).sum().item()
        else:
            count = input_ids.ne(-100).sum().item()
        total_loss += loss.item() * count
        total_count += count

    return np.exp(total_loss / total_count)

#
# def eval_wiki2_ppl(model, tokenizer, nsamples='all', seqlen=2048, split='test',**kwargs):
#     logging.info("Evaluating Perplexity (PPL) on the wikitext2")
#     dataloader = get_wikitext2(tokenizer, nsamples=nsamples, seqlen=seqlen, split=split)
#     ppl = compute_ppl(model, tokenizer, dataloader)
#     logging.info(f'wikitext2 PPL {ppl}')
#     return ppl
#
#
# def eval_ptb_ppl(model, tokenizer, nsamples='all', seqlen=2048, split='test',**kwargs):
#     logging.info("Evaluating Perplexity (PPL) on the ptb")
#     dataloader = get_ptb(tokenizer, nsamples=nsamples, seqlen=seqlen, split=split)
#     ppl = compute_ppl(model, tokenizer, dataloader)
#     logging.info(f'ptb PPL {ppl}')
#     return ppl
#
#
# def eval_c4_ppl(model, tokenizer, nsamples='all', seqlen=2048, split='validation',**kwargs):
#     logging.info("Evaluating Perplexity (PPL) on the c4")
#     dataloader = get_c4(tokenizer, nsamples=nsamples, seqlen=seqlen, split=split)
#     ppl = compute_ppl(model, tokenizer, dataloader)
#     logging.info(f'c4 PPL {ppl}')
#     return ppl

def eval_ppl(model, tokenizer,data_name,**kwargs):
    logging.info(f"Evaluating Perplexity (PPL) on the {data_name}")
    dataloader = get_calibrate_loader(tokenizer,data_name,**kwargs)
    ppl = compute_ppl(model, tokenizer, dataloader)
    logging.info(f'{data_name} PPL {ppl}')
    return ppl

def run_evaluation(
        model,
        tokenizer=None,
        device="cuda",
        task: str = None,
        datasets: List[str] = None,
        **kwargs
) -> Dict:

    if task == "ppl":
        for name in datasets:
            eval_ppl(model, tokenizer,name, **kwargs)

    elif task == "acc":
        results = evaluate_model(model=model, tokenizer=tokenizer, task_list=datasets, device=device,**kwargs)
        total_acc = 0
        for task in datasets:
            logger.info(results['results'][task])

# def run_evaluation(
#         model,
#         tokenizer=None,
#         tasks: List[str] = None,
#         datasets: List[str] = None,
#         **kwargs
# ) -> Dict:
#     """
#     通用评测入口：
#     - 通过 `category` 选择一类任务，或直接传 `tasks` 覆盖。
#     - model/tokenizer 为已加载的 HF 对象（无需路径）。
#     """
#     # 1) 解析任务列表
#     test_task_name = []
#     task_list = None
#     if tasks:
#         for task in tasks:
#             if task == "ppl":
#                 if "c4" in datasets:
#                     test_task_name.append("c4")
#                     eval_c4_ppl(model, tokenizer, **kwargs)
#                 if "wikitext2" in datasets:
#                     test_task_name.append("wikitext2")
#                     eval_wiki2_ppl(model, tokenizer, **kwargs)
#
#                 if "ptb" in datasets:
#                     test_task_name.append("ptb")
#                     eval_ptb_ppl(model, tokenizer, **kwargs)
#                 continue
#             tmp_task = []
#             for dataname in TASK_CATEGORIES[task]:
#                 if dataname in datasets:
#                     tmp_task.append(dataname)
#                     test_task_name.append(dataname)
#             if not tmp_task:
#                 tmp_task = TASK_CATEGORIES[task]
#                 test_task_name.extend(tmp_task)
#             task_list = _unique_keep_order(tmp_task)
#             results = evaluate_model(model=model, tokenizer=tokenizer, task_list=task_list, **kwargs)
#             total_acc = 0
#             logger.info(f"当前测试类别:{task}")
#             try:
#                 for task in task_list:
#                     logger.info(results['results'][task])
#                     total_acc += results['results'][task]['acc,none']
#                 logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
#             except:
#                 pass
#
#     elif not tasks:
#         all_task_list = []
#         tmp_task = []
#         for key, val in TASK_CATEGORIES.items():
#             all_task_list.extend(val)
#         for dataname in datasets:
#             if dataname in all_task_list:
#                 tmp_task.append(dataname)
#                 test_task_name.append(dataname)
#         task_list = _unique_keep_order(tmp_task)
#         total_acc = 0
#         results = evaluate_model(model=model, tokenizer=tokenizer, task_list=task_list, **kwargs)
#         try:
#             for task in task_list:
#                 total_acc += results['results'][task]['acc,none']
#             logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
#         except:
#             pass
#
#     if not test_task_name:
#         raise ValueError("Empty task list to evaluate.")
#     not_in_datasets = [x for x in datasets if x not in test_task_name]
#     if not_in_datasets:
#         logger.info(f"以下任务存不在任务列表中无法完成测试：{not_in_datasets}")


def evaluate_model(
        model,
        tokenizer,
        task_list=None,
        num_fewshot=0,
        batch_size=1,
        device="cuda",
        **kwargs
):
    """
    通用的 lm_eval 评测封装函数

    Args:
        model: HuggingFace 模型对象 (PreTrainedModel)
        tokenizer: HuggingFace tokenizer
        task_list: list[str]，任务列表
        num_fewshot: few-shot 样例数量
        batch_size: 批大小
        device: 设备 ("cuda:0")

    Returns:
        dict: lm_eval 评测结果
    """
    # 1) 处理 HFLM 参数
    task_manager = lm_eval.tasks.TaskManager()
    # 2) 封装成 HFLM
    lm = _wrap_hflm(model, tokenizer, batch_size=batch_size, device=device,**kwargs)

    # 3) 调用 lm_eval 进行评测
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    return results
