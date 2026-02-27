# cola/dataset_processing.py
"""
Dataset processing module for the COLA framework (Stage 2).
Optimizes the compositional properties of the selected datasets.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any
from transformers import PreTrainedTokenizer
import datasets

logger = logging.getLogger("COLA")

def tokenize_text(text: str, tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, Any]:
    """
    Tokenize text and prepare for model input.
    
    Args:
        text: Input text to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with tokenized inputs
    """
    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    return {
        "text": text,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

def optimize_sequence_length(
    samples: List[Dict],
    tokenizer: PreTrainedTokenizer,
    target_length: int = 2048
) -> List[Dict]:
    """
    Optimize sequence length of samples.
    
    Args:
        samples: List of text samples
        tokenizer: Tokenizer to use
        target_length: Target sequence length
        
    Returns:
        List of processed samples with optimized sequence length
    """
    processed_samples = []
    
    for sample in samples:
        text = sample["text"]
        
        # Check current length
        tokens = tokenizer.encode(text)
        current_length = len(tokens)
        
        if current_length <= target_length:
            # If shorter than target, keep as is
            processed_sample = tokenize_text(text, tokenizer, target_length)
            processed_samples.append(processed_sample)
        else:
            # If longer than target, truncate with some context awareness
            # Try to break at paragraph or sentence boundaries
            paragraphs = text.split("\n\n")
            
            current_text = ""
            current_tokens = 0
            
            for paragraph in paragraphs:
                paragraph_tokens = tokenizer.encode(paragraph)
                paragraph_length = len(paragraph_tokens)
                
                # If adding this paragraph would exceed target length, break
                if current_tokens + paragraph_length > target_length - 2:  # -2 for special tokens
                    break
                
                current_text += paragraph + "\n\n"
                current_tokens += paragraph_length + 1  # +1 for newline
            
            # If we didn't get enough text, take the first target_length tokens
            if current_tokens < target_length // 2:
                current_text = tokenizer.decode(tokens[:target_length - 2])
            
            processed_sample = tokenize_text(current_text, tokenizer, target_length)
            processed_samples.append(processed_sample)
    
    return processed_samples

def enhance_format_with_reasoning(sample: Dict) -> Dict:
    """
    Enhance sample format with explicit reasoning chains if possible.
    
    Args:
        sample: Input sample dictionary
        
    Returns:
        Sample with enhanced format if possible
    """
    text = sample["text"]
    
    # Check if it already has explicit reasoning format
    if "Reasoning:" in text or "Step 1:" in text or "Chain of Thought:" in text:
        return sample
    
    # Patterns to detect implicit reasoning
    qa_pattern = re.search(r"(Question|Q):?([^\n]+).*?(Answer|A):?([^\n]+)", text, re.DOTALL)
    math_pattern = re.search(r"(Problem|Calculate|Solve):?([^\n]+).*?(Solution|Result|Answer):?([^\n]+)", text, re.DOTALL)
    
    if qa_pattern:
        # This looks like a QA pair, try to insert reasoning
        question = qa_pattern.group(2).strip()
        answer = qa_pattern.group(4).strip()
        
        # Look for explanatory text between question and answer
        explanation_match = re.search(f"{re.escape(question)}(.*?){re.escape(answer)}", text, re.DOTALL)
        
        if explanation_match and len(explanation_match.group(1).strip()) > 20:
            # There seems to be an explanation, format it as reasoning
            explanation = explanation_match.group(1).strip()
            
            # Format with explicit reasoning chain
            formatted_text = f"Question: {question}\n\nReasoning:\n{explanation}\n\nAnswer: {answer}"
            
            # Update the sample
            updated_sample = sample.copy()
            updated_sample["text"] = formatted_text
            updated_sample["format_enhanced"] = True
            return updated_sample
    
    elif math_pattern:
        # This looks like a math problem, try to insert reasoning
        problem = math_pattern.group(2).strip()
        solution = math_pattern.group(4).strip()
        
        # Look for steps between problem and solution
        steps_match = re.search(f"{re.escape(problem)}(.*?){re.escape(solution)}", text, re.DOTALL)
        
        if steps_match and len(steps_match.group(1).strip()) > 20:
            # There seems to be solution steps, format it as reasoning
            steps = steps_match.group(1).strip()
            
            # Format with explicit reasoning steps
            formatted_text = f"Problem: {problem}\n\nSolution Steps:\n{steps}\n\nAnswer: {solution}"
            
            # Update the sample
            updated_sample = sample.copy()
            updated_sample["text"] = formatted_text
            updated_sample["format_enhanced"] = True
            return updated_sample
    
    # If we couldn't enhance the format, return the original sample
    return sample

def process_datasets(
    selected_datasets: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    add_reasoning_chains: bool = True,
    min_length: int = 256,
    filter_low_quality: bool = True,
    max_samples_per_dataset: int = 5000   # 测试时用小数字，跑通后改大
) -> List[Dict]:
    """
    Process the selected datasets by optimizing their compositional properties.
    """
    all_processed_samples = []
    
    # === 全局修复：pad_token ===
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"已自动设置 pad_token = eos_token")

    for dataset_info in selected_datasets:
        dataset_name = dataset_info["name"]
        logger.info(f"Processing dataset: {dataset_name} (max {max_samples_per_dataset} samples)")

        try:
            # ==================== 加载数据集（streaming 加速） ====================
            if dataset_name == "c4":
                ds = datasets.load_dataset("allenai/c4", data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                                split='validation')
            else:
                full_ds = datasets.load_dataset(dataset_name)
                split_name = list(full_ds.keys())[0]
                ds = datasets.load_dataset(dataset_name, split=split_name, streaming=True)

            # ==================== 一次检测 text_field（不再在循环里 item.items()） ====================
            text_field = "text"
            if hasattr(ds, 'features') and ds.features is not None:
                if text_field not in ds.features:
                    for field in ds.features:
                        # 兼容 string 类型字段
                        if "string" in str(ds.features[field]):
                            text_field = field
                            break

            logger.info(f"使用字段: {text_field}  (从 features 自动检测)")

            # ==================== 读取样本（安全版） ====================
            raw_samples = []
            for i, item in enumerate(ds):
                if i >= max_samples_per_dataset:
                    break
                if not isinstance(item, dict):
                    continue  # 跳过坏数据

                # 获取文本（超级安全）
                if text_field in item and isinstance(item[text_field], str):
                    text = item[text_field]
                else:
                    # 兜底：找第一个长字符串
                    text = None
                    for v in item.values():
                        if isinstance(v, str) and len(v) > 50:
                            text = v
                            break
                    if text is None:
                        continue  # 跳过这条坏样本

                raw_samples.append({"text": text, "source": dataset_name})

            logger.info(f"成功加载 {len(raw_samples)} 条有效样本 from {dataset_name}")

            # ==================== 长度过滤 ====================
            if filter_low_quality:
                filtered_samples = []
                for sample in raw_samples:
                    tokens = tokenizer.encode(sample["text"])
                    if len(tokens) >= min_length:
                        filtered_samples.append(sample)
                logger.info(f"Filtered {len(raw_samples) - len(filtered_samples)} short samples")
                raw_samples = filtered_samples

            # ==================== 后续处理（长度优化 + 推理链） ====================
            length_optimized_samples = optimize_sequence_length(
                raw_samples, tokenizer, max_length
            )

            if add_reasoning_chains:
                enhanced_samples = [enhance_format_with_reasoning(s) for s in length_optimized_samples]
                enhanced_count = sum(1 for s in enhanced_samples if s.get("format_enhanced", False))
                logger.info(f"Enhanced {enhanced_count} samples with reasoning chains")
                final_samples = enhanced_samples
            else:
                final_samples = length_optimized_samples

            # 添加元数据
            for sample in final_samples:
                sample["dataset_name"] = dataset_name
                sample["dataset_score"] = dataset_info["total_score"]
                sample["capability_scores"] = dataset_info["capability_scores"]

            all_processed_samples.extend(final_samples)

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    logger.info(f"Total processed samples: {len(all_processed_samples)}")
    return all_processed_samples