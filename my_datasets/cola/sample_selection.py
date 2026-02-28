# cola/sample_selection.py
"""
Sample selection module for the COLA framework (Stage 3).
Selects samples to maximize representativeness and diversity in activation space.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger("COLA")

class ActivationHook:
    """Hook for extracting activations from model layers. 支持 Llama/Qwen 等 tuple 输出"""
    
    def __init__(self, module):
        self.activations = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Store the output activations — 智能处理 tuple / tensor"""
        if isinstance(output, tuple):
            # Llama / Mistral / Qwen 等 decoder layer 返回 tuple，第一个元素就是 hidden_states
            act = output[0]
            logger.debug(f"Hook 捕获到 tuple 输出，自动取第 0 个元素 (shape: {act.shape if hasattr(act, 'shape') else 'N/A'})")
        else:
            act = output
        
        self.activations = act.detach() if hasattr(act, 'detach') else act
    
    def remove(self):
        self.hook.remove()

def extract_activations(
    model,
    inputs,
    layers=None,
    batch_size=4,
    device="cuda"
):
    """
    增强版：自动检测 + 强力 fallback（兼容 Llama/Qwen/Gemma/Mistral/Phi 等几乎所有模型）
    【已修改：全程 GPU，不再搬到 CPU（彻底解决 Half 相关错误）】
    """
    model = model.to(device)
    model.eval()

    # ==================== 强力自动检测层 ====================
    transformer_layers = []
    
    # 优先路径1：最常见的 model.model.layers 或 model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        transformer_layers = [(f"layer_{i}", layer) for i, layer in enumerate(model.model.layers)]
    elif hasattr(model, "layers"):
        transformer_layers = [(f"layer_{i}", layer) for i, layer in enumerate(model.layers)]
    elif hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
        transformer_layers = [(f"layer_{i}", layer) for i, layer in enumerate(model.model.transformer.h)]
    
    # 优先路径2：如果上面没找到，用 named_modules 兜底
    if not transformer_layers:
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ["layers", ".layer", "transformer.h", "decoder.layers"]):
                # 只取实际的 layer 模块（不是父模块）
                if hasattr(module, "forward") and len(list(module.children())) > 0:
                    transformer_layers.append((name, module))
    
    # ==================== 最终 fallback：取最后 16 层（几乎所有模型都有效） ====================
    if not transformer_layers:
        logger.warning("无法自动检测层结构 → 使用最后 16 层作为 fallback")
        # 遍历所有模块，找最后 16 个有 forward 的子模块
        all_modules = list(model.named_modules())
        for name, module in all_modules[-40:]:   # 从后往前多拿一些
            if hasattr(module, "forward") and "layer" in name.lower():
                transformer_layers.append((name, module))
            if len(transformer_layers) >= 16:
                break
    
    logger.info(f"成功检测到 {len(transformer_layers)} 个层用于提取激活")

    # ==================== 注册 hooks（只 hook 每个层的 output） ====================
    hooks = []
    for name, module in transformer_layers[-16:]:   # ← 保持你原来的 16 层
        # 优先 hook 能输出 hidden_states 的部分
        if hasattr(module, "output"):
            target = module.output
        elif hasattr(module, "feed_forward") or hasattr(module, "ffn"):
            target = module.feed_forward if hasattr(module, "feed_forward") else module.ffn
        else:
            target = module
        
        hook = ActivationHook(target)
        hooks.append((name, hook))

    # ==================== 实际前向传播（全程 GPU） ====================
    # 只把原始数据留在 CPU，batch 才搬运（避免一次性占满显存）
    input_ids_cpu = inputs["input_ids"]
    attention_mask_cpu = inputs["attention_mask"]
    num_samples = input_ids_cpu.shape[0]
    all_activations = {}

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_ids = input_ids_cpu[i:i+batch_size].to(device)
            batch_mask = attention_mask_cpu[i:i+batch_size].to(device)
            
            _ = model(input_ids=batch_ids, attention_mask=batch_mask)
            
            for name, hook in hooks:
                if hook.activations is not None:
                    # 🔥 关键修改：全程 GPU，不再 .cpu()
                    mean_act = hook.activations.mean(dim=1)
                    if name not in all_activations:
                        all_activations[name] = []
                    all_activations[name].append(mean_act)
                    hook.activations = None   # 立即释放，防止多层同时占用显存
            
            torch.cuda.empty_cache()   # 每 batch 清缓存，防碎片

    # 合并（仍在 GPU）
    for name in all_activations:
        all_activations[name] = torch.cat(all_activations[name], dim=0)

    # 清理 hooks
    for _, hook in hooks:
        hook.remove()

    logger.info(f"成功提取激活（全程 GPU），形状: {[v.shape for v in all_activations.values()]}")
    return all_activations


def aggregate_layer_activations(layer_activations):
    """
    增强版：空列表保护 + 自动 fallback
    """
    if not layer_activations or len(layer_activations) == 0:
        logger.error("没有提取到任何层激活！请检查模型是否加载正确")
        raise ValueError("No activations extracted. Please check your model architecture.")
    
    all_layers = []
    for name, activations in layer_activations.items():
        # 归一化
        norm_activations = activations / (activations.norm(dim=1, keepdim=True) + 1e-8)
        all_layers.append(norm_activations)
    
    aggregated = torch.cat(all_layers, dim=1)
    logger.info(f"成功聚合 {len(all_layers)} 层激活，形状: {aggregated.shape}")
    return aggregated


def random_projection(activation_vectors, reduced_dim=64):
    """
    随机投影 - GPU 专用版（全程 GPU，不再有 CPU Half 问题）
    """
    if not isinstance(activation_vectors, torch.Tensor):
        activation_vectors = torch.tensor(activation_vectors, device="cuda")
    
    device = activation_vectors.device
    dtype = activation_vectors.dtype
    original_dim = activation_vectors.shape[1]
    
    # 用 Python 原生计算 scale（最稳）
    scale = reduced_dim ** -0.5
    scale_tensor = torch.tensor(scale, device=device, dtype=dtype)
    
    random_matrix = torch.randn(
        original_dim, 
        reduced_dim, 
        device=device, 
        dtype=dtype
    ) * scale_tensor
    
    projected_vectors = torch.matmul(activation_vectors, random_matrix)
    
    logger.info(f"随机投影完成（GPU）：{original_dim} → {reduced_dim} 维，dtype={dtype}")
    return projected_vectors


def cluster_samples(activation_vectors, n_clusters=128, random_state=42):
    """
    Cluster samples based on their activation vectors.
    
    Args:
        activation_vectors: Tensor of activation vectors
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        
    Returns:
        KMeans object with cluster assignments
    """
    # Convert to numpy for sklearn（只在这里转 CPU）
    if isinstance(activation_vectors, torch.Tensor):
        vectors_np = activation_vectors.cpu().numpy()
    else:
        vectors_np = activation_vectors
    
    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        n_init=10
    )
    kmeans.fit(vectors_np)
    
    return kmeans


def select_representative_samples(samples, activation_vectors, kmeans):
    """
    Select representative samples from each cluster.
    
    Args:
        samples: List of processed samples
        activation_vectors: Tensor of activation vectors
        kmeans: KMeans object with cluster assignments
        
    Returns:
        List of selected representative samples
    """
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    
    # Convert to numpy for distance calculations
    if isinstance(activation_vectors, torch.Tensor):
        vectors_np = activation_vectors.cpu().numpy()
    else:
        vectors_np = activation_vectors
    
    # Find the closest sample to each cluster center
    selected_indices = []
    
    for cluster_idx in range(len(cluster_centers)):
        # Get indices of samples in this cluster
        cluster_sample_indices = np.where(cluster_labels == cluster_idx)[0]
        
        if len(cluster_sample_indices) == 0:
            continue
        
        # Get activation vectors for samples in this cluster
        cluster_vectors = vectors_np[cluster_sample_indices]
        
        # Calculate distances to cluster center
        center = cluster_centers[cluster_idx]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        
        # Find the closest sample
        closest_idx_in_cluster = np.argmin(distances)
        closest_idx_overall = cluster_sample_indices[closest_idx_in_cluster]
        
        selected_indices.append(closest_idx_overall)
    
    # Get the selected samples
    selected_samples = [samples[i] for i in selected_indices]
    
    return selected_samples


def select_samples(
    processed_samples: List[Dict],
    model,
    tokenizer,
    device: str = "cuda",
    num_clusters: int = 128,
    reduced_dim: int = 64,
    activation_layers: Optional[List[int]] = None,
    batch_size: int = 4,
    random_state: int = 42
) -> List[Dict]:
    """
    Select diverse and representative samples based on their activation patterns.
    
    Args:
        processed_samples: List of processed samples from Stage 2
        model: The LLM model
        tokenizer: The tokenizer
        device: Device to run the model on
        num_clusters: Number of clusters (determines final sample count)
        reduced_dim: Dimension after random projection
        activation_layers: List of layer indices to use (None for all)
        batch_size: Batch size for activation extraction
        random_state: Random seed for reproducibility
        
    Returns:
        List of selected samples
    """
    logger.info("Starting sample selection based on activation patterns")
    logger.info(f"Using {num_clusters} clusters and {reduced_dim} dimensions after projection")
    
    # Prepare inputs for all samples
    sample_texts = [sample["text"] for sample in processed_samples]
    
    # Batch tokenization
    logger.info("Tokenizing samples...")
    inputs = tokenizer(
        sample_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=2048  # Adjust based on your model's context size
    )
    
    # Extract activations from model（全程 GPU）
    logger.info("Extracting activations from model...")
    layer_activations = extract_activations(
        model=model,
        inputs=inputs,
        layers=activation_layers,
        batch_size=batch_size,
        device=device
    )
    
    # Aggregate activations from all layers
    logger.info("Aggregating activations from all layers...")
    aggregated_activations = aggregate_layer_activations(layer_activations)
    
    # Random projection to reduce dimensionality（全程 GPU）
    logger.info(f"Applying random projection to reduce dimension to {reduced_dim}...")
    projected_activations = random_projection(aggregated_activations, reduced_dim)
    projected_activations = projected_activations.cpu()
    
    # Cluster samples
    logger.info(f"Clustering samples into {num_clusters} clusters...")
    kmeans = cluster_samples(projected_activations, n_clusters=num_clusters, random_state=random_state)
    
    # Select representative samples from each cluster
    logger.info("Selecting representative samples from each cluster...")
    selected_samples = select_representative_samples(processed_samples, projected_activations, kmeans)
    
    logger.info(f"Selected {len(selected_samples)} samples for final calibration dataset")
    
    # Add selection metadata to each sample
    for i, sample in enumerate(selected_samples):
        sample["selection_index"] = i
        sample["selection_method"] = "activation_clustering"
    
    return selected_samples