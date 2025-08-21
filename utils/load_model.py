#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：model-quantification-tool 
@File    ：load_model.py
@Author  ：zxy
@Date    ：2025/8/20 15:12 
'''
import inspect
import logging
from typing import Dict, List, Optional, Union
from collections import defaultdict

import accelerate
import torch
import transformers
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logger  # Add required imports
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers


class BaseModel():
    def __init__(self, config,device_map=None,use_cache=False):
        self.vision_model = None
        self.model_config = None
        self.config = config
        self.use_cpu_to_save_cuda_mem_for_catcher = self.config.model.get('use_cpu_to_save_cuda_mem_for_catcher',
                                                                          False)  # noqa
        self.model_type = self.config.model.type
        self.model_path = self.config.model.path
        self.torch_dtype = self.config.model.torch_dtype
        self.tokenizer_mode = self.config.model.get('tokenizer_mode', 'fast')
        self.model = None  # 子类加载具体模型
        self.tokenizer = None  # 子类加载具体tokenizer
        self.use_cache = use_cache
        self.device_map = device_map

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.model_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
    def from_quantized(
            cls,
            model_name_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = None,
            low_cpu_mem_usage: bool = False,
            use_triton: bool = False,
            use_qigen: bool = False,
            use_marlin: bool = False,
            torch_dtype: Optional[torch.dtype] = None,
            inject_fused_attention: bool = False,
            inject_fused_mlp: bool = False,
            use_cuda_fp16: bool = True,
            quantize_config: Optional[BaseQuantizeConfig] = None,
            model_basename: Optional[str] = None,
            use_safetensors: bool = True,
            trust_remote_code: bool = False,
            warmup_triton: bool = False,
            trainable: bool = False,
            disable_exllama: Optional[bool] = None,
            disable_exllamav2: bool = False,
            use_tritonv2: bool = False,
            checkpoint_format: Optional[str] = None,
            **kwargs,
    ):
        """load quantized model from local disk"""
        # If disable_exllamav2 is True, we want to fall back on the exllama kernel and not the cuda/cuda_old ones.
        if disable_exllama is None:
            if disable_exllamav2:
                disable_exllama = False
            else:
                disable_exllama = True

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        if use_qigen and not QIGEN_AVAILABLE:
            logger.warning("Qigen is not installed, reset use_qigen to False.")
            use_qigen = False
        if use_triton and use_tritonv2:
            logging.warn(
                "Both use_triton and use_tritonv2 are set to True. Defaulting to use_triton"
            )
            use_tritonv2 = False
        if (use_triton or use_tritonv2) and not TRITON_AVAILABLE:
            logger.warning("Triton is not installed, reset use_triton to False.")
            use_triton = False
            use_tritonv2 = False
        if not disable_exllama and not EXLLAMA_KERNELS_AVAILABLE:
            logger.warning(
                "Exllama kernel is not installed, reset disable_exllama to True. "
                "This may because you installed auto_gptq using a pre-build wheel "
                "on Windows, in which exllama_kernels are not compiled. To use "
                "exllama_kernels to further speedup inference, you can re-install "
                "auto_gptq from source."
            )
            disable_exllama = True
        if not disable_exllamav2 and not EXLLAMAV2_KERNELS_AVAILABLE:
            logger.warning(
                "Exllamav2 kernel is not installed, reset disable_exllamav2 to True. "
                "This may because you installed auto_gptq using a pre-build wheel "
                "on Windows, in which exllama_kernels are not compiled. To use "
                "exllama_kernels to further speedup inference, you can re-install "
                "auto_gptq from source."
            )
            disable_exllamav2 = True
        if not AUTOGPTQ_CUDA_AVAILABLE:
            logger.warning(
                "CUDA kernels for auto_gptq are not installed, this will result in "
                "very slow inference speed. This may because:\n"
                "1. You disabled CUDA extensions compilation by setting BUILD_CUDA_EXT=0 when install auto_gptq from source.\n"
                "2. You are using pytorch without CUDA support.\n"
                "3. CUDA and nvcc are not installed in your device."
            )

        if use_qigen and QIGEN_AVAILABLE:
            logger.warning("QIgen is active. Ignores all settings related to cuda.")
            inject_fused_attention = False
            inject_fused_mlp = False
            use_triton = False
            disable_exllama = True
            disable_exllamav2 = True

        if not disable_exllamav2 and not disable_exllama:
            logger.warning(
                "You have activated both exllama and exllamav2 kernel. Setting disable_exllama to True and keeping disable_exllamav2 to False"
            )
            disable_exllama = True

        # == 步骤1：准备配置和文件名 == #
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
        )
        # 判断是否是支持的模型
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        # BaseQuantizeConfig 配置文件是否存在
        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path,
                                                                 checkpoint_format=checkpoint_format,
                                                                 **cached_file_kwargs, **kwargs)
        else:
            if not isinstance(quantize_config, BaseQuantizeConfig):
                quantize_config = BaseQuantizeConfig.from_quant_config(quantize_config, checkpoint_format)



        if model_basename is None:
            if quantize_config.model_file_base_name:
                possible_model_basenames = [quantize_config.model_file_base_name]
            else:
                possible_model_basenames = [
                    f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g",
                    "model",
                ]
        else:
            possible_model_basenames = [model_basename]

        quantize_config.model_name_or_path = model_name_or_path

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)

        # 检索（如果需要的话下载）量化检查点。
        is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(model_name_or_path=model_name_or_path,
                                                                                 extensions=extensions,
                                                                                 possible_model_basenames=possible_model_basenames,
                                                                                 **cached_file_kwargs)

        quantize_config.model_file_base_name = true_model_basename

        model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.

        if (not disable_exllama or not disable_exllamav2) and trainable:
            logger.warning(
                "QuantLinear with the exllama backend not does support the trainable mode yet, switching to cuda/cuda_old/triton backend."
            )
            disable_exllama = True
            disable_exllamav2 = True

        elif not (use_triton or use_tritonv2) and trainable:
            logger.warning(
                "QuantLinear with cuda backend not support trainable mode yet, Switch to the pytorch backend."
            )

        # == step2：将模型转换为gptq-model（将Linear替换为QuantLinear） == #
        def skip(*args, **kwargs):
            pass

        if torch_dtype is None:
            if not use_qigen:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        if torch_dtype != torch.float16:
            logger.warning("Overriding use_cuda_fp16 to False since torch_dtype is not torch.float16.")
            use_cuda_fp16 = False

        if not use_qigen:
            torch.nn.init.kaiming_uniform_ = skip
            torch.nn.init.uniform_ = skip
            torch.nn.init.normal_ = skip

            transformers.modeling_utils._init_weights = False

            init_contexts = [no_init_weights()]
            if low_cpu_mem_usage:
                init_contexts.append(accelerate.init_empty_weights(include_buffers=False))

            with ContextManagers(init_contexts):
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
                )

                layers = find_layers(model)
                ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
                for name in list(layers.keys()):
                    if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers) or all(
                            not name.endswith(ignore_layer)
                            for sublist in cls.inside_layer_modules
                            for ignore_layer in sublist
                    ):
                        logger.info(f"The layer {name} is not quantized.")
                        del layers[name]

                make_quant(
                    model,
                    layers,
                    quantize_config.bits,
                    quantize_config.group_size,
                    use_triton=use_triton,
                    disable_exllama=disable_exllama,
                    disable_exllamav2=disable_exllamav2,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act,
                    trainable=trainable,
                    use_tritonv2=use_tritonv2,
                )
                model.tie_weights()

            # == step3: load checkpoint and dispatch == #
            if isinstance(device_map, str) and device_map not in [
                "auto",
                "balanced",
                "balanced_low_0",
                "sequential",
            ]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )
            if isinstance(device_map, dict):
                max_memory = None
            else:
                if device is None and not device_map and not max_memory:
                    device_map = "auto"
                if device is not None:
                    device = torch.device(device)
                    if not max_memory and not device_map:
                        device_map = {"": device.index if device.type == "cuda" else device.type}
                if not isinstance(device_map, dict) and device_map != "sequential":
                    max_memory = accelerate.utils.get_balanced_memory(
                        model=model,
                        max_memory=max_memory,
                        no_split_module_classes=[cls.layer_type],
                        low_zero=(device_map == "balanced_low_0"),
                    )
            if not isinstance(device_map, dict):
                device_map = accelerate.infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type],
                )

            if low_cpu_mem_usage:
                make_sure_no_tensor_in_meta_device(
                    model,
                    use_triton,
                    quantize_config.desc_act,
                    quantize_config.group_size,
                    bits=quantize_config.bits,
                    disable_exllama=disable_exllama,
                    disable_exllamav2=disable_exllamav2,
                    use_tritonv2=use_tritonv2,
                )

            # TODO: move this logic in an awq_utils.py file.
            if quantize_config.checkpoint_format == CHECKPOINT_FORMAT.AWQ_GEMM:
                if is_sharded:
                    raise ValueError(
                        "The loading of sharded checkpoints with AWQ checkpoints is currently not supported. Please raise an issue in AutoGPTQ repository.")

                if use_marlin:
                    raise ValueError(
                        "Tried to load an AWQ model with use_marlin=True. This is currently not supported. Please open an issue in AutoGPTQ repository."
                    )

                model_cache_name, is_cached = quantize_config.get_cache_file_path()

                if is_cached:
                    model_save_name = model_cache_name
                    logger.info(f"Loading an AWQ model, detected a cached repacked weight at {model_save_name}.")
                else:
                    logger.info(
                        "Loading an AWQ model. This requires repacking the weights, and no cached repacked checkpoint was found. Grab a coffee!"
                    )

                    if "safetensors" not in model_save_name:
                        raise NotImplementedError(
                            f"Conversion from AWQ checkpoints is implemented only for safetensors checkpoints, found {model_save_name}"
                        )
                    if quantize_config.bits != 4:
                        raise NotImplementedError(
                            f"Conversion from AWQ checkpoints is supported only for 4 bits models. Found {quantize_config.bits} bits."
                        )
                    gptq_layers = set()
                    non_gptq_params = set()
                    with safe_open(model_save_name, framework="pt") as f:
                        for state_dict_key in f.keys():
                            if (
                                    "qweight" not in state_dict_key
                                    and "qzeros" not in state_dict_key
                                    and "scales" not in state_dict_key
                            ):
                                non_gptq_params.add(state_dict_key)
                                continue

                            # e.g. prefix "model.layers.3.self_attn.k_proj"
                            prefix, _ = state_dict_key.rsplit(".", 1)
                            gptq_layers.add(prefix)

                        new_state_dict = {}

                        for state_dict_key in non_gptq_params:
                            new_state_dict[state_dict_key] = f.get_tensor(state_dict_key)

                        gptq_layers = sorted(gptq_layers)
                        max_layer_name_length = len(max(gptq_layers, key=len))
                        pbar = tqdm(gptq_layers)
                        i = 0
                        for gptq_layer_name in pbar:
                            i += 1
                            desc = f"Unpacking {gptq_layer_name} + '...'"
                            desc = desc + " " * (max_layer_name_length - len(desc))

                            awq_qweight = f.get_tensor(gptq_layer_name + ".qweight")
                            awq_qzeros = f.get_tensor(gptq_layer_name + ".qzeros")
                            awq_scales = f.get_tensor(gptq_layer_name + ".scales")

                            # TODO: add FAST unpacking.
                            unpacked_qweight, unpacked_qzeros = unpack_awq(
                                awq_qweight,
                                awq_qzeros,
                                awq_scales,
                                bits=quantize_config.bits,
                                group_size=quantize_config.group_size,
                            )

                            # TODO: add FAST repacking, this is too slow.
                            desc = f"Repacking {gptq_layer_name}..."
                            desc = desc + " " * (max_layer_name_length + 12 - len(desc))
                            pbar.set_description(desc)
                            gptq_qweight, gptq_qzeros = pack_from_tensors(
                                unpacked_qweight,
                                unpacked_qzeros,
                                awq_scales,
                                bits=quantize_config.bits,
                                group_size=quantize_config.group_size,
                            )

                            new_state_dict[gptq_layer_name + ".qweight"] = gptq_qweight
                            new_state_dict[gptq_layer_name + ".qzeros"] = gptq_qzeros
                            new_state_dict[gptq_layer_name + ".scales"] = awq_scales

                        safe_save(new_state_dict, model_cache_name)
                        model_save_name = model_cache_name

            if use_marlin:
                if is_sharded:
                    raise ValueError(
                        "The loading of sharded checkpoints with Marlin is currently not supported. Please raise an issue in AutoGPTQ repository.")
                if torch.version.hip:
                    raise ValueError(
                        "Can not use Marlin int4*fp16 kernel with AMD ROCm version of PyTorch as the kernel is not compatible. Please do not use `use_marlin=True` when using ROCm devices.")
                if not _validate_marlin_device_support():
                    raise ValueError(
                        f'Can not use Marlin int4*fp16 kernel with a device of compute capability {torch.cuda.get_device_capability()}, the minimum compute capability is 8.0 for Marlin kernel. Please do not use `use_marlin=True`, or please upgrade your GPU ("The more you buy, the more you save." - Taiwanese proverb).')

                # Validate the model can run in Marlin.
                if torch_dtype != torch.float16:
                    raise ValueError("Marlin kernel requires torch_dtype=torch.float16.")
                unsupported_reason = _validate_marlin_compatibility(quantize_config)
                if unsupported_reason is not None:
                    raise ValueError(
                        f"The model {model_name_or_path} can not be converted to use the Marlin kernel for the following reason: {unsupported_reason}, which is not supported by Marlin kernel."
                    )

                # Load the quant linear type we need.
                # TODO: load directy marlin with the right quantlinear class.
                quant_linear_class = dynamically_import_QuantLinear(
                    use_triton=use_triton,
                    desc_act=quantize_config.desc_act,
                    group_size=quantize_config.group_size,
                    bits=quantize_config.bits,
                    disable_exllama=disable_exllama,
                    disable_exllamav2=disable_exllamav2,
                    use_marlin=False,
                    use_tritonv2=use_tritonv2,  # Get the "original" QuantLienar class
                )

                # Prepare model for marlin load.
                #   If stub is marlin serialzed         --> load from directly
                #   If stub has cached marlin version   --> load from the cached versin
                #   Otherwise                           --> convert to marlin, cache, load from cache
                model, model_save_name = prepare_model_for_marlin_load(
                    model=model,
                    quantize_config=quantize_config,
                    quant_linear_class=quant_linear_class,
                    torch_dtype=torch_dtype,
                    current_model_save_name=model_save_name,
                    device_map=device_map,
                )

                # Disable incompatible optimizations.
                if inject_fused_attention or inject_fused_mlp:
                    # TODO: Validate whether that can be used.
                    logger.info("Disabling fused attention and mlp injection because Marlin kernel is used.")
                    inject_fused_attention = False
                    inject_fused_mlp = False

            load_checkpoint_in_model(
                model,
                dtype=torch_dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )

            # TODO: Why are we using this custom function and not dispatch_model?
            model = simple_dispatch_model(model, device_map)
        else:
            # Using QiGen.

            if is_sharded:
                raise ValueError(
                    "The loading of sharded checkpoints with QiGen is currently not supported. Please raise an issue in AutoGPTQ repository.")

            if quantize_config.desc_act:
                NotImplementedError("desc_act=True is not yet supported with QiGen.")
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )

            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
            for name in list(layers.keys()):
                if any(name.startswith(ignore_layer) for ignore_layer in ignore_layers):
                    logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                    del layers[name]

            if model_save_name.endswith(".safetensors"):
                checkpoint = safe_load(model_save_name)
            else:
                checkpoint = torch.load(model_save_name)
            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                disable_exllama=disable_exllama,
                disable_exllamav2=disable_exllamav2,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act,
                trainable=trainable,
                use_qigen=True,
                use_tritonv2=use_tritonv2,
                use_marlin=quantize_config.checkpoint_format == CHECKPOINT_FORMAT.MARLIN,
            )
            preprocess_checkpoint_qigen(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                checkpoint,
            )
            model.load_state_dict(checkpoint)

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any(k in model_config for k in seq_len_keys):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # == step5: (optional) inject optimized module == #
        if inject_fused_attention:
            if cls.fused_attn_module_type is None:
                inject_fused_attention = False
                logger.warning(f"{cls.__name__} hasn't fused attention module yet, will skip inject fused attention.")
            else:
                cls.fused_attn_module_type.inject_to_model(
                    model,
                    use_triton=use_triton,
                    group_size=quantize_config.group_size,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act,
                    trainable=trainable,
                    bits=quantize_config.bits,
                    disable_exllama=disable_exllama,
                    disable_exllamav2=disable_exllamav2,
                    use_tritonv2=use_tritonv2,
                )
        if inject_fused_mlp:
            if cls.fused_mlp_module_type is None:
                inject_fused_mlp = False
                logger.warning(f"{cls.__name__} hasn't fused mlp module yet, will skip inject fused mlp.")
            else:
                cls.fused_mlp_module_type.inject_to_model(model, use_triton=use_triton)

        # Any post-initialization that require device information, for example buffers initialization on device.
        model = autogptq_post_init(model, use_act_order=quantize_config.desc_act)

        model.eval()

        # == step6: (optional) warmup triton == #
        if (use_triton or use_tritonv2) and warmup_triton:
            if use_tritonv2:
                from ..nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
            else:
                from ..nn_modules.qlinear.qlinear_triton import QuantLinear

            QuantLinear.warmup(model, seqlen=model.seqlen)

            if inject_fused_mlp and cls.fused_mlp_module_type is not None:
                cls.fused_mlp_module_type.warmup(model, seqlen=model.seqlen)

        # == step7: make model compatible with peft
        # cls.make_sure_compatible_with_peft(
        #     model,
        #     use_triton,
        #     quantize_config.desc_act,
        #     quantize_config.group_size,
        #     bits=quantize_config.bits,
        #     disable_exllama=disable_exllama,
        #     disable_exllamav2=disable_exllamav2,
        #     use_marlin=use_marlin,
        #     use_qigen=use_qigen,
        # )

        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton or use_tritonv2,
            injected_fused_attention=inject_fused_attention,
            injected_fused_mlp=inject_fused_mlp and (use_triton or use_tritonv2),
            trainable=trainable,
        )


    def build_tokenizer(self):
        if self.model_type not in ['Vit', 'WanT2V', 'WanI2V']:
            assert self.tokenizer_mode in ['fast', 'slow']
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=self.tokenizer_mode, trust_remote_code=True
            )
            if 'Intern' in self.model_type:
                self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None
    def get_catcher(self, first_block_input):
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                if 'output_router_logits' in kwargs:
                    assert kwargs['output_router_logits'] is False
                    kwargs.pop('output_router_logits')
                first_block_input['kwargs'].append(kwargs)
                raise ValueError
        return Catcher

    def move_embed_to_device(self, device):
        for embed_layer in self.get_embed_layers():
            embed_layer.to(device)
        for attention_rotary_layer in self.get_attention_rotary_layers():
            attention_rotary_layer.to(device)

    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)

        Catcher = self.get_catcher(first_block_input)

        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            self.move_embed_to_device('cuda')
            if self.vision_model:
                self.vision_model.cuda()
            if self.vision_projector:
                self.vision_projector.cuda()
            if self.audio_model:
                self.audio_model.cuda()
            if self.audio_projector:
                self.audio_projector.cuda()
            self.blocks[0] = self.blocks[0].cuda()
        self.blocks[0] = Catcher(self.blocks[0])

        for data in calib_data:
            data = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            try:
                if not self.mm_model:
                    self.model(**data)
                else:
                    self.mm_model.generate(**data, max_new_tokens=128, do_sample=False)
            except ValueError:
                pass
        self.first_block_input = first_block_input
        assert len(self.first_block_input) > 0, 'Catch input data failed.'
        if padding_mask:
            for idx in range(len(self.first_block_input['data'])):
                token_num = self.first_block_input['data'][idx].shape[1]
                if token_num != padding_mask[idx].shape[1]:
                    padding_mask[idx] = F.pad(
                        padding_mask[idx],
                        self.get_one_pad_setting(
                            self.tokenizer.padding_side,
                            token_num - padding_mask[idx].shape[1]
                        ),
                        value=1
                    )
        self.padding_mask = padding_mask
        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            if self.vision_model:
                self.vision_model.cpu()
            if self.vision_projector:
                self.vision_projector.cpu()
            if self.audio_model:
                self.audio_model.cpu()
            if self.audio_projector:
                self.audio_projector.cpu()
            self.blocks[0] = self.blocks[0].cpu()
            self.move_embed_to_device('cpu')
        self.blocks[0] = self.blocks[0].module

