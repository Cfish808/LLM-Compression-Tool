# COSQuant

We introduce a versatile quantization toolkit COSQuant that integrates a diverse array of quantization techniques and supports the flexible combination of multiple model compression strategies.

## Table of Contents

- [COSQuant](#cosquant)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from Source](#install-from-source)
  - [Usage](#usage)
    - [Quick Start](#quick-start)
    - [Quantization, Save and Evaluation](#quantization-save-evaluation)
    - [Quantization](#quantization)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [OOD Benchmark Results](#ood-benchmark-results)
    - [Perplexity (PPL) of the LLaMA-2-7B Model](#perplexity-ppl-of-the-llama-2-7b-model)
    - [Evaluation Of Quantized Model Capabilities](#evaluation-of-quantized-model-capabilities)
  - [Cite](#cite)



## Introduction
Although LLMs excel in various NLP tasks, their computational and memory demands may limit their deployment in real-time applications and on resource-constrained devices. This project addresses this challenge by employing quantization techniques to compress these models, ensuring they maintain performance while remaining adaptable to a wide range of scenarios. 


## Features

### PTQ
- [x] AWQ
- [x] GPTQ
- [x] SmoothQuant
- [x] OmniQuant
- [x] QuIP/QUIP#
- [x] OWQ
- [x] SpQR
- [x] BiLLM
- [x] RTN


### QAT
- [x] QAT-LLM 
- [x] EFFICIENTQAT 
- [x] mix-precision QAT 

### QAT + Low-rank 
- [x] FBI-LLM
- [x] QLoRA
- [x] QA-LoRA
- [x] IR-QLoRA
- [x] OneBit

### PTQ + PTQ
  - [x] GPTQ + AWQ
  - [x] SmoothQuant + GPTQ

###  Supports for various base model, including:
    - LLaMA
    - Qwen
    - deepseek


## Installation

### Prerequisites

- **Python**: Python 3.7 or higher must be installed on your system. 

- **Libraries**: Ensure that all necessary libraries are installed. These libraries are listed in the `requirements.txt` file. 

### Install from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/Cfish808/LLM-Compression-Tool.git
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    conda create -n {env_name} python=3.10
    conda activate {env_name}
    ```

3. Use pip to install packages from requirements.
   ```
   pip install -r requirements.txt
   ```

## Usage
### Quick Start
```
python main.py --config config/llama_gptq.yml
```
### Quantization, Save and Evaluation
Below is an example of how to set up the quantization process for a model. For detailed information on all available quantization configuration options, please refer to the [quantization configuration guide](configs/README.md).
```
def main(config):
    SEQUENTIAL_COMPRESSION_MAP = {
        "llama": llama_sequential,
        "qwen": qwen_sequential,
        "deepseek": deepseek_sequential,
    }
    OMNIQUANT_COMPRESSION_MAP = {
        "llama": llama_omniquant,
        "deepseek": deepseek_omniquant,
    }
    compression_func = SEQUENTIAL_COMPRESSION_MAP[config.base_model.type.lower()]
    
    new_model = None
    if config.get("quant", False):

        if config.quant.method in ["qlora", "qalora","irlora"]:
            model,tokenizer = get_accelerate_model(config.base_model,config.quant.method)
            calibrate = make_data_module(tokenizer=tokenizer, args=config.quant.data)
        elif config.quant.method == "onebit" or config.quant.method == "qat-llm":
            from quantization.onebit.core import get_train_args
            model_args, data_args, training_args, finetuning_args = get_train_args(config.quant.args)
            model,tokenizer = load_model_and_tokenizer(config.quant.method, model_args,finetuning_args,training_args.do_train)
            calibrate = get_dataset_loader(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
        elif config.quant.method == "efficientqat_e2e":
            pass
        else:
            model, tokenizer, basemodel = get_model(config)
            if config.quant.data.name == "cola":
                calibrate = cola_calibrate_loader(tokenizer=tokenizer, model=model, **config.quant.data)
            else:
                calibrate = get_calibrate_loader(tokenizer=tokenizer, **config.quant.data)
        
        if config.get("sparse", False):
            # Handling n:m sparsity
            prune_n, prune_m = 0, 0
            device = model.device
            model.seqlen = config.quant.seqlen
            if config.sparse.sparsity_type != "unstructured":
                assert config.sparse.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
                logger.info(f"sparse config: {config.sparse}")
                logger.info(f"sparsity type: {config.sparse.sparsity_type}")
                prune_n, prune_m = map(int, config.sparse.sparsity_type.split(":"))
            
            if config.sparse.sparsity_ratio != 0:
                logger.info("pruning starts")
                if config.sparse.method == "wanda":
                    prune_wanda(config.sparse, calibrate, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif config.sparse.method == "magnitude":
                    prune_magnitude(config.sparse, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif config.sparse.method == "sparsegpt":
                    prune_sparsegpt(config.sparse, calibrate, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif "ablate" in config.sparse.method:
                    prune_ablate(config.sparse, calibrate, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                
                sparsity_ratio = check_sparsity(model)
                logger.info(f"sparsity sanity check {sparsity_ratio:.4f}")
            else:
                logger.info("sparsity ratio is 0, no pruning")

        if config.quant.method == "omniquant":
            omniquant_compression_func = OMNIQUANT_COMPRESSION_MAP[config.base_model.type.lower()]
            model = omniquant_compression_func(config.base_model.path, model, calibrate, config.quant, logger=logger)
            new_model = model
            logger.info(f"omniquant compression_func: {omniquant_compression_func.__name__}")
        elif config.quant.method == "quip_sharp":
            from quantization.llama_seq import llama_quipsharp
            from quantization.qwen_seq import qwen_quipsharp
            from quantization.deepseek_seq import deepseek_quipsharp
            QUIP_SHARP_COMPRESSION_MAP = {
                "llama": llama_quipsharp,
                "qwen": qwen_quipsharp,
                "deepseek": deepseek_quipsharp,
            }
            quip_sharp_compression_func = QUIP_SHARP_COMPRESSION_MAP[config.base_model.type.lower()]
            logger.info(f"quip_sharp compression_func: {quip_sharp_compression_func.__name__}")
            model = quip_sharp_compression_func(calibrate,config)
            new_model = model
        elif config.quant.method == "fbi_llm":
            from quantization.fbi_llm.fbi_train import train_fbi
            model = train_fbi(model, calibrate, config)
            new_model = model
        elif config.quant.method == "efficientqat_block":
            trainloader, valloader = get_loaders(
                config.quant.data.name,
                tokenizer,
                config.quant.data.train_size,
                config.quant.data.val_size,
                seed=config.quant.data.seed,
                seqlen=config.quant.data.training_seqlen,
                model_type=config.base_model.type,
            )
            block_ap(
                model,
                config,
                trainloader,
                valloader,
                logger,
            )
        elif config.quant.method == "efficientqat_e2e":
            from quantization.efficientqat.e2e import train
            new_model,tokenizer = train(config)
        elif config.quant.method in ["qlora", "qalora","irlora"]:
            from quantization.qlora.qlora import train
            model = train(model=model,tokenizer=tokenizer,calibrate_data=calibrate, args=config.quant.args)
            new_model = model
        elif config.quant.method == "onebit" or config.quant.method == "qat-llm":
            from quantization.onebit.kd import run_kd
            model = run_kd(model=model,tokenizer=tokenizer,dataset=calibrate,model_args=model_args,data_args=data_args,training_args=training_args)
            new_model = model
        else:
            new_model = basemodel.replace_module(model, exclude_layers=config.quant.skip_layers, include_layers=['.*'])
            new_model = compression_func(model=new_model, calibrate_data=calibrate, **config.quant)
            logger.info(f'model: {model}')
            logger.info(f'tokenizer: {tokenizer}')

    if config.get("save", False) and config.get("quant", False):
        if config.quant.method == "efficientqat_e2e":
            model = trans_e2e2llama_model(new_model, mixed_precision=config.quant.mixed_precision, maskfile_dir=config.quant.maskfile_dir)
        elif config.quant.method == "efficientqat_block":
            model = trans_blockwise2llama_model(model)
        # config.quant.method in ["gptq"]:
        else:
            model = basemodel.replace_module(new_model, module_type=LinearQuantHub, new_module_type="", display=True)

        gen_config = model.generation_config
        gen_config.do_sample = True
        model.save_pretrained(config.save)
        tokenizer.save_pretrained(config.save)

    if config.get("eval", False):
        if new_model is None:
            config.base_model.path = config.save
            if config.quant and config.quant.method == "onebit":
                from transformers import BitLlamaForCausalLM, AutoTokenizer
                new_model = BitLlamaForCausalLM.from_pretrained(config.base_model.path)
                tokenizer = AutoTokenizer.from_pretrained(config.base_model.path, use_fast=False)
            elif config.quant and config.quant.method == "qat-llm":
                from transformers import QatLlamaForCausalLM, AutoTokenizer
                new_model = QatLlamaForCausalLM.from_pretrained(config.base_model.path)
                tokenizer = AutoTokenizer.from_pretrained(config.base_model.path, use_fast=False)
            else:
                new_model, tokenizer, _ = get_model(config)
        evals = config.eval
        device = evals.get('device', "cpu")

        if new_model.device.type == device:
            model = new_model
        else:
            model = new_model.to(device)

        for eval_config in evals.tasks:
            eval_config = dict(eval_config)
            run_evaluation(model, tokenizer, device,**eval_config)
```

### Quantization
Below is an example of how to quantize model on various datasets. 
```
base_model:
    type: Llama
    path: /netcache/huggingface/llama2_7b/
    torch_dtype: torch.float16
    tokenizer_mode: fast
    device_map: auto
quant:
    method: gptq
    skip_layers: [ lm_head ]
    seqlen: 2048
    device: cuda
    weight:
      wbit: 3
      abit: 16
      offload: cpu
      block_sequential: True
      layer_sequential: True
      w_qtype: per_group
      groupsize: 128
      blocksize: 128
      percdamp: 0.01
      actorder: False
    special:
      actorder: True
      static_groups: False
      percdamp: 0.01
      blocksize: 128
      true_sequential: True
    data:
      name: c4
      nsamples: 128
      seqlen: 2048
      download: False
      path: /netcache/huggingface/c4_local/c4-train.00000-of-01024.json.gz
      batch_size: 32
      split: train
      seed: 42

save: /home/yejinyu/llama2_7b/output/llama27b_miom_2
```

### Evaluation 
Below is an example of how to evaluate a quantized model on various datasets. 
```
eval:
  device: cuda
  tasks: [
    {
      task: acc,
      datasets: [winogrande, arc_easy, arc_challenge, piqa, mmlu, openbookqa, mathqa],
      batch_size: auto,
      num_fewshot: 2
    },
    {
      task: ppl,
      datasets: [wikitext2, c4],
      download: True,
      seqlen: 2048,
      nsamples: all
    }
  ]
```

### Inference


## OOD Benchmark Results
Below are some test results obtained from the Out-of-Distribution (OOD) benchmark evaluation:

### Perplexity (PPL) of the LLaMA-2-7B Model

<table>
  <thead>
    <tr>
      <th></th>
      <th>bit-width</th>
      <th>wikitext2</th>
      <th>c4</th>
      <th></th>
      <th>bit-width</th>
      <th>wikitext2</th>
      <th>c4</th>
      <th></th>
      <th>bit-width</th>
      <th>wikitext2</th>
      <th>c4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">COSQuant AWQ</td>
      <td>W2A16</td>
      <td>206385.3</td>
      <td>150178.06</td>
      <td rowspan="3">AWQ</td>
      <td>W2A16</td>
      <td>222486.37</td>
      <td>168183.6</td>
      <td rowspan="3">LLMC AWQ</td>
      <td>W2A16</td>
      <td>nan</td>
      <td>8895.39</td>
    </tr>
    <tr>
      <td>W3A16</td>
      <td>6.54</td>
      <td>8.72</td>
      <td>W3A16</td>
      <td>6.25</td>
      <td>8.23</td>
      <td>W3A16</td>
      <td>6.67</td>
      <td>8.88</td>
    </tr>
    <tr>
      <td>W4A16</td>
      <td>5.77</td>
      <td>7.62</td>
      <td>W4A16</td>
      <td>5.6</td>
      <td>7.39</td>
      <td>W4A16</td>
      <td>5.66</td>
      <td>7.5</td>
    </tr>
  </tbody>
</table>


### Evaluation Of Quantized Model Capabilities



## Cite
If you found this work useful, please consider citing
