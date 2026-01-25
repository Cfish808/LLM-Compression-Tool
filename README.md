# LLM Compression Tool

LLM Compression Tool is a project designed to optimize machine learning inference processes. The framework combines multiple modules to provide a cohesive and efficient pipeline for model quantization and evaluation.

## Table of Contents

- [LLM Compression Tool](#llm-compression-tool)
  - [Table of Contents](#table-of-contents)
  - [TODO](#todo)
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


## TODO
- PTQ + Sparsification
- PTQ + PTQ
- Base model extension


## Introduction
Although LLMs excel in various NLP tasks, their computational and memory demands may limit their deployment in real-time applications and on resource-constrained devices. This project addresses this challenge by employing quantization techniques to compress these models, ensuring they maintain performance while remaining adaptable to a wide range of scenarios. 

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
- [ ] QAT-LLM 海达
- [x] EFFICIENTQAT 锦宇
- [x] mix-precision QAT 锦宇

### QAT + Low-rank 
- [x] FBI-LLM
- [x] QLoRA
- [x] QA-LoRA
- [x] IR-QLoRA
- [x] OneBit

### todo
- 量化模型的浮点型保存：OmniQuant，QUIP#（无法从向量反推会浮点型），QAT-LLM，QAT + Low-rank
- 基座模型推广，已完成llama, Qwen和部分deepseek模型，MOE架构的未完成
- 校准数据集的优选
- 部分评测任务缺失（Big-BenchHard(BBH), longbench_hotpotqa, CrowS-Pairs/BBO, truthfulga, IFEvaL）
- 稀疏化
- 评测

## Features
- Support for various quantization algorithms, including:
  - [x] RTN 志炀
  - [x] GPTQ 锦宇
  - [x] AWQ 锦宇
  - [x] OmniQuant 博瀚
  - [x] SPQR 海达
  - [x] OWQ 海达
  - [x] SmoothQuant 博瀚
  - [x] QuIP/QUIP# 志炀
  - [ ] BiLLM 海达
  - [ ] QAT-LLM
  - [ ] EfficientQAT
  - [x] QLoRA 志炀
  - [x] QA-LoRA 志炀
  - [x] IR-QLoRA 博翰
  - [x] FBI-LLM 志炀
  - [ ] OneBit
  - [ ] mix-precision QAT
  - [ ] Joint Sparsification and
Quantization (JSQ)
  - [ ] GPTQ + AWQ
  - [ ] SmoothQuant + GPTQ
-  Supports for various base model, including:
    - LLaMA
    - Qwen
    - deepseek
    - Mixtral
    - Qwen2-VL
    - LLaVA
-  Supports a variety of datasets for calibration and testing, including CEVAL, CMMLU, BOSS, lm-evaluation-harness, and user-provided custom datasets.
- Allows combination of different quantization methods within the same model for enhanced performance and efficiency.
- Offers tools for users to quantize and evaluate their own models, ensuring optimal performance tailored to specific needs.

## Installation

### Prerequisites

- **Python**: Python 3.7 or higher must be installed on your system. 

- **Libraries**: Ensure that all necessary libraries are installed. These libraries are listed in the `requirements.txt` file. 

### Install from Source

1. Clone the repository:

    ```bash
    git clone git@gitee.com:zhangyujie0218/model-quantification-tool.git
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
4. Install the package in editable mode:

    ```bash
    pip install -e .
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
    basemodel = BaseModel(config)
    tokenizer = basemodel.build_tokenizer()
    model=basemodel.build_model()

    new_model=basemodel.replace_module(model, exclude_layers=config.quant.skip_layers, include_layers=['.*'])
    calibrate = get_calibrate_loader(tokenizer=tokenizer, calibrate_config=config.quant.data)

    new_model=llama_sequential(model=new_model, calibrate_data=calibrate, **config.quant)
    new_model = new_model.to("cuda")
    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {tokenizer}')

    if config.save:
        model = basemodel.replace_module(new_model, module_type=LinearQuantHub, new_module_type="", display=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)

    if config.eval:
        eval_config=config.eval
        benchmark = Benchmark()
        model=model.to(eval_config.get('device',"cpu"))
        if eval_config.ppl is not None and eval_config.ppl is not "":
            results_ceval = benchmark.eval_ppl(model, tokenizer, nsamples=eval_config.nsamples,seqlen=eval_config.seq_len, test_datasets=eval_config.ppl)
            logging.info("\n转换后的模型:")
            logging.info(results_ceval)
        else:
            pass
```

### Quantization
Below is an example of how to quantize model on various datasets. 
```
base_model:
    type: Llama
    path: /home/yejinyu/llama2_7b/Llama-2-7b-ms
    torch_dtype: auto
    tokenizer_mode: fast
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
      nsamples: 1
      seqlen: 2048
      download: True
      path: eval data path
      batch_size: 1
      seed: 42
```

### Evaluation 
Below is an example of how to evaluate a quantized model on various datasets. 
```
eval:
    task: [ppl]
    dataset: [wikitext2,c4]
    download: True
    path: eval data path
    seq_len: 2048
    nsamples: all
    device: cuda
    seqlen: 2048
    bs: 1
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
      <td rowspan="3">Model_quant_tool AWQ</td>
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