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
    - [Quantization](#quantization)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
    - [Quantize \&\& Evaluation(Command-line)](#quantize--evaluationcommand-line)
    - [Arguments](#arguments)
    - [Example Command](#example-command)
    - [Quantize \&\& Evaluation (Code)](#quantize--evaluation-code)
  - [OOD Benchmark Results](#ood-benchmark-results)
    - [Perplexity (PPL) of the LLaMA-2-7B Model](#perplexity-ppl-of-the-llama-2-7b-model)
    - [Evaluation Of Quantized Model Capabilities](#evaluation-of-quantized-model-capabilities)
  - [Cite](#cite)


## TODO
- The code for the GPTQ implementation needs optimization. The current 3-bit perplexity (PPL) is around 8, which does not match the performance level of AutoGPTQ.
- The supported base models should be expanded. The current implementation only supports LLaMA 2. Expanding to include other models would significantly improve the paper's contribution and relevance.

## Introduction
Although LLMs excel in various NLP tasks, their computational and memory demands may limit their deployment in real-time applications and on resource-constrained devices. This project addresses this challenge by employing quantization techniques to compress these models, ensuring they maintain performance while remaining adaptable to a wide range of scenarios. 


## Features
- Support for various quantization algorithms, including:
  - RTN
  - GPTQ
  - AWQ
  - OmniQuant
  - SPQR
  - OWQ
  - SmoothQuant
  - QuIP
  - SqueezeLLM
  - BiLLM
  - QAT-LLM
  - EfficientQAT
  - PEQA
  - QLoRA
  - QA-LoRA
  - IR-QLoRA
  - LCQ
  - OneBit
  - mix-precision QAT
  - Joint Sparsification and
Quantization (JSQ)
  - GPTQ + AWQ
  - SmoothQuant + GPTQ
-  Supports for various base model, including:
    -  LLaMA
    -  Qwen
    -  deepseek
    -  Mixtral
    -  Qwen2-VL
    -  LLaVA
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
python mian.py --config.yml config/llama_gptq.yml
```
### Quantization 
Below is an example of how to set up the quantization process for a model. For detailed information on all available quantization configuration options, please refer to the [quantization configuration guide](configs/README.md).
```
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize.export import export_module

# Define paths for the pre-trained model and quantized model
model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = 'llama-2-7b-quant.pth'

# Define quantization configuration
quant_config = {
    "algo": "rtn",
    "kwargs": {
        "w_dtype": "int4",           
        "a_dtype": "float16",        
        "device": "cuda",
        "offload": "cpu",
        "w_qtype": "per_channel",
        "w_has_zero": False,
        "w_unsign": True,
        "quantization_type": "static",
        "layer_sequential": True,
        "skip_layers": [             
            "lm_head"
        ]
    },
    "calibrate_config": {
        "name": "wikitext2",
        "split": "train",
        "nsamples": 1,
        "seqlen": 2048
    }}

# Load the pre-trained Hugging Face model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()  
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Quantize the model
quantize_model = quantize(model=model, tokenizer=tokenizer, quant_config=quant_config)

# print('model device', model.device)
quantize_model.to('cuda')
print(quantize_model)
input_text = "Llama is a large language model"

input_ids = tokenizer.encode(input_text, return_tensors="pt").to(quantize_model.device)

output = model.generate(input_ids, max_length=20, num_return_sequences=1, do_sample= False)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

# Save the quantized model
model = export_module(model)
torch.save(model, quant_path)
```

### Evaluation 
Below is an example of how to evaluate a quantized model on various datasets. For a full explanation of all input parameters used in the evaluation functions, please refer to the [detailed parameter documentation](benchmark/PARAMETER_DETAILS.md).
```
import torch
from mi_optimize import Benchmark
from transformers import LlamaTokenizer, AutoModelForCausalLM

# model_path = 'meta-llama/Llama-2-7b-hf'
quantize_model_path = 'llama-2-7b-quant.pth'
# Load Benchmark
benchmark = Benchmark()

# Load Model && tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = torch.load(quantize_model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()

# Evaluate Perplexity (PPL) on various datasets
test_dataset = ['wikitext2']  
results_ppl = benchmark.eval_ppl(model, tokenizer, test_dataset)
print(results_ppl)

# Evaluate the model on the ceval_benchmark
results_ceval = benchmark.eval_ceval(model, tokenizer, model_type='baichuan', subject='all', num_shot=0)
print(results_ceval)

# Evaluate the model on the mmlu benchmark
results_cmmlu = benchmark.eval_cmmlu(model, tokenizer, model_type='baichuan', subject='all', num_shot=0)
print(results_cmmlu)

# Evaluate the model on the BOSS benchmark
results_boss = benchmark.eval_boss(model, tokenizer, test_dataset='QuestionAnswering_advqa', split='test', ICL_split='test', num_shot=0)
print(results_boss)

# Evaluate using lm-evaluation-harness
eval_tasks = [
    "winogrande",       
    "piqa",             
    "hellaswag",       
]
results_lm_evaluation = benchmark.eval_lmeval(model, tokenizer, eval_tasks, num_shot=5)
print(results_lm_evaluation)
```

### Inference
```
import torch
import time
from transformers import LlamaTokenizer, TextGenerationPipeline
from mi_optimize.export import qnn

# Path to the quantized model
quant_path = 'llama-2-7b-quant.pth'

# Path to the tokenizer
tokenizer_path = 'meta-llama/Llama-2-7b-hf'

# Load the quantized model
model = torch.load(quant_path)

model = model.cuda()

# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# # Input prompt
prompt = "Llama is a large language model"

# # Tokenize the input prompt
inputs_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

# Choose the backend for inference ('naive', 'vllm', 'tensorrt')
backend = 'naive'   

if backend == 'naive':
    start_time = time.time()
    output = model.generate(inputs_ids, max_length=100, num_return_sequences=1, do_sample= False)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded_output)
    print(f'quantize time {time.time() - start_time}')

elif backend == 'vllm':
    pass  # This will be added soon
```

### Quantize && Evaluation(Command-line)
To run the quantization and evaluation pipeline, use the provided `quantization.py` script. Below are the command-line arguments and an example command.

### Arguments

- `--model-path` (str): Path to the pre-trained language model (LLM) that you want to quantize.
- `--algo` (str): Quantization algorithm to use. Choices: `rtn`, `gptq`, `awq`, `spqr`, `smoothquant`, `quip`.
- `--wbit` (int): Number of bits for weight quantization.
- `--abit` (int): Number of bits for activation quantization.
- `--w-groupsize` (int): Group size for quantization. The group size determines how weights and activations are grouped together during the quantization process. (Options: 32, 64, 128)
- `--benchmark` (str): Specifies the benchmark dataset to use for evaluation.
- `--num-calibrate` (int): Number of samples used for calibration during the quantization process.
- `--num-shot` (int): Number of few-shot examples used for evaluation.
- `--calibrate-name` (str): Path to the calibration dataset used for quantization.
- `--seqlen` (int): Sequence length for the input data.
- `--device` (str): Device to run the quantization (e.g., `cuda:0`).
- `--offload` (flag): Enables offloading to save memory during quantization.
- `--skip-layers` (str): Specifies the layers to exclude from quantization.
- `--block-sequential` (flag): Uses block-sequential quantization.
- `--layer-sequential` (flag): Uses layer-sequential quantization.
- `--half` (flag): Uses half-precision floating point during quantization.
- `--save` (str): Path to save the quantized model after quantization.
### Example Command

```
python examples/llama/quantization.py \
    --model /home/user/models/Llama-2-7b-hf \
    --algo awq \
    --w-bits 4 \
    --w-groupsize 128 \
    --device cuda:0 \
    --num-calibrate 128 \
    --calibrate-name 'c4' \
    --benchmark ppl \
    --num-shot 1 \
    --save /home/user/tmd-optimize/models
```
or
```
bash scripts/run_llama.sh
```
### Quantize && Evaluation (Code)
For quantization and evaluation within your code, refer to the provided script and customize it according to your requirements. You can adjust the parameters and integrate them into your codebase for seamless quantization and evaluation.

```
from transformers import AutoModelForCausalLM, LlamaTokenizer
from mi_optimize import quantize
from mi_optimize import Benchmark

# Define paths for the pre-trained model and quantized model
model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = 'llama-2-7b-quant.pth'

# Define quantization configuration
quant_config = {
    "algo": "rtn",
    "kwargs": {'w_dtype': "int4", 'a_type': "float16"},
    "calibrate_data": "wikitext2"  # select from  ['wikitext2', 'c4', 'ptb', 'cmmlu', 'cmmlu_hm', 'cmmlu_st', 'cmmlu_ss', 'NaturalLanguageInference_mnli']
 }

# Load the pre-trained Hugging Face model
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half()  
tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Quantize the model
model = quantize(model, quant_config=quant_config)

benchmark = Benchmark()
# Evaluate Perplexity (PPL) on various datasets
test_dataset = ['wikitext2']  
results_ppl = benchmark.eval_ppl(model, tokenizer, test_dataset)
print(results_ppl)
```
## OOD Benchmark Results
Below are some test results obtained from the Out-of-Distribution (OOD) benchmark evaluation:

### Perplexity (PPL) of the LLaMA-2-7B Model


### Evaluation Of Quantized Model Capabilities



## Cite
If you found this work useful, please consider citing
