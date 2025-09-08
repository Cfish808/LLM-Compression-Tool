#!/bin/bash

# export HF_ENDPOINT='https://hf-mirror.com'
# export HF_HOME='/home/yubohan/huggingface'
# export HF_DATASETS_CACHE='/home/yubohan/huggingface'
export CUDA_VISIBLE_DEVICES=0

config_path="/home/yubohan/code/model-quantification-tool/config/llama_smoothquant.yml"
# config_path="/home/yubohan/code/model-quantification-tool/config/llama_gptq.yml"
config_name=$(basename $config_path .yml)


nohup python main.py \
    --config $config_path \
    > logs/${config_name}.log 2>&1 &