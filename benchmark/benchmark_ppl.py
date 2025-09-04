import argparse
import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.benchmark import Benchmark


def main(args: argparse.Namespace):
    logging.info(args)

    # Initialize the model and tokenizer
    if args.quantized_model and not args.pretrain:
        model = torch.load(args.quantized_model)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model = model.to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Benchmark ceval
    logging.info("\nEvaluating the model on the ppl_benchmark...")


    benchmark = Benchmark()
    results = benchmark.eval_ppl(model=model, tokenizer=tokenizer, nsamples=args.nsamples, seqlen=args.seqlen,
                                 test_datasets=args.eval_tasks)
    logging.info(results)

    # Output ceval results if specified
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model on ceval_benchmark.')
    parser.add_argument('--model', type=str, default="/home/yejinyu/llama2_7b/output/llama27b_miom_2",
                        help='Path to the model file or Huggingface model identifier.')
    parser.add_argument('--quantized-model', default="/ssd/yejinyu/llama2_7b/output/llama27b_model_quant_tool2.pt",
                        help='Whether to use a quantized model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on.')
    parser.add_argument('--eval-tasks', nargs='+', default=["wikitext2"])
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--nsamples', type=str, default='all')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--output-json', type=str, default=None, help='Path to save the ceval results in JSON format.')
    args = parser.parse_args()
    main(args)
