# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging
import torch
import transformers
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig
)

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)




def train(model,tokenizer, calibrate_data, args):
    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    training_args_dict = args.training_args.to_dict() if hasattr(args.training_args, 'to_dict') else dict(args.training_args)
    gen_args = training_args_dict.pop('generation_args', None)
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    if gen_args is not None:
        training_args.generation_config = GenerationConfig(**gen_args)

    transformers.logging.set_verbosity_info()
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in calibrate_data.items() if k != 'predict_dataset'},
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)

    all_metrics = {"run_name": args.run_name}

    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    return model

if __name__ == "__main__":
    train()
