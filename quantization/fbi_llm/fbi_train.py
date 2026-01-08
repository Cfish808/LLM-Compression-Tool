from datetime import datetime
from pytz import timezone
import time
from functools import partial
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from transformers import LlamaConfig

from quantization.fbi_llm.model_utils.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from quantization.fbi_llm.qat.replace_module import (
    replace_with_learnable_binarylinear, 
    check_para_state
)
import json
from pathlib import Path

from quantization.fbi_llm.main_utils import (
    get_cosine_lr_decay_fn,
    get_grad_norm)


PROJECT_NAME = 'FBI-LLM'
TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]



def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    return {'input_ids': token_ids[:, :-1], 'labels': token_ids[:, 1:]}


def train_chunk(fabric,
                model,
                teacher,
                use_kd,
                optimizer,
                lr_schedule_fn,
                examples,
                per_device_batch_size,
                accumulate_grad_batches,
                chunk_name,
                GRAD_NORM_CLIP):

    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'Training chunk {chunk_name} (global_micro_batch_size='
             f'{per_device_batch_size * fabric.world_size}, '
             f'accumulate_grad_batches={accumulate_grad_batches})')
    step = 0
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        current_batch_examples = examples[i:i+per_device_batch_size]
        if not current_batch_examples:
            continue
            
        batch = collate_fn(
            examples=current_batch_examples, device=fabric.device)
        if batch is None:
            continue
            
        input_ids, labels = batch['input_ids'], batch['labels']
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            if use_kd == 1:
                student_logits = model(input_ids).logits
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
                teacher_prob = F.softmax(teacher_logits, dim=2).clone().detach()
                loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), teacher_prob.reshape((-1, teacher_prob.size(-1))))
            elif use_kd == 2:
                student_logits = model(input_ids).logits
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits
                teacher_prob = F.softmax(teacher_logits, dim=2).clone().detach()
                kd_loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), teacher_prob.reshape((-1, teacher_prob.size(-1))))
                ar_loss = torch.nn.functional.cross_entropy(
                    student_logits.reshape((-1, student_logits.size(-1))), labels.reshape(-1))
                loss = 0.5*ar_loss + 0.5*kd_loss
            else:
                logits = model(input_ids).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape((-1, logits.size(-1))), labels.reshape(-1))
                
            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': loss.item(),
            'learning_rate': lr,
            'step': step,
            'speed(#tok/s/gpu)': int(input_ids.numel() / (time.time() - t0)),
        }
        if use_kd == 2:
            log['ar_loss'] = ar_loss.item()
            log['kd_loss'] = kd_loss.item()
        
        if not is_accumulating:
            log['grad_norm'] = grad_norm

        example_batch_idxes.set_postfix(log)



def train_fbi(teacher,calibrate,config):
    quant_config = config.quant
    fbi_args = quant_config['fbi_args']

    n_nodes = fbi_args['n_nodes']
    n_devices_per_node = fbi_args['nproc_per_node']
    per_device_batch_size = fbi_args['per_device_train_batch_size']
    accumulate_grad_batches = fbi_args['gradient_accumulation_steps']
    use_kd = fbi_args['use_kd']
    config_path = fbi_args['config_path']

    LEARNING_RATE = float(fbi_args['learning_rate'])
    END_LEARNING_RATE = float(fbi_args['end_learning_rate'])
    WARMUP_GRAD_STEPS = int(fbi_args['warmup_steps'])
    GRAD_NORM_CLIP = float(fbi_args['max_grad_norm'])
    WEIGHT_DECAY = float(fbi_args['weight_decay'])
    BETA1 = float(fbi_args['beta1'])
    BETA2 = float(fbi_args['beta2'])
    PRECISION = fbi_args['precision']
    ACCELERATOR = 'cuda' 

    fabric = L.Fabric(
        accelerator=ACCELERATOR,
        num_nodes=n_nodes,
        devices=n_devices_per_node,
        precision=PRECISION,
        strategy=FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer}),
            activation_checkpointing_policy={LlamaDecoderLayer},
            cpu_offload=False,
            limit_all_gathers=True))
        # strategy = 'auto')
    fabric.launch()

    with Path(config_path).open('r') as r_f:
        _config = json.load(r_f)
    config = LlamaConfig(**_config)
    model = LlamaForCausalLM(config=config)
            
    model = replace_with_learnable_binarylinear(model, 'column', ['lm_head'])

    if fabric.global_rank == 0:
        print(config)
        check_para_state(model)

    optimizer = torch.optim.AdamW( 
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False)
 
    model, optimizer = fabric.setup(model, optimizer)

    if use_kd > 0:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.config.use_cache = False
        teacher = fabric.setup(teacher)
    else:
        teacher = None
        
    torch.cuda.empty_cache()

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    
    total_examples = len(calibrate)
    total_steps = total_examples // global_micro_batch_size
    
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=WARMUP_GRAD_STEPS * accumulate_grad_batches,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)
    

    examples = calibrate 
    if fabric.world_size > 1:
        n_examples = len(examples) // global_micro_batch_size * global_micro_batch_size
        example_idxes = np.arange(n_examples)
        my_indices = example_idxes[fabric.global_rank:n_examples:fabric.world_size]
        examples = [examples[i] for i in my_indices]

    train_chunk(
        fabric=fabric,
        model=model,
        teacher=teacher,
        use_kd=use_kd,
        optimizer=optimizer,
        lr_schedule_fn=lr_schedule_fn,
        examples=examples,
        per_device_batch_size=per_device_batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        chunk_name="full_calibrate",
        GRAD_NORM_CLIP=GRAD_NORM_CLIP)