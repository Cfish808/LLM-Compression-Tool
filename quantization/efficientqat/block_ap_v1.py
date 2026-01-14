import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

import quantization.efficientqat.int_linear_fake as int_linear_fake
import quantization.efficientqat.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
from my_datasets import get_c4, get_wikitext2, get_redpajama
import copy
import math
import utils
import pdb
import gc
from quantization.efficientqat.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from torch.utils.data import DataLoader
import shutil
import os

from torch.utils.data import Dataset

from utils import efficientqat_utils
from utils.config_utils import to_dotdict, flatten_dict


class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data',
                 off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size % batch_size == 0

        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size // self.batch_size, self.batch_size, self.seqlen, self.hidden_size),
                                    dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data

def update_dataset(layer, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0].to('cpu')
                dataset.update_data(index,new_data)

def ce_loss(student_logits, teacher_logits):

    model_output_log_prob = F.log_softmax(student_logits, dim=2)
    real_output_soft = F.softmax(teacher_logits, dim=2)

    loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
    return loss

def cal_logits(input, other_layers, part_model, attention_mask, position_ids, use_quant=False):
    with torch.no_grad():
        data = copy.deepcopy(input)
        # if type(qlayer) == torch.nn.modules.container.ModuleList:
        #     for layer_index in range(len(qlayer)):
        #         if use_quant: set_quant_state(qlayer[layer_index], weight_quant=True)
        #         else: set_quant_state(qlayer[layer_index], weight_quant=False)
        #         data = qlayer[layer_index](data, attention_mask=attention_mask, position_ids=position_ids)[0]
        # else:
        #     if use_quant: set_quant_state(qlayer,weight_quant=True)  # activate quantization
        #     else: set_quant_state(qlayer, weight_quant=False)
        #     data = qlayer(data, attention_mask=attention_mask, position_ids=position_ids)[0]
        '''
        if not use_quant:
            set_quant_state(qlayer, weight_quant=False)
        data = qlayer(data, attention_mask=attention_mask, position_ids=position_ids)[0]
        '''
        for module in other_layers:
            data = module(data, attention_mask=attention_mask, position_ids=position_ids)[0]
        for module in part_model:
            data = module(data)
    return data

def block_ap(
    model,
    config,
    trainloader,
    valloader,
    logger=None,
):
    args = to_dotdict(flatten_dict(config))

    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    block_size = args.block_size
    # args.type == LLama  获取模型类型
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        print("use the llama-3.1")
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
                ## test
                '''
                pdb.set_trace()
                data = data.to(dev)
                model = model.to(dev)
                test0 = model(data.to(dev))
                embed = model.model.embed_tokens(data)
                test1 = layers[0](embed)
                '''
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 6: start training    
    #import pdb;pdb.set_trace()
    loss_func = torch.nn.MSELoss()
    kd_loss_scale = 0.000
    logger.info(f"kd_loss_scale: {kd_loss_scale}")
    mixed_precision = args.mixed_precision
    if mixed_precision:
        import json
        mask_file = args.maskfile_dir
        salient_columns = json.loads(open(mask_file).read())
        logger.info("Train with a mixed-precision strategy and obtain salient columns.")
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        #if block_size > 1: layer = layers[block_index : block_index + block_size].to(dev)
        #else: layer = layers[block_index]
        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                if not mixed_precision: quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                else:
                    #import pdb;pdb.set_trace()
                    columns = salient_columns[f"{block_index}_{name}_salient_cols"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    dim1, dim2 = module.weight.shape
                    #mask3 = torch.full((dim2,), False, device='cuda')
                    mask3 = torch.full((dim2,), False)
                    mask3[columns] = True

                    columns = salient_columns[f"{block_index}_{name}_Non-salient_cols1"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    #mask2 = torch.full((dim2,), False, device='cuda')
                    mask2 = torch.full((dim2,), False)
                    mask2[columns] = True

                    columns = salient_columns[f"{block_index}_{name}_Non-salient_cols0"]
                    # columns = columns[: (len(columns) // args.group_size) * args.group_size]
                    #mask1 = torch.full((dim2,), False, device='cuda')
                    mask1 = torch.full((dim2,), False)
                    mask1[columns] = True

                    columns_mask = {"mask3": mask3, "mask2": mask2, "mask1": mask1}
                    quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, columns_mask)
                set_op_by_name(qlayer, name, quantlinear)  
                del module  
        qlayer.to(dev)
        
        
        # step 6.2: obtain output of full-precision model for MSE
        set_quant_state(qlayer,weight_quant=False) # deactivate quantization for obtaining ground truth
        #import pdb;pdb.set_trace()
        if args.epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids)
        set_quant_state(qlayer, weight_quant=True)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = efficientqat_utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlayer(input, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                        if kd_loss_scale > 0.0:
                            output_logits = cal_logits(label, layers[(block_index + block_size) : ].to(dev),
                                                       [model.model.norm.to(dev), model.lm_head.to(dev)], 
                                                       attention_mask, position_ids, use_quant=False)
                            quant_logits = cal_logits(quant_out.detach(), layers[(block_index + block_size):].to(dev),
                                                       [model.model.norm.to(dev), model.lm_head.to(dev)], 
                                                       attention_mask, position_ids, use_quant=True)
                            kd_loss = ce_loss(output_logits, quant_logits)
                            reconstruction_loss = reconstruction_loss + kd_loss_scale * kd_loss
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch, position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                            # add the distil loss
                            if kd_loss_scale > 0.0:
                                output_logits = cal_logits(label, layers[(block_index + block_size):].to(dev),
                                                           [model.model.norm.to(dev), model.lm_head.to(dev)], 
                                                           attention_mask, position_ids, use_quant=False)
                                quant_logits = cal_logits(quant_out.detach(), layers[(block_index + block_size):].to(dev),
                                                          [model.model.norm.to(dev), model.lm_head.to(dev)], 
                                                          attention_mask, position_ids, use_quant=True)
                                kd_loss = ce_loss(output_logits, quant_logits)
                                reconstruction_loss = reconstruction_loss + kd_loss_scale * kd_loss

                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False)  # weight has been quantized inplace

        # step 6.7: update inputs of quantization model
        if args.epochs>0:
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids)
        layers[block_index] = qlayer.to("cpu")

        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model


def efficient_get_c4(tokenizer, train_size, val_size,seed, seqlen, test_only,nsamples=256):
    print("get_c4")

    traindata = load_dataset(
        "json",
        data_files={"validation": "c4_local/c4-train.00000-of-01024.json.gz"},  # 改为本地路径或镜像 URL
        split="validation"
    )
    valdata = load_dataset(
        "json",
        data_files={"validation": "c4_local/c4-validation.00000-of-00008.json.gz"},  # 改为本地路径或镜像 URL
        split="validation"
    )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break

        # Ensure we don't go out of bounds when selecting the range
        length = tmp.input_ids.shape[1]
        if length <= seqlen:
            continue  # Skip this sample if it's too short

        i = random.randint(0, length - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])

    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break

        # Ensure we don't go out of bounds when selecting the range
        length = trainenc.input_ids.shape[1]
        if length <= seqlen:
            continue  # Skip this sample if it's too short

        i = random.randint(0, length - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break

        # Ensure we don't go out of bounds when selecting the range
        length = trainenc.input_ids.shape[1]
        if length <= seqlen:
            continue  # Skip this sample if it's too short

        i = random.randint(0, length - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader

def efficient_get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False,pos_Entropy=False, bucket_num=1,model_type="llama"
):
    from_cache = True
    cache_trainloader = f'data_tmp/{name}_{model_type}_train.cache'
    cache_valloader = f'data_tmp/{name}_{model_type}_val.cache'
    if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader) and from_cache:
        trainloader = torch.load(cache_trainloader)
        valloader = torch.load(cache_valloader)
    else:
        if 'wikitext2' in name:
            trainloader, valloader = efficient_get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
        elif 'c4' in name:
            trainloader, valloader = efficient_get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
        elif 'redpajama' in name:
            trainloader, valloader = get_redpajama(tokenizer,train_size,val_size,seed,seqlen,pos_Entropy=pos_Entropy, bucket_num=bucket_num)
        else:
            raise NotImplementedError
        torch.save(trainloader, cache_trainloader)
        torch.save(valloader, cache_valloader)
    return trainloader, valloader