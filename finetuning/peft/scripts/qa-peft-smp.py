import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq,AutoConfig

from datasets import load_from_disk
from tqdm import tqdm
from utils import parse_args
from utils import parse_args,is_local_main_process,is_main_process,wait_for_everyone,main_process_first
import math
import smdistributed.modelparallel
import smdistributed.modelparallel.torch as smp
import time
import numpy as np

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType # New

@smp.step
def train_step(model, batch):
    loss = model(**batch)["loss"]
    #loss = outputs.loss
    model.backward(loss)
    return loss

@smp.step
def test_step(model, batch):
    loss = model(**batch)["loss"]
    return loss

def compute_num_params(model):
    num_params = 0
    seen = set()
    for p in model.parameters():
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape) 
            else:
                num_params += np.prod(p.size())
    
    return num_params 

def main():
    args = parse_args()

    model_id = "philschmid/flan-t5-xxl-sharded-fp16" # New
    config = AutoConfig.from_pretrained(model_id) 
    
    with smp.model_creation(dtype=torch.bfloat16):
        model =  AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                       cache_dir=f"/tmp",
                                                       config=config,
                                                       torch_dtype=torch.bfloat16,
                                                       # load_in_8bit=True, device_map="auto" # New
                                                      )
        # New
        lora_config = LoraConfig(
             r=16,
             lora_alpha=32,
             target_modules=["q", "v"],
             lora_dropout=0.05,
             bias="none",
             task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        
        model.config.use_cache = False

    num_params = compute_num_params(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    train_dataset = load_from_disk(args.training_dir)
    eval_dataset = load_from_disk(args.test_dir)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_sampler = torch.utils.data.DistributedSampler(
                        train_dataset,
                        shuffle=True,
                        seed=args.seed,
                        rank=smp.dp_rank(),
                        num_replicas=smp.dp_size(),
                        drop_last=True,
                    )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                        eval_dataset,
                        shuffle=True,
                        seed=args.seed,
                        rank=smp.dp_rank(),
                        num_replicas=smp.dp_size(),
                        drop_last=True,
                    )
    
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size,
                                  pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, 
                                 sampler = eval_sampler,
                                 collate_fn=data_collator, 
                                 batch_size=args.per_device_eval_batch_size, 
                                 pin_memory=True
    )

    model = smp.DistributedModel(model, trace_device="gpu")

    # enable activation checkpointing.
    m = model.get_module()
    for c in m.encoder.block.children():
        smp.set_activation_checkpointing(c)
    for c in m.decoder.block.children():
        smp.set_activation_checkpointing(c)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate
                                 )
    
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=0,
                                                   num_training_steps=(len(train_dataloader) * args.epochs),
    )
    optimizer = smp.DistributedOptimizer(optimizer)

    # Train!
    total_batch_size = args.per_device_train_batch_size * smp.size() * args.gradient_accumulation_steps

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.epochs
    
    if is_main_process(smp.rank()):
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"Number of eval examples ={len(eval_dataset)}")
        print(f"  Num Epochs = {args.epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  max_train_steps = {args.max_train_steps}")
        print(f"  total_steps = {total_steps}")

    progress_bar = tqdm(range(total_steps), disable=not is_local_main_process(smp.local_rank()))
    completed_steps = 0
    total_loss = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            device = torch.device("cuda")
            batch = {k: v.to(device) for k, v, in batch.items()}
            loss = train_step(model,batch)
            
            loss = loss.reduce_mean()
            total_loss += loss.detach().float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1

            # if completed_steps >= args.max_train_steps:
            #     break

        model.eval()
        eval_loss = 0
        losses = []

        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                device = torch.device("cuda")
                batch = {k: v.to(device) for k, v, in batch.items()}
                loss = test_step(model, batch).reduce_mean()
            losses.append(loss)
            
            if step > 50: # TODO: Might wanna comment this part
                break
        
        try:
            eval_loss = torch.mean(torch.tensor(losses,dtype=float))
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        print(f"epoch {epoch}: Eval perplexity: {perplexity} Eval loss: {eval_loss}")

        
    smp.save_checkpoint(args.checkpoint_dir,
                tag=f"flan_t5_weights.pt",
                partial=False,
                model=model,
                optimizer=optimizer)
        
    print("saving the final model")

    wait_for_everyone()

    if is_main_process(smp.rank()):
        tokenizer.save_pretrained(args.checkpoint_dir)


if __name__ == "__main__":
    main()