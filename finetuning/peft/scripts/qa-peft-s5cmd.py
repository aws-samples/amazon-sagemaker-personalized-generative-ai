import os
import subprocess
import sys
import json
import argparse
import logging
import numpy as np
from datasets import load_from_disk, load_metric
from transformers import ( AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import torch
import gc

def report_gpu():
    print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()

def training_function(args):
    print(f'Args: {args}')
    
    # is needed for Amazon SageMaker Training Compiler
    os.environ["GPU_NUM_DEVICES"] = args.n_gpus
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    logger.info(f"loaded train_dataset \n{train_dataset}\n")
    
    # huggingface hub model id
    model_id = "philschmid/flan-t5-xxl-sharded-fp16"
    
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    
    # Define LoRA Config
    lora_config = LoraConfig(
         r=16,
         lora_alpha=32,
         target_modules=["q", "v"],
         lora_dropout=0.05,
         bias="none",
         task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    # logger.info(f'Printing trainable parameters \n{model.print_trainable_parameters()}\n')
    
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    logger.info('Tokenizer defined!')
    
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # define training args
    output_dir = '/tmp/output/checkpoints'
    training_args = Seq2SeqTrainingArguments(
        # auto_find_batch_size = True,
        output_dir           = output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing = True,
        learning_rate        = args.learning_rate, # higher learning rate
        num_train_epochs     = args.epochs,
        logging_dir          = f"{args.output_data_dir}/logs",
        logging_strategy     = "steps",
        logging_steps        = 500,
        save_strategy        = "no"
    )
    
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model         = model,
        args          = training_args,
        data_collator = data_collator,
        train_dataset = train_dataset,
    )
    model.config.use_cache = False
    
    # train model
    trainer.train()
    logger.info('Model training completed!')
    
    report_gpu()
    
    save_model_dir = '/tmp/output/asset'
    tokenizer.save_pretrained(save_model_dir)
    trainer.save_model(save_model_dir)
    print('Trainer completed saving the model')
    
    save_model_dir = save_model_dir + '/'
    final_asset_folder_path = args.save_model_s3_path + 'asset/'
    os.system("./scripts/s5cmd sync --delete {0} {1}".format(save_model_dir, final_asset_folder_path))
    print('Completed saving the model with s5cmd')
    
def launch_func(command):
    try:
        subprocess.run(command, shell=False)
    except Exception as e:
        logger.info(e)
    return

def parse_arge():
    parser = argparse.ArgumentParser()

    # # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--save_model_s3_path", type=str, default=None, help="s3 path to save model")

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args = parser.parse_known_args()
    return args # NOTE: This is a tuple

def main():
    args, _ = parse_arge()
    s5cmd_command = "chmod +x ./scripts/s5cmd"
    print(f"s5cmd_command = {s5cmd_command}")
    launch_func(s5cmd_command)
    training_function(args)
    
if __name__ == "__main__":
    main()