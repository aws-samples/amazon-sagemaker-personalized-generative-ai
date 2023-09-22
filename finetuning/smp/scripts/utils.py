import argparse
from contextlib import contextmanager
import torch
import os
import json
from transformers import (
    MODEL_MAPPING,
    SchedulerType
)

# Distributed training helper methods.
def wait_for_everyone():
    #deepspeed.comm.barrier
    torch.distributed.barrier()


def is_main_process(rank):
    if rank == 0:
        return True
    else:
        return False

def _goes_first(is_main):
    if not is_main:
        wait_for_everyone()
    yield
    if is_main:
        wait_for_everyone()

@contextmanager
def main_process_first(rank):
    """
    Lets the main process go first inside a with block.
    The other processes will enter the with block after the main process exits.
    """
    yield from _goes_first(is_main_process(rank))

def is_local_main_process(local_rank):
    return local_rank == 0

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# args parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a FLAN T5 model on a Seq2Seq task")

    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    
     # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for reproducible training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument("--save_model_s3_path", type=str, default=None, help="s3 path to save model")
    parser.add_argument("--optim", type=str, default='adamw_hf', help="The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--group_texts",default=False,help="Whether to group texts together when tokenizing")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument("checkpoint_dir",type=str,default="/opt/ml/checkpoints")
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args,_ = parser.parse_known_args()

    return args