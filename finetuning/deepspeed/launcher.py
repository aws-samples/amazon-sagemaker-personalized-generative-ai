import sys
import os
import subprocess
import json
import sys
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(
        description=("SageMaker DeepSpeed Launch helper utility that will spawn deepspeed training scripts")
    )
    parser.add_argument(
        "--training_script",
        type=str,
        help="Path to the training program/script to be run in parallel, can be either absolute or relative",
    )

    # rest from the training program
    parsed, nargs = parser.parse_known_args()

    return parsed.training_script, nargs


def main():
    num_gpus     = int(os.environ.get("SM_NUM_GPUS", 0))
    hosts        = json.loads(os.environ.get("SM_HOSTS", "{}"))
    num_nodes    = len(hosts)
    current_host = os.environ.get("SM_CURRENT_HOST", 0)
    rank         = hosts.index(current_host)
    
    print(f"num_gpus = {num_gpus}, num_nodes = {num_nodes}, current_host = {current_host}, rank = {rank}")
    
    s5cmd_command = "chmod +x ./scripts/s5cmd"
    print(f"s5cmd_command = {s5cmd_command}")
    launch_func(s5cmd_command)
    
    train_script, args = parse_args()
    deepspeed_command = f"deepspeed --num_gpus={num_gpus} {train_script} {' '.join(args)}"
    print(f"deepspeed_command = {deepspeed_command}")
    launch_func(deepspeed_command)

def launch_func(command):
    try:
        subprocess.run(command, shell=False)
    except Exception as e:
        logger.info(e)

if __name__ == "__main__":
    main()