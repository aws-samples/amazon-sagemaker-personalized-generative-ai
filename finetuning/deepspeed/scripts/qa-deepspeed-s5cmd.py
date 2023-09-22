import os
import argparse
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
import torch
import evaluate
import nltk
from nltk.tokenize import sent_tokenize

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# nltk.download("punkt", quiet=True)

# Metric
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation Steps")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing.")
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument("--save_model_s3_path", type=str, default=None, help="s3 path to save model")
    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(args.training_dir)
    eval_dataset = load_from_disk(args.test_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        cache_dir = "/opt/ml/input/" # changed for SM to have enough storage space
    )

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
    )

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    output_dir = '/tmp/output/checkpoints'

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        predict_with_generate       = True,
        generation_max_length       = args.generation_max_length,
        generation_num_beams        = args.generation_num_beams,
        fp16                        = False, # Overflows with fp16
        bf16                        = args.bf16,
        learning_rate               = args.learning_rate,
        num_train_epochs            = args.epochs,
        deepspeed                   = args.deepspeed,
        gradient_checkpointing      = args.gradient_checkpointing,
        
        # logging & evaluation strategies
        logging_dir                 = f"{args.output_data_dir}/logs",
        logging_strategy            = "steps",
        logging_steps               = 500,
        evaluation_strategy         = "epoch", # Default - steps
        save_strategy               = "epoch", # Default - steps
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        # metric_for_best_model      = "overall_f1", # default is loss (eval)
    )
    
    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics, #TODO: Commenting to check if this why evaluation is slow
    )

    # Start training
    trainer.train()
    
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('Best Model Checkpoint:', best_ckpt_path)
    
    print("Starting: Save best model artifacts")
    save_model_dir = '/tmp/output/asset'
    tokenizer.save_pretrained(save_model_dir)
    trainer.save_model(save_model_dir)
    print("Ending: Save best model artifacts")
    
    WORLD_RANK = int(os.environ['RANK'])
    if WORLD_RANK == 0:
        output_dir = output_dir + '/'
        checkpoints_folder_path = args.save_model_s3_path + 'checkpoints/'
        os.system("./scripts/s5cmd sync --delete {0} {1}".format(output_dir, checkpoints_folder_path))
        save_model_dir = save_model_dir + '/'
        final_asset_folder_path = args.save_model_s3_path + 'asset/'
        os.system("./scripts/s5cmd sync --delete {0} {1}".format(save_model_dir, final_asset_folder_path))


if __name__ == "__main__":
    # main()
    args, _ = parse_arge()
    if int(os.environ['LOCAL_RANK']) == 0:
        nltk.download("punkt", quiet=True)
    training_function(args)
