import os
import argparse
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)
import numpy as np
import torch
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
from datasets import load_from_disk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

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

    
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
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
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 2,
        load_best_model_at_end      = True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
    )

    # Start training
    trainer.train()
    
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])
    tokenizer.save_pretrained(os.environ["SM_MODEL_DIR"])

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
