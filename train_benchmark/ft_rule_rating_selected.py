import sys
sys.path.append("/DataSelection/")

from modules.module_generation import *
from modules.utils import *
import argparse
import pandas as pd
import json
import os
from benchmark.evaluate_model import evaluate_model
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import concatenate_datasets, load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # Import tqdm for the progress bar


if __name__ == "__main__":
    print("free memory at beginning...")
    free_memory()

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--training_output_dir', type=str, default='GEN_sft_output', help="Directory for training outputs.")
    parser.add_argument('-g', '--training_logging_dir', type=str, default='GEN_sft_log', help="Directory for training logs.")
    parser.add_argument('-r', '--learning_rate', type=float, default=2e-5, help="Learning rate.")
    parser.add_argument('-H', '--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('-t', '--finetune_per_device_train_batch_size', type=int, default=1, help="Batch size per device for finetune training.")
    parser.add_argument('-W', '--warmup_steps', type=int, default=40, help="Number of warmup steps.")
    parser.add_argument('-L', '--logging_steps', type=int, default=100, help="Number of logging steps.")
    parser.add_argument('-S', '--save_steps', type=int, default=500000, help="Number of steps between saving checkpoints.")

    parser.add_argument('-R', '--lora_r', type=int, default=64, help="LoRA rank.")
    parser.add_argument('-m', '--model_path', type=str, default='meta-llama/Meta-Llama-3-8B', help="Model name.")
    # Other models to consider: "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-Instruct",  "meta-llama/Meta-Llama-3-8B", "princeton-nlp/Sheared-LLaMA-1.3B"
    parser.add_argument('-f', '--folder_path', type=str, default='ft_all50rules/', help="Dataset name.")

    parser.add_argument('-d', '--dataset_path', type=str, default='selected_data/SlimPajama6B_short_1M_1percent.jsonl', help="Dataset name.")
    parser.add_argument('-s', '--save_model_path', type=str, default='finetuned_models/finetuned_Pythia1B_1percent', help="Path to save the model.")
    parser.add_argument('-l', '--tokenizer_max_length', type=int, default=4096, help="Maximum length of the tokenizer.")
    parser.add_argument('--dtype_str', type=str, default='bfloat16', help="Data type for model training.")
    parser.add_argument('-w', '--wandb_proj_name', type=str, default='ft_50Rules', help="Weights and Biases project name.")

    args = parser.parse_args()
    training_output_dir = args.training_output_dir
    training_logging_dir = args.training_logging_dir
    learning_rate = args.learning_rate
    num_train_epochs = args.num_train_epochs
    finetune_per_device_train_batch_size = args.finetune_per_device_train_batch_size
    warmup_steps = args.warmup_steps
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    lora_r = args.lora_r
    model_path = args.model_path

    folder_path = args.folder_path
    dataset_path = args.dataset_path
    save_model_path = args.save_model_path
    tokenizer_max_length = args.tokenizer_max_length
    dtype_str = args.dtype_str
    wandb_proj_name = args.wandb_proj_name

    # training arguments:
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=finetune_per_device_train_batch_size,
        warmup_steps=warmup_steps, 
        logging_dir=training_logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps, 
        report_to=["wandb"],
    )

    folder_path = os.path.join("/DataSelection/experiments_Math/", folder_path)
    dataset_path = os.path.join(folder_path, dataset_path)
    save_model_path = os.path.join(folder_path, save_model_path)
    benchmark_results_path = f"benchmark_results_{os.path.basename(save_model_path)}.json"
    benchmark_results_path = os.path.join(folder_path, benchmark_results_path)


    print("\n\n==============Init LLMTextClassification:==================\n")
    print(f"PARAMS: model_path={model_path}, tokenizer_max_length={tokenizer_max_length}, dtype_str={dtype_str}")
    generator = LLMTextGeneration(model_path, tokenizer_max_length, dtype_str)



    print("\n\n=================Data:===================\n")
    def read_jsonl_file_texts(file_path):
        """Read a JSONL file and return a list of texts."""
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                texts.append(json.loads(line)['text'])
        return texts

    texts = read_jsonl_file_texts(dataset_path)  # Returns a list of dictionaries
    train_dataset = Dataset.from_dict({"text": texts})
    data_info(generator.tokenizer, train_dataset)


    print("\n\n==============SlimPajama Finetune & SaveModel:==================\n")
    free_memory()
    generator.finetune(train_dataset, None, lora_r, training_args, wandb_proj_name, save_model_path)



