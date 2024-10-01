import sys
sys.path.append('/code')
import os
import argparse
import logging
from datasets import Dataset
import time
import json
import sys
from datasets import load_dataset
import numpy as np
import pandas as pd

from modules.module_generation import *
from rating_prompts import *

import torch
def report_cuda_memory():
    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Memory allocated: {memory_allocated:.2f} GB")
    print(f"Memory reserved: {memory_reserved:.2f} GB")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process specific rule index.')
    parser.add_argument('rule_idx_cmd', type=int, help='The index of the rule to process')
    args = parser.parse_args()
    rule_idx_cmd = args.rule_idx_cmd
    free_memory()

    print("\n\n==============Init LLMTextGeneration:==================\n")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_max_length=1024
    max_new_tokens = 4
    dtype_str='bfloat16'
    print(f"PARAMS: model_name={model_name}, tokenizer_max_length={tokenizer_max_length}, max_new_tokens={max_new_tokens}, dtype_str={dtype_str}")
    generator = LLMTextGeneration(model_name, tokenizer_max_length, max_new_tokens, dtype_str)

    print("\n\n==============Data:==================\n")
    start_time = time.time()
    # Load the data and keep only the "text" field
    texts = []
    print("Loading data...")
    with open("/SlimPajama_data/SlimPajama6B_short_1M.jsonl", 'r') as jsonl_file:
        for idx, line in enumerate(jsonl_file):
            data = json.loads(line)
            texts.append(data["text"])
            if idx % 1000000 == 0: print(f"{idx}", end=",")
    total_size = len(texts)
    print(f"\nDone! Loaded SlimPajama {total_size} texts from JSONL file in {(time.time() - start_time):.2f} seconds.")
     # Create a Hugging Face dataset
    train_dataset = Dataset.from_dict({"text": texts})
    print(f"Created Hugging Face dataset with {len(train_dataset)} examples.")


    print("\n\n==============Fetch Rules:==================\n")
    all_rules = rules #imported from rating_prompts.py
    print(f"len(all_rules)={len(all_rules)}")

    def rule_based_rating(rule, batch_data):
        size = len(batch_data)
        questions = []
        for i in range(size):
            prompt_i = rule_rating_prompt(rule, batch_data[i])
            questions.append(prompt_i)

        start_time = time.time()
        print(f"Done preparing prompts for this batch. len(questions) = {len(questions)}"); sys.stdout.flush();
        free_memory()
        responses = generator.inference(questions)
        print(f"Done! inference {len(questions)} questions took {time.time() - start_time} seconds.")
        return responses

    print("\n\n==============Rule-based rating:==================\n")
    batch_size = 1000  # Adjust batch size based on memory constraints

    # Iterate over each rule
    rule_idx = rule_idx_cmd
    rule = all_rules[rule_idx]
    
    # for rule_idx, rule in enumerate(all_rules):
    print(f"Rule{rule_idx}:{rule}")
    rule_folder = f'rule_rating_results/rule{rule_idx}'
    
    # Create folder for each rule if it doesn't exist
    if not os.path.exists(rule_folder):
        os.makedirs(rule_folder)

    # Process the dataset in batches
    for start_idx in range(0, total_size, batch_size):
        free_memory()
        end_idx = min(start_idx + batch_size, total_size)
        batch_data = train_dataset[start_idx:end_idx]["text"]
        batch_number = start_idx // batch_size
        batch_save_path = os.path.join(rule_folder, f'batch{batch_number}.json')
        
        # Skip the batch if it has already been processed
        if os.path.exists(batch_save_path):
            print(f"Batch {batch_number} for rule {rule_idx} already exists, skipping.")
            continue
    
        print(f"\nProcessing batch {batch_number} for rule {rule_idx} with size {len(batch_data)}"); sys.stdout.flush()
        
        # Compute ratings for the current batch
        quality_responses = rule_based_rating(rule, batch_data)
        
        # Save batch results
        with open(batch_save_path, 'w') as json_file:
            json.dump(quality_responses, json_file, indent=4)
        print(f"Batch {batch_number} data saved to {batch_save_path}"); sys.stdout.flush()

    print("All batches processed.")


