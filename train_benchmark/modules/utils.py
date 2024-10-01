import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score
import numpy as np
import gc
import wandb
import os
import sys
import time


hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY")
# MODEL_PATH_PREFIX = os.getenv("MODEL_HUGGINGFACE")

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

def model_memory(model, dtype_str):
    dtype_bytes_map = {'float16': 2, 'float32': 4, 'bfloat16': 2}
    total_memory_bytes = sum(p.numel() for p in model.parameters()) * dtype_bytes_map[dtype_str]  # XM: 2 bytes for float16, 4 bytes for float32, etc
    total_memory_gb = total_memory_bytes / (1024 ** 3)  # Convert to gugabytes
    print(f"Approximate memory usage (GB): {total_memory_gb:.2f} G")


def get_torch_dtype(dtype_str):
    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid data type specified: {dtype_str}. Must be 'float16', 'float32', or 'bfloat16'.")
    return dtype_map[dtype_str]


def classification_loss(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    return accuracy, precision, recall, f1, mcc

# def query(model, tokenizer, questions, dtype=torch.float16):
#     pipeline = transformers.pipeline(
#         "text-classification",
#         model=model,
#         tokenizer=tokenizer,
#         torch_dtype=dtype,
#         device_map="auto",
#     )
#     return pipeline(questions)

class CustomDataset_Train_Cls(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, tokenizer_max_length=128):
        # self.tokenizer = AutoTokenizer.from_pretrained("bigscience/llama2-7B")
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.tokenizer_max_length = tokenizer_max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class CustomDataset_Inference(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, tokenizer_max_length=128):
        self.encodings = tokenizer(texts, padding='max_length', max_length=tokenizer_max_length, truncation=True)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])


def json_to_str(example):
    s = ""
    for j, pair in enumerate(example['messages']):
        if pair['role']=='system':
            s += ('Instruction: ' + pair['content'])
        elif pair['role']=='user':
            s += ('Prompt: ' + pair['content'])
        elif pair['role']=='assistant':
            s += ('Response: ' + pair['content'])
        if j != len(example['messages'])-1:
            s += " "
    return s

# Expects it has 'text' column for max token count calcluation
def data_info(tokenizer, train_dataset, valid_dataset=None, test_dataset=None):
    print("Num of examples in train_dataset:", train_dataset.num_rows)
    if valid_dataset:
        print("Num of examples in val_dataset:", valid_dataset.num_rows)
    if test_dataset:
        print("Num of examples in test_dataset:", test_dataset.num_rows)
    print("Features/columns:", train_dataset.column_names)
    count = min(200, train_dataset.num_rows)
    token_lengths = [len(tokenizer.encode(text)) for text in train_dataset['text'][:count]]
    print(f"Token Length (first {count} examples from train_dataset): Max = {np.max(token_lengths)}. Average = {np.mean(token_lengths)}. Median = {np.median(token_lengths)}. 90thPercentile = {np.percentile(token_lengths, 90)}")

def add_path_prefix(path_name):
    MODEL_PATH_PREFIX = "/n/holyscratch01/lu_lab/Users/xiaominli/models/huggingface"
    return os.path.join(MODEL_PATH_PREFIX, path_name) if len(MODEL_PATH_PREFIX) > 0 else path_name
