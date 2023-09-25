from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np
import psutil
import functools
import operator
import argparse
import torch
import json


parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()

config = json.load(open(args.config, 'rb'))
ds_name = config["dataset_name"]
ds_config = config["dataset_config"]
hf_tokenizer = config["hf_tokenizer"]
max_seq_len = config["max_seq_len"]
output = config["output"]

NUM_CPU = psutil.cpu_count()
print(f"Using {NUM_CPU} CPUs...")

tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)

def tokenize_and_pack(batch, max_seq_len=max_seq_len):
    tokenized_batch = tokenizer(batch["text"]).input_ids
    tokenized_batch_flat = functools.reduce(operator.iconcat, tokenized_batch, [])
    packed_batch = np.reshape(tokenized_batch_flat[:-(len(tokenized_batch_flat)%max_seq_len)], (-1, max_seq_len))
    return packed_batch.tolist()

print("Loading dataset...")
ds = load_dataset(ds_name, ds_config)

print("Packing dataset...")
ds = ds.map(lambda x: {"packed":tokenize_and_pack(x)}, remove_columns=ds['train'].column_names, batched=True, batch_size=100000, num_proc=NUM_CPU)

ds = ds.shuffle()

print("Saving dataset...")
ds.save_to_disk(output, num_proc=NUM_CPU)
