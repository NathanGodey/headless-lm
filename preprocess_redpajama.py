from datasets import load_dataset, logging
from transformers import AutoTokenizer, logging as T_logging
import torch
import numpy as np
import psutil
import functools
import operator

# logging.set_verbosity(logging.CRITICAL)

# T_logging.set_verbosity(T_logging.CRITICAL)

NUM_CPU = psutil.cpu_count()
print(f"Using {NUM_CPU} CPUs...")

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')

def tokenize_and_pack(batch, max_seq_len=2048):
    tokenized_batch = tokenizer(batch["text"]).input_ids
    tokenized_batch_flat = functools.reduce(operator.iconcat, tokenized_batch, [])
    packed_batch = np.reshape(tokenized_batch_flat[:-(len(tokenized_batch_flat)%max_seq_len)], (-1, max_seq_len))
    return packed_batch.tolist()

print("Loading dataset...")
ds = load_dataset("togethercomputer/RedPajama-Data-1T", "default")

print("Packing dataset...")
ds = ds.map(lambda x: {"packed":tokenize_and_pack(x)}, remove_columns=["meta", "text"], batched=True, batch_size=100000, num_proc=NUM_CPU)

ds = ds.shuffle()

print("Saving dataset...")
ds.save_to_disk("/gpfsscratch/rech/awr/uof65ov/datasets/redpajama_pythia.hf", num_proc=NUM_CPU)