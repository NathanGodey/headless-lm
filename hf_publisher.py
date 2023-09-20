from engine.lit.lightning_module import TaskTrainer
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--hf_name")
parser.add_argument("--model_ckpt")
parser.add_argument("--mode")
args = parser.parse_args()
hf_name = args.hf_name
model_ckpt = args.model_ckpt
mode = args.mode


task_trainer = TaskTrainer.load_from_checkpoint(model_ckpt, map_location="cpu")

tokenizer = task_trainer.task.tokenizer

if mode == "mlm":
    model = task_trainer.task.mlm_model
else: 
    model = task_trainer.task.lm_model

if mode == "add_head":
    vocab_len, hs = model.gpt_neox.get_input_embeddings().weight.shape

    model.embed_out = nn.Linear(hs, vocab_len, bias=False)
    model.embed_out.weight.data = model.get_input_embeddings().weight.data.clone()

model.push_to_hub(hf_name, private=True)
tokenizer.push_to_hub(hf_name, private=True)