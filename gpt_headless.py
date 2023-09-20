import os
from engine.data import DataModule
from engine.tasks.pretraining import GptHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import psutil
import argparse
import torch
import time

print("CPU count: ", psutil.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--num_nodes")
parser.add_argument("--global_bs")
parser.add_argument("--gpu_bs")
parser.add_argument("--dataset")
parser.add_argument("--hf_tokenizer")

parser.add_argument("--run_name")
parser.add_argument("--hf_path")
parser.add_argument("--accelerator", default="hf")
parser.add_argument("--precision", default="16-mixed")
parser.add_argument('--ckpt_path', nargs='?', const=None, type=str)

parser.add_argument('--saved_ckpt_path')
parser.add_argument("--ckpt_every", default=10000)

args = parser.parse_args()

config = args.config
num_nodes = int(args.num_nodes)
ckpt_path = args.ckpt_path
global_bs = int(args.global_bs)
gpu_bs = int(args.gpu_bs)
dataset = args.dataset
hf_tokenizer = args.hf_tokenizer

model_max_seq_len = args.config.pop("model_max_seq_len", 2048)

run_name = args.run_name
hf_path = args.hf_path
accelerator = args.accelerator
precision = args.precision
ckpt_every = args.ckpt_every

saved_ckpt_path = args.saved_ckpt_path

if accelerator == "xformers":
  from engine.models.xformers.efficient_gpt_neox import GPTNeoXForCausalLM
elif accelerator == "flash_attention":
  from engine.models.flash_attention.efficient_gpt_neox import GPTNeoXForCausalLM
elif accelerator == "hf":
  from transformers import GPTNeoXForCausalLM
else:
    raise NotImplementedError(f"Unknown accelerator {accelerator}. Please pick between 'hf', 'flash_attention', 'xformers'.")

if "A100" in torch.cuda.get_device_name():
  torch.set_float32_matmul_precision('high')

gpus_by_node = torch.cuda.device_count()

if ((gpus_by_node * num_nodes) % global_bs) == 0:
  raise argparse.ArgumentError(f"Requested a batch size of {global_bs} on {gpu_bs}x{gpus_by_node} GPUs : not a multiple!")
accu_grad_batches = global_bs // (gpus_by_node * num_nodes * gpu_bs)
print(f"GPU BS: {gpu_bs}; Grad. accumulating factor: {accu_grad_batches}")


datamodule = DataModule.from_datasets(dataset, train_batch_size=gpu_bs, infer_batch_size=gpu_bs,
split_names=["train(:0.9999)", "train(0.9999:)"], from_disk=True, num_workers=0)

tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer)
lm_config = AutoConfig.from_pretrained(hf_path)

lm_config.max_position_embeddings = model_max_seq_len
lm_model = GPTNeoXForCausalLM(lm_config)
print(lm_model)


task = GptHeadlessPretraining(
    tokenizer, lm_model, config = config
)

version_name = run_name
trainer = TaskTrainer(task, logger_args={"version": version_name})

checkpoints = [
  ModelCheckpoint(every_n_train_steps=ckpt_every, dirpath=f'{saved_ckpt_path}/{version_name}', save_top_k=-1),
  ModelCheckpoint(every_n_train_steps=1000, dirpath=f'{saved_ckpt_path}/{version_name}', save_top_k=1)
]

trainer.fit(
  datamodule,
  num_nodes=num_nodes,
  precision=precision,
  accumulate_grad_batches=accu_grad_batches,
  callbacks=checkpoints,
  limit_val_batches=10,
  val_check_interval=0.1,
  gradient_clip_val=1.0,
  benchmark=True,
  default_root_dir=f'{saved_ckpt_path}/{version_name}',
  ckpt_path=ckpt_path
)
