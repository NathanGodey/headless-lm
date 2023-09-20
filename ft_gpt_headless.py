import os
from engine.data import DataModule
from engine.tasks.pretraining import GptHeadlessPretraining
from engine.lit.lightning_module import TaskTrainer
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import psutil
import argparse
import torch

print("CPU count: ", psutil.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--num_nodes")
parser.add_argument("--global_bs")
parser.add_argument("--gpu_bs")
parser.add_argument("--run_name")
parser.add_argument("--mode")
parser.add_argument("--ft_model")
parser.add_argument("--precision", default="16-mixed")
parser.add_argument('--ckpt_path', nargs='?', const=None, type=str)
args = parser.parse_args()

config = args.config
num_nodes = int(args.num_nodes)
ckpt_path = args.ckpt_path
global_bs = int(args.global_bs)
gpu_bs = int(args.gpu_bs)
run_name = args.run_name
precision = args.precision
mode = args.mode
ft_model = args.ft_model

gpus_by_node = torch.cuda.device_count()

if ((gpus_by_node * num_nodes) % global_bs) == 0:
  raise argparse.ArgumentError(f"Requested a batch size of {global_bs} on {gpu_bs}x{gpus_by_node} GPUs : not a multiple!")
accu_grad_batches = global_bs // (gpus_by_node * num_nodes * gpu_bs)
print(f"Grad. accumulating factor: {accu_grad_batches}")


datamodule = DataModule.from_datasets("/gpfsscratch/rech/awr/uof65ov/datasets/redpajama_test_pythia.hf", train_batch_size=gpu_bs, infer_batch_size=gpu_bs,
split_names=["train(:0.9999)", "train(0.9999:)"], from_disk=True, num_workers=0)

task_trainer = TaskTrainer.load_from_checkpoint(ft_model, map_location="cuda")

tokenizer = task_trainer.task.tokenizer
lm_model = task_trainer.task.lm_model
if mode=="probe":
  lm_model.gpt_neox.requires_grad_(False)

vocab_len, hs = lm_model.gpt_neox.get_input_embeddings().weight.shape

lm_model.embed_out = torch.nn.Linear(hs, vocab_len, bias=False)
lm_model.embed_out.weight.data = lm_model.get_input_embeddings().weight.data.clone()
print(lm_model)


#
task = GptHeadlessPretraining(
    tokenizer, lm_model, config = config
)

trainer = TaskTrainer(task, logger_args={"version": f'{run_name}_{os.environ["SLURM_JOB_ID"]}'}, display_sample_length=20)

checkpoints = [ModelCheckpoint(every_n_train_steps=2500, dirpath=f'/gpfsscratch/rech/rcy/uof65ov/{run_name}_{os.environ["SLURM_JOB_ID"]}', save_top_k=-1), ModelCheckpoint(every_n_train_steps=1000, dirpath=f'/gpfsscratch/rech/rcy/uof65ov/{os.environ["SLURM_JOB_ID"]}', save_top_k=1)]

trainer.fit(
  datamodule,
  num_nodes=num_nodes,
  precision=precision,
  accumulate_grad_batches=accu_grad_batches,
  callbacks=checkpoints,
  limit_val_batches=10,
  val_check_interval=2500,
  gradient_clip_val=1.0,
  benchmark=True,
  default_root_dir=f'/gpfsscratch/rech/rcy/uof65ov/{run_name}_{os.environ["SLURM_JOB_ID"]}',
  ckpt_path=ckpt_path
)

