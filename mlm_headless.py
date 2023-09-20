import os
from engine.data import DataModule
from engine.tasks.pretraining import MlmHeadlessPretraining
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
parser.add_argument("--hf_path")
parser.add_argument("--accelerator", default="hf")
parser.add_argument("--precision", default="16-mixed")
parser.add_argument('--ckpt_path', nargs='?', const=None, type=str)
args = parser.parse_args()

config = args.config
num_nodes = int(args.num_nodes)
ckpt_path = args.ckpt_path
global_bs = int(args.global_bs)
gpu_bs = int(args.gpu_bs)
run_name = args.run_name
hf_path = args.hf_path
accelerator = args.accelerator
precision = args.precision

if accelerator == "xformers":
  from engine.models.xformers.efficient_bert import BertForMaskedLM
elif accelerator == "flash_attention":
  from engine.models.flash_attention.efficient_bert import BertForMaskedLM
  torch.set_float32_matmul_precision('medium')
elif accelerator == "hf":
  from transformers import BertForMaskedLM
else:
    raise NotImplementedError(f"Unknown accelerator {accelerator}. Please pick between 'hf', 'flash_attention', 'xformers'.")

gpus_by_node = torch.cuda.device_count()

if ((gpus_by_node * num_nodes) % global_bs) == 0:
  raise argparse.ArgumentError(f"Requested a batch size of {global_bs} on {gpu_bs}x{gpus_by_node} GPUs : not a multiple!")
accu_grad_batches = global_bs // (gpus_by_node * num_nodes * gpu_bs)
print(f"Grad. accumulating factor: {accu_grad_batches}")


datamodule = DataModule.from_datasets("/gpfsscratch/rech/awr/uof65ov/datasets/redpajama_test_pythia.hf", train_batch_size=gpu_bs, infer_batch_size=gpu_bs,
split_names=["train(:0.9999)", "train(0.9999:)"], from_disk=True, num_workers=0)


tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
lm_config = AutoConfig.from_pretrained(hf_path)

torch.set_float32_matmul_precision('medium')

lm_config.vocab_size = len(tokenizer.vocab)
tokenizer.mask_token_id = 1
lm_config.max_position_embeddings = 512
lm_model = BertForMaskedLM(lm_config)


task = MlmHeadlessPretraining(
    tokenizer, lm_model, config = config
)

version_name = run_name
trainer = TaskTrainer(task, logger_args={"version": f'{version_name}_{os.environ["SLURM_JOB_ID"]}'}, display_sample_length=20)

checkpoints = [ModelCheckpoint(every_n_train_steps=10000, dirpath=f'/gpfsscratch/rech/rcy/uof65ov/{version_name}_{os.environ["SLURM_JOB_ID"]}', save_top_k=-1), ModelCheckpoint(every_n_train_steps=1000, dirpath=f'/gpfsscratch/rech/rcy/uof65ov/{os.environ["SLURM_JOB_ID"]}', save_top_k=1)]

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
  default_root_dir=f'/gpfsscratch/rech/rcy/uof65ov/{os.environ["SLURM_JOB_ID"]}',
  ckpt_path=ckpt_path,
)
