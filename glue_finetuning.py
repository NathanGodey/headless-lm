from engine.tasks.benchmark.glue import GlueBenchmark
from transformers import AutoTokenizer, AutoModel

model_id = 'nthngdy/headless-bert-bs64-owt2'

tokenizer = AutoTokenizer.from_pretrained(model_id)
mlm_model = AutoModel.from_pretrained(model_id)

backbone = mlm_model

GlueBenchmark(
    tokenizer, backbone, logger='wandb', logger_args={'project': 'GLUE'}, train_batch_size=32, accumulate_grad_batches=1,
    learning_rate=1e-5, weighted_ce=True, weight_decay=0.01, shuffle=True
)
