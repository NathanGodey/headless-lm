from engine.tasks.classification import SentenceClassification, SentencePairClassification
from engine.tasks.regression.sts import TextualSimilarity
from engine.data import DataModule
from engine.lit.lightning_module import TaskTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from engine.lit.task_logging import TensorBoardLogger, WandbLogger
import os
import copy

_ANLI_TASK_CONFIG = [
    ("plain_text", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 2500
        },
        'datamodule_config': {
            'split_names': ['train_r1', 'dev_r1', 'test_r1']
        },
        'nb_epochs': 200
    }),
    ("plain_text", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 100
        },
        'datamodule_config': {
            'split_names': ['train_r2', 'dev_r2', 'test_r2']
        },
        'nb_epochs': 50
    }),
    ("plain_text", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 50
        },
        'datamodule_config': {
            'split_names': ['train_r3', 'dev_r3', 'test_r3']
        },
        'nb_epochs': 20
    })
]


class AnliBenchmark:
    def __init__(self, tokenizer, backbone, train_batch_size=32, infer_batch_size=32,
                 accumulate_grad_batches=4, version=None, logger='tensorboard', logger_args=None):
        self.tokenizer = tokenizer
        self.backbone = backbone

        if logger == 'tensorboard':
            self.logger = TensorBoardLogger(**logger_args, save_dir=os.getcwd(), version=version, name="anli_logs", default_hp_metric=False)
        elif logger == 'wandb':
            self.logger = WandbLogger(**logger_args)

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

        self.run_metrics = [f'hp/{task_name}_score' for task_name in _ANLI_TASK_CONFIG]

        self.fit()

    def __call__(self):
        for i, (task_name, task_attr) in enumerate(_ANLI_TASK_CONFIG):
            task = task_attr['task_class'](self.tokenizer,
                                           copy.deepcopy(self.backbone),
                                           config=task_attr.get('task_config'))

            task_datamodule = DataModule.from_datasets('anli', task_name,
                                                       train_batch_size=self.train_batch_size,
                                                       infer_batch_size=self.infer_batch_size,
                                                       **task_attr.get('datamodule_config', {})
                                                       )

            trainer = TaskTrainer(task, log_prefix=f"round_{i}", run_metrics=self.run_metrics if i == 0 else None,
                                  logger=self.logger)

            task_metric = task_attr.get('metric', 'accuracy')
            checkpoint_callback = ModelCheckpoint(monitor=f'round_{i}/val/{task_metric}', mode='max',
                                                  save_top_k=0)
            trainer.fit(task_datamodule, gpus=1,
                        accumulate_grad_batches=self.accumulate_grad_batches,
                        callbacks=[checkpoint_callback],
                        max_epochs=task_attr.get('nb_epochs', 10))
            best_score = checkpoint_callback.best_model_score
            if best_score:
                self.logger.log_metrics({f'hp/{task_name}_score': best_score.item()})

    def fit(self):
        self()
