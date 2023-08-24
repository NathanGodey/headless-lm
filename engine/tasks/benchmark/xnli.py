from engine.tasks.classification import SentenceClassification, SentencePairClassification
from engine.tasks.regression.sts import TextualSimilarity
from engine.data import DataModule
from engine.lit.lightning_module import TaskTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from engine.lit.task_logging import TensorBoardLogger, WandbLogger
import os
import copy

_XNLI_TASK_CONFIG = [
    ("ar", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("de", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("en", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("es", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("fr", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("hi", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    }),
    ("zh", {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 10000
        },
        'datamodule_config': {
            'split_names': ['train', 'validation', 'test']
        },
        'nb_epochs': 10
    })
]



class XnliCrossLingualBenchmark:
    def __init__(self, tokenizer, backbone, train_batch_size=32, infer_batch_size=32,
                 accumulate_grad_batches=4, version=None, logger='tensorboard', logger_args=None,
                 learning_rate=None, train_lang="en"):
        self.tokenizer = tokenizer
        self.backbone = backbone

        if logger == 'tensorboard':
            self.logger = TensorBoardLogger(**logger_args, save_dir=os.getcwd(), version=version, name="xnli_logs", default_hp_metric=False)
        elif logger == 'wandb':
            self.logger = WandbLogger(**logger_args)

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.learning_rate = learning_rate
        self.accumulate_grad_batches = accumulate_grad_batches

        self.train_lang = train_lang

        self.run_metrics = [f'hp/{task_name}_score' for task_name in _XNLI_TASK_CONFIG]

        self.fit()

    def __call__(self):
        train_dl = None
        val_dls = []
        val_dls_mapping = []
        for i, (task_name, task_attr) in enumerate(_XNLI_TASK_CONFIG):
            task_datamodule = DataModule.from_datasets('xnli', task_name,
                                                       train_batch_size=self.train_batch_size,
                                                       infer_batch_size=self.infer_batch_size,
                                                       **task_attr.get('datamodule_config', {})
                                                       )
            task_datamodule.setup()
            if task_name == self.train_lang:
                train_dl = task_datamodule.train_dataloader()
                train_task_name, train_task_attr = task_name, task_attr
            val_dls.append(task_datamodule.val_dataloader())
            val_dls_mapping.append(task_name)
        task = train_task_attr['task_class'](self.tokenizer,
                                        copy.deepcopy(self.backbone),
                                        config=train_task_attr.get('task_config'))
        
        if self.learning_rate is not None:
            task.backbone_lr = self.learning_rate
            task.head_lr = self.learning_rate

        

        trainer = TaskTrainer(task, log_prefix=train_task_name, run_metrics=self.run_metrics if i == 0 else None,
                                logger=self.logger)

        task_datamodule = {
            "train_dataloaders": train_dl,
            "val_dataloaders": val_dls,
        }
        trainer.fit(task_datamodule, gpus=1,
                    accumulate_grad_batches=self.accumulate_grad_batches,
                    max_epochs=train_task_attr.get('nb_epochs', 10))

    def fit(self):
        self()

class XnliBenchmark:
    def __init__(self, tokenizer, backbone, train_batch_size=32, infer_batch_size=32,
                 accumulate_grad_batches=4, version=None, logger='tensorboard', logger_args=None,
                 learning_rate=None):
        self.tokenizer = tokenizer
        self.backbone = backbone

        if logger == 'tensorboard':
            self.logger = TensorBoardLogger(**logger_args, save_dir=os.getcwd(), version=version, name="xnli_logs", default_hp_metric=False)
        elif logger == 'wandb':
            self.logger = WandbLogger(**logger_args)

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.learning_rate = learning_rate
        self.accumulate_grad_batches = accumulate_grad_batches

        self.run_metrics = [f'hp/{task_name}_score' for task_name in _XNLI_TASK_CONFIG]

        self.fit()

    def __call__(self):
        for i, (task_name, task_attr) in enumerate(_XNLI_TASK_CONFIG):
            task = task_attr['task_class'](self.tokenizer,
                                           copy.deepcopy(self.backbone),
                                           config=task_attr.get('task_config'))
            
            if self.learning_rate is not None:
                task.backbone_lr = self.learning_rate
                task.head_lr = self.learning_rate

            task_datamodule = DataModule.from_datasets('xnli', task_name,
                                                       train_batch_size=self.train_batch_size,
                                                       infer_batch_size=self.infer_batch_size,
                                                       **task_attr.get('datamodule_config', {})
                                                       )

            trainer = TaskTrainer(task, log_prefix=task_name, run_metrics=self.run_metrics if i == 0 else None,
                                  logger=self.logger)

            task_metric = task_attr.get('metric', 'accuracy')
            checkpoint_callback = ModelCheckpoint(monitor=f'{task_name}/val/{task_metric}', mode='max',
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
