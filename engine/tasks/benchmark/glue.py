from engine.tasks.classification import SentenceClassification, SentencePairClassification
from engine.tasks.regression.sts import TextualSimilarity
from engine.data import DataModule
from engine.lit.lightning_module import TaskTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from engine.lit.task_logging import TensorBoardLogger, WandbLogger
import os
import copy

_GLUE_TASK_CONFIG = {
    # "wnli": {
    #     'task_class': SentencePairClassification,
    #     'task_config': {
    #         'num_class': 2,
    #         'warmup_steps': 25
    #     }
    # },
    "rte": {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 160
        },
        'nb_epochs': 20
    },
    "sst2": {
        'task_class': SentenceClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 4200
        },
        'nb_epochs': 20,
    },
    "cola": {
        'task_class': SentenceClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 530
        },
        'metric': 'mcc',
        'nb_epochs': 20,
    },
    "stsb": {
        'task_class': TextualSimilarity,
        'task_config': {
            'warmup_steps': 360
        },
        'metric': 'spearman',
        'nb_epochs': 20,
    },
    "mrpc": {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 230
        },
        'nb_epochs': 20,
    },
    "qqp": {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 3400,
            'text_keys': ['question1', 'question2']
        },
    },
    "qnli": {
        'task_class': SentencePairClassification,
        'task_config': {
            'num_class': 2,
            'warmup_steps': 980,
            'text_keys': ['question', 'sentence']
        },
        'nb_epochs': 10,
    },
    "mnli": {
        'task_class': SentencePairClassification,
        'datamodule_config': {
            'split_names': ['train', 'validation_matched+validation_mismatched', 'test_matched+test_mismatched']
        },
        'task_config': {
            'num_class': 3,
            'text_keys': ['premise', 'hypothesis'],
            'warmup_steps': 80
        },
        'nb_epochs': 10,
    },
}


class GlueBenchmark:
    def __init__(self, tokenizer, backbone, train_batch_size=8, infer_batch_size=32,
                 accumulate_grad_batches=4, version=None, logger='tensorboard', logger_args=None,
                 learning_rate=None):
        self.tokenizer = tokenizer
        self.backbone = backbone
        logger_args = logger_args if logger_args else {}

        if logger == 'tensorboard':
            self.logger = TensorBoardLogger(**logger_args, save_dir=os.getcwd(), version=version, name="glue_logs", default_hp_metric=False)
        elif logger == 'wandb':
            self.logger = WandbLogger(**logger_args)

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.learning_rate = learning_rate

        self.run_metrics = [f'hp/{task_name}_score' for task_name in _GLUE_TASK_CONFIG]

        self.fit()

    def __call__(self):
        for i, (task_name, task_attr) in enumerate(_GLUE_TASK_CONFIG.items()):
            task = task_attr['task_class'](self.tokenizer,
                                           copy.deepcopy(self.backbone),
                                           config=task_attr.get('task_config'))
            print(task, self.backbone)
            
            if self.learning_rate is not None:
                task.backbone_lr = self.learning_rate
                task.head_lr = self.learning_rate

            task_datamodule = DataModule.from_datasets('glue', task_name,
                                                       train_batch_size=self.train_batch_size,
                                                       infer_batch_size=self.infer_batch_size,
                                                       **task_attr.get('datamodule_config', {})
                                                       )

            trainer = TaskTrainer(task, log_prefix=task_name, run_metrics=self.run_metrics if i == 0 else None,
                                  logger=self.logger)

            task_metric = task_attr.get('metric', 'accuracy')
            checkpoint_callback = ModelCheckpoint(monitor=f'{task_name}/val/{task_metric}', mode='max',
                                                  save_top_k=0)
            trainer.fit(task_datamodule, gpus=4, num_nodes=1, strategy="ddp",
                        accumulate_grad_batches=self.accumulate_grad_batches,
                        callbacks=[checkpoint_callback],
                        max_epochs=task_attr.get('nb_epochs', 10))
            best_score = checkpoint_callback.best_model_score
            if best_score:
                self.logger.log_metrics({f'hp/{task_name}_score': best_score.item()})

    def fit(self):
        self()
