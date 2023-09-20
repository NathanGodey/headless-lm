from engine.tasks.classification import SentenceClassification, SentencePairClassification
from engine.tasks.regression.sts import TextualSimilarity
from engine.data import DataModule
from engine.lit.lightning_module import TaskTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from engine.lit.task_logging import TensorBoardLogger, WandbLogger
import os
import copy


class RuCoLABenchmark:
    def __init__(self, tokenizer, backbone, train_batch_size=32, infer_batch_size=32,
                 accumulate_grad_batches=4, version=None, logger='tensorboard', logger_args=None):
        self.tokenizer = tokenizer
        self.backbone = backbone

        if logger == 'tensorboard':
            self.logger = TensorBoardLogger(**logger_args, save_dir=os.getcwd(), version=version, name="rucola_logs", default_hp_metric=False)
        elif logger == 'wandb':
            self.logger = WandbLogger(**logger_args)

        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

        self.run_metrics = f'hp/rucola_score'

        self.fit()

    def __call__(self):
        task = SentenceClassification(self.tokenizer,
                                        copy.deepcopy(self.backbone),
                                        config={
                                                'num_class': 2,
                                                'warmup_steps': 120,
                                                "backbone_lr": 5e-5,
                                                "head_lr": 5e-5,
                                                "weight_decay": 0.
                                            },
                                        )

        task_datamodule = DataModule.from_datasets('nthngdy/rucola',
                                                    train_batch_size=self.train_batch_size,
                                                    infer_batch_size=self.infer_batch_size,
                                                    )

        trainer = TaskTrainer(task, run_metrics=self.run_metrics,
                                logger=self.logger)

        task_metric = "accuracy"
        checkpoint_callback = ModelCheckpoint(monitor=f'val/accuracy', mode='max',
                                                save_top_k=0)
        trainer.fit(task_datamodule, gpus=1,
                    accumulate_grad_batches=self.accumulate_grad_batches,
                    callbacks=[checkpoint_callback],
                    max_epochs=10)
        best_score = checkpoint_callback.best_model_score
        if best_score:
            self.logger.log_metrics({f'hp/rucola_score': best_score.item()})

    def fit(self):
        self()
