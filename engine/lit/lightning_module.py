from pytorch_lightning import LightningModule, Trainer
import torch
from torchviz import make_dot
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from engine.lit.utils import get_logger
from engine.lit.task_logging.types import Distribution


class TaskTrainer(LightningModule):
    def __init__(self, task=None, logger=None, logger_args=None, log_prefix=None, run_metrics=None,
                 display_training_samples=False, display_validation_samples=True,
                 display_sample_length=5, display_computational_graph=False):

        super().__init__()
        self.task = task
        self.task_logger = get_logger(logger, logger_args)
        self.display_training_samples = display_training_samples
        self.display_validation_samples = display_validation_samples
        self.display_sample_length = display_sample_length
        self.train_batch_size = 1
        self.infer_batch_size = 1
        self._displays_comp_graph = display_computational_graph
        self.dataset_name = 'NA'
        self.log_prefix = log_prefix
        self.run_metrics = run_metrics if run_metrics else ['hp_metric']
        self._stored_histogram_dict = {}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['task'] = self.task

    def on_load_checkpoint(self, checkpoint):
        self.task = checkpoint['task']

    def log(self, *args, **kwargs):
        if self.log_prefix:
            args = list(args)
            args[0] = f'{self.log_prefix}/{args[0]}'
        return super().log(*args, **kwargs)

    def log_hparams(self):
        if self.logger is None:
            return
        hparams_dict = {
            'task': type(self.task).__name__,
            'dataset': self.dataset_name,
            'train_batch_size': self.accumulate_grad_batches * self.train_batch_size,
            'val_batch_size': self.accumulate_grad_batches * self.infer_batch_size,
        }

        metrics_dict = {metric_name: -1 for metric_name in self.run_metrics}

        for task_attr_name in dir(self.task):
            if task_attr_name.startswith("_"):
                continue
            attr_value = getattr(self.task, task_attr_name)
            if isinstance(attr_value, torch.nn.Module):
                transformer_name = getattr(attr_value, 'name_or_path', attr_value.__class__.__name__)
                hparams_dict[task_attr_name] = transformer_name.split('/')[-1]
            elif isinstance(attr_value, torch.nn.Parameter):
                hparams_dict[task_attr_name] = attr_value.item()
            elif isinstance(attr_value, (int, float, str)):
                hparams_dict[task_attr_name] = attr_value

        hparams_dict.pop('training', None)
        hparams_dict.pop('dump_patches', None)

        self.task_logger.log_hyperparams(params=hparams_dict, metrics=metrics_dict)

    def display_computational_graph(self, loss_tensor):
        self._displays_comp_graph = False

        prev_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(30000)
        graph = make_dot(loss_tensor, params=dict(self.task.named_parameters()))
        graph.format = 'pdf'
        os.makedirs('graphviz_logs', exist_ok=True)
        graph.render(directory='graphviz_logs', view=True)
        sys.setrecursionlimit(prev_recursion_limit)

    def on_fit_start(self):
        if self.task is None:
            raise AttributeError('No task was provided to the TaskTrainer object.')
        self.log_hparams()

    def common_step(self, input):
        processed_input = self.task.preprocess(input)
        self.task._pre_loss_step()
        return self.task.loss(processed_input, global_step=self.global_step)

    def log_text(self, *args, **kwargs):
        if self.logger:
            if self.log_prefix:
                args = list(args)
                args[0] = f'{self.log_prefix}/{args[0]}'
            return self.task_logger.log_text(*args, step=self.global_step, **kwargs)

    def log_metrics(self, step_name, loss_info, batch_size):
        if loss_info is None or self.logger is None:
            return
        for field in loss_info.__dataclass_fields__:
            value = getattr(loss_info, field)
            if field == 'optimization':
                for key, metric in value.items():
                    self.log(f'optimization/{key}', metric, batch_size=batch_size)
            elif torch.is_tensor(value) and not value.shape:
                self.log(f'{step_name}/{field}', value, batch_size=batch_size)
            elif isinstance(value, Distribution):
                self.task_logger.log_distribution(f'{step_name}/{field}', value, step=self.global_step)

        lr_scheduler = self.lr_schedulers()
        if lr_scheduler and not isinstance(lr_scheduler, list):
            self.log('optimization/learning_rate', lr_scheduler.get_last_lr()[0], batch_size=batch_size)

    def _log_stored_histograms(self, step_name, display_top_k=15):
        matplotlib.rcParams['font.family'] = ['Noto Sans']
        for key, histogram_dict in self._stored_histogram_dict.items():
            top_elements = histogram_dict.most_common(display_top_k)
            total_nb_elements = sum(histogram_dict.values())
            elements, counts = zip(*top_elements)
            fig = plt.figure()
            plt.bar(elements, np.array(counts)/total_nb_elements)
            plt.xticks(rotation=45, horizontalalignment='right')
            plt.subplots_adjust(bottom=0.15)

            self.task_logger.log_figure(f'{step_name}/{key}', fig, step=self.global_step)

    def log_histograms(self, step_name, histogram_dict, display_top_k=15):
        if not histogram_dict:
            return
        for key, data in histogram_dict.items():
            if key not in self._stored_histogram_dict:
                self._stored_histogram_dict[key] = Counter(data)
            else:
                self._stored_histogram_dict[key].update(data)

            if self.global_step % self.trainer.log_every_n_steps == 0:
                self._log_stored_histograms(step_name)

    def training_step(self, input, batch_idx):
        train_loss_info = self.common_step(input)
        if self._displays_comp_graph:
            self.display_computational_graph(train_loss_info.total_loss)
        if self.display_training_samples:
            samples_to_display = self.task.display_sample(train_loss_info,
                                                          self.display_sample_length)
            self.log_text('train/samples', samples_to_display)

            histogram_data_to_display = self.task.display_histograms(train_loss_info)
            self.log_histograms('train', histogram_data_to_display)

        self.log_metrics('train', train_loss_info, self.train_batch_size)

        return train_loss_info.total_loss

    def compute_epoch_metrics(self):
        labs, preds = self.task.get_stored_labs_and_preds()
        if labs is not None and preds is not None:
            return self.task.get_epoch_metrics(labs, preds)

    def on_validation_epoch_start(self):
        epoch_metrics = self.compute_epoch_metrics()
        if epoch_metrics:
            self.log_metrics('train', epoch_metrics, self.infer_batch_size)
        self.task.empty_stored_attrs()

    def validation_step(self, input, batch_idx):
        val_loss_info = self.common_step(input)
        if self.display_validation_samples:
            if batch_idx == 0:
                samples_to_display = self.task.display_sample(val_loss_info,
                                                              self.display_sample_length)
                self.log_text('val/samples', samples_to_display)
            histogram_data_to_display = self.task.display_histograms(val_loss_info)
            self.log_histograms('val', histogram_data_to_display)

        self.log_metrics('val', val_loss_info, self.infer_batch_size)

        return val_loss_info.total_loss

    def on_validation_epoch_end(self):
        epoch_metrics = self.compute_epoch_metrics()
        if epoch_metrics:
            self.log_metrics('val', epoch_metrics, self.infer_batch_size)
        self._log_stored_histograms('val')
        self.task.empty_stored_attrs()

    def configure_optimizers(self):
        return self.task.configure_optimizers()

    def fit(self, datamodule, *args, **kwargs):
        self.train_batch_size = datamodule.train_batch_size
        self.infer_batch_size = datamodule.infer_batch_size
        self.accumulate_grad_batches = kwargs.get('accumulate_grad_batches', 1)
        self.ckpt_path = kwargs.pop('ckpt_path', None)
        self.total_nb_steps = getattr(self.task, "total_nb_steps", 1e6)

        self.dataset_name = datamodule.name

        trainer = Trainer(reload_dataloaders_every_n_epochs=1, logger=self.task_logger.pl_logger, max_steps=self.total_nb_steps, *args, **kwargs)
        self.task.trainer = trainer
        trainer.fit(model=self, datamodule=datamodule, ckpt_path=self.ckpt_path)
        return trainer.logged_metrics
