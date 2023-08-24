from engine.lit.task_logging.base import TaskLogger
from pytorch_lightning.loggers import WandbLogger as pl_WandbLogger
import wandb


class WandbLogger(TaskLogger):
    def __init__(self, *args, **kwargs):
        self.pl_logger = pl_WandbLogger(*args, **kwargs)

    def log_text(self, *args, step=0, **kwargs):
        key, data = args
        html_data = '<br><br>'.join(data)
        self.pl_logger.experiment.log({key: wandb.Html(html_data)})

    def log_image(self, key, image, step=0):
        #ToDo: add step argument
        self.pl_logger.log_image(key, images=[image])

    def log_distribution(self, key, distribution, step=0):
        self.pl_logger.log_metrics({key: wandb.Histogram(distribution.get_values())})

    def log_hyperparams(self, params, *args, metrics=None, **kwargs):
        self.pl_logger.experiment.config.update(params, *args, allow_val_change=True, **kwargs)
        self.pl_logger.experiment.config.update(metrics, *args, allow_val_change=True, **kwargs)
