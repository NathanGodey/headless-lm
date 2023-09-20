from engine.lit.task_logging.base import TaskLogger


class TensorBoardLogger(TaskLogger):
    def __init__(self, *args, save_dir='lightning_logs', **kwargs):
        from pytorch_lightning.loggers import TensorBoardLogger as pl_TensorBoardLogger
        self.pl_logger = pl_TensorBoardLogger(*args, save_dir=save_dir, **kwargs)

    def log_text(self, *args, step=0, **kwargs):
        field_name, data = args
        text_data = '<br><br>'.join(data)
        self.pl_logger.experiment.add_text(field_name, text_data, global_step=step, **kwargs)

    def log_image(self, *args, step=0, **kwargs):
        self.pl_logger.experiment.add_image(*args, global_step=step, **kwargs)

    def log_figure(self, *args, step=0, **kwargs):
        self.pl_logger.experiment.add_figure(*args, global_step=step, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        self.pl_logger.log_hyperparams(*args, **kwargs)
