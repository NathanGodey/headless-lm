import os
from engine.lit.task_logging import TensorBoardLogger, WandbLogger


def get_checkpoint_from_version(version_name, log_dir='lightning_logs'):
    assert os.path.isdir(log_dir), f'{log_dir} is not a directory'
    assert version_name in os.listdir(log_dir), f'{version_name} is not a subdirectory of {log_dir}'

    version_path = os.path.join(log_dir, version_name)

    assert 'checkpoints' in os.listdir(version_path), f"Could not find any 'checkpoints' directory in {version_path}"

    checkpoints_path = os.path.join(version_path, 'checkpoints')

    checkpoints = [os.path.join(checkpoints_path, file_name)
                   for file_name in os.listdir(checkpoints_path) if file_name.endswith('ckpt')]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoint (ckpt) file could be found in {checkpoints_path}.')

    return checkpoints


def get_logger(logger, logger_args):
    logger_args = logger_args if logger_args else dict()
    if logger == 'tensorboard' or logger is None:
        return TensorBoardLogger(**logger_args)
    elif logger == 'wandb':
        return WandbLogger(**logger_args)
    else:
        return logger
