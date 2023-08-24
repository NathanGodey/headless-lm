from abc import ABC, abstractmethod
import os
import json
import torch.nn as nn


class Task(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.trainer = None

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def display_sample(self, sample, *args, **kwargs):
        print(sample)

    def display_histograms(self, sample):
        pass

    @property
    @abstractmethod
    def init_heads(self):
        pass

    def _pre_loss_step(self):
        return

    def get_stored_labs_and_preds(self):
        return None, None

    def get_epoch_metrics(self, labels, predictions):
        return

    def empty_stored_attrs(self):
        for attr_name in dir(self):
            if attr_name.startswith('_stored_'):
                attr_class = getattr(self, attr_name).__class__
                setattr(self, attr_name, attr_class())

    def load_config(self, config):
        if config is None:
            return

        if isinstance(config, dict):
            self.load_config_from_dict(config_dict=config)
        elif isinstance(config, str):
            assert os.path.isfile(config)
            self.load_config_from_file(config_path=config)
        else:
            raise AttributeError(f'Unrecognized config argument of type {type(config)}.')

    def load_config_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                error_msg = f"Unknown parameter {key} for {type(self).__name__}. Check the config file."
                raise AttributeError(error_msg)

    def load_config_from_file(self, config_path):
        _, file_extension = os.path.splitext(config_path)
        file_extension = file_extension[1:]
        load_method_name = f'load_config_from_{file_extension}'
        if hasattr(self, load_method_name):
            return getattr(self, load_method_name)(config_path)
        raise NotImplementedError(f'Cannot parse config from {file_extension} files.')

    def load_config_from_json(self, config_json):
        config_dict = json.load(open(config_json))
        self.load_config_from_dict(config_dict)
