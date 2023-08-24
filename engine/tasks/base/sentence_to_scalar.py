from engine.tasks.tasks import Task
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as M
from transformers import get_constant_schedule_with_warmup


class SentenceToScalar(Task):
    def __init__(self, tokenizer, backbone):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.text_keys = ['sentence']
        self.label_keys = ['label']

        self.device = self.backbone.device

        # token ids
        self.pad_token_id = 0
        self.mask_token_id = 2
        self.mask_ignore_token_ids = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        ]
        self.max_seq_len = 128

        self.backbone_lr = 5e-5
        self.head_lr = 5e-5
        self.warmup_steps = 50
        self.weight_decay = 0.
        self.dropout = 0.1

        self.pooling = 'cls'

        self._stored_predictions = []
        self._stored_labels = []

    def extract_input_sentences(self, input):
        try:
            return tuple(input[key] for key in self.text_keys)

        except (KeyError, TypeError) as err:
            print(err)
            if isinstance(err, TypeError):
                reason = f"default 'input['text']' is not a valid operation for input of type {type(input)}"
            elif isinstance(err, KeyError):
                reason = f"{err} key does not exist in input. Possible keys: {list(input.keys())}"
            error_msg = f"extract_input_sentences method is not implemented for object {self.__class__.__name__}, and {reason}."
            raise NotImplementedError(
                error_msg
            )

    def extract_input_labels(self, input):
        try:
            return tuple(input[key] for key in self.label_keys)

        except (KeyError, TypeError) as err:
            print(err)
            if isinstance(err, TypeError):
                reason = f"default 'input['label']' is not a valid operation for input of type {type(input)}"
            elif isinstance(err, KeyError):
                reason = f"{err} key does not exist in input. Possible keys: {list(input.keys())}"
            error_msg = f"extract_input_labels method is not implemented for object {self.__class__.__name__}, and {reason}."
            raise NotImplementedError(
                error_msg
            )

    def extract_input_info(self, input):
        return

    def preprocess(self, input, **kwargs):
        self.device = self.backbone.device
        input_sentences_tuple = self.extract_input_sentences(input)
        input_labels_tuple = self.extract_input_labels(input)
        input_info = self.extract_input_info(input)

        tokenized_input = self.tokenizer(*input_sentences_tuple, return_tensors='pt', padding=True,
                                         truncation=True, max_length=self.max_seq_len,
                                         **kwargs).input_ids.to(self.device)

        if input_info:
            return tokenized_input, *input_labels_tuple, input_info
        return tokenized_input, *input_labels_tuple

    def pool_from_representations(self, representations):
        if self.pooling == 'mean':
            return representations.mean(1)
        else:
            return representations[:, 0]

    def _pre_loss_step(self):
        assert hasattr(self, 'head'), 'A head module needs to be implemented.'

    def loss(self, input, **kwargs):
        raise NotImplementedError(f'A loss method is required for object {self.__class__.__name__}.')

    def get_stored_labs_and_preds(self):
        if not len(self._stored_predictions):
            return None, None
        labs, preds = torch.cat(tuple(self._stored_labels)), torch.cat(tuple(self._stored_predictions))
        return labs, preds

    def init_heads():
        return

    def configure_optimizers(self):
        adam_opt = torch.optim.Adam([
            {'params': self.backbone.parameters(), 'lr': self.backbone_lr},
            {'params': self.head.parameters(), 'lr': self.head_lr}
            ],
                                    weight_decay=self.weight_decay)
        lr_scheduler_func = get_constant_schedule_with_warmup(
            adam_opt,
            self.warmup_steps
        )
        lr_scheduler = {
            'scheduler': lr_scheduler_func,
            'interval': 'step',
            'frequency': 1
        }
        return [adam_opt], [lr_scheduler]
