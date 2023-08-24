from engine.tasks.tasks import Task
from engine.tasks.base.sentence_to_scalar import SentenceToScalar
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as M
from transformers import get_constant_schedule_with_warmup


@dataclass
class TextualSimilarityLossLogs:
    total_loss: torch.Tensor
    inputs: torch.Tensor
    outputs: torch.Tensor


@dataclass
class TextualSimilarityEpochLogs:
    pearson: torch.Tensor
    spearman: torch.Tensor


class TextualSimilarity(SentenceToScalar):
    def __init__(self, tokenizer, backbone, config=None):
        super().__init__(tokenizer, backbone)
        self.text_keys = ['sentence1', 'sentence2']

        self.load_config(config)

        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(backbone.config.hidden_size, 1,
                      device=self.device)
        )

    def extract_input_info(self, input):
        return self.extract_input_sentences(input)

    def loss(self, input, **kwargs):
        batch_sent, batch_labels, init_sents = input
        batch_labels = batch_labels.float()

        representations = self.backbone(batch_sent)[0]

        sentence_representations = self.pool_from_representations(representations)

        predicted_values = self.head(sentence_representations).flatten()

        mse_loss = F.mse_loss(
            predicted_values, batch_labels
        )

        # gather metrics
        with torch.no_grad():
            inputs = (list(zip(*init_sents)), batch_labels)
            outputs = predicted_values

            self._stored_labels.append(batch_labels.cpu())
            self._stored_predictions.append(predicted_values.cpu())

        return TextualSimilarityLossLogs(total_loss=mse_loss,
                                         inputs=inputs,
                                         outputs=outputs)

    def get_epoch_metrics(self, labels, predictions):
        return TextualSimilarityEpochLogs(
            pearson=M.pearson_corrcoef(predictions, labels),
            spearman=M.spearman_corrcoef(predictions, labels)
        )

    def display_sample(self, sample, num_samples):
        batch_sents_pairs, batch_labels = sample.inputs
        sample_sents_pairs = batch_sents_pairs[:num_samples]
        sample_labels = batch_labels[:num_samples]
        sample_predictions = sample.outputs[:num_samples]

        zipped_iter = zip(sample_sents_pairs, sample_labels, sample_predictions)
        pretty_sents = []

        for (sent1, sent2), lab, pred_lab in zipped_iter:
            pretty_sents.append(f'1 - {sent1}<br>2 - {sent2}<br>gt={lab}, pred={pred_lab}, delta={abs(lab-pred_lab)}')

        return pretty_sents

    def init_heads():
        return
