from engine.tasks.tasks import Task
from engine.tasks.base.sentence_to_scalar import SentenceToScalar
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as M
from transformers import get_constant_schedule_with_warmup


@dataclass
class SentenceClassificationLossLogs:
    total_loss: torch.Tensor
    inputs: torch.Tensor
    outputs: torch.Tensor
    accuracy: torch.Tensor
    f1_score: torch.Tensor


@dataclass
class SentenceClassificationEpochLogs:
    mcc: torch.Tensor


class SentenceClassification(SentenceToScalar):
    def __init__(self, tokenizer, backbone, config=None):
        super().__init__(tokenizer, backbone)

        self.num_class = 2

        self.load_config(config)

        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(backbone.config.hidden_size, self.num_class,
                      device=self.device)
        )

    def loss(self, input, **kwargs):
        batch_sent, batch_labels = input

        representations = self.backbone(batch_sent)[0]

        sentence_representations = self.pool_from_representations(representations)

        predicted_logits = self.head(sentence_representations)

        bce_loss = F.cross_entropy(
            predicted_logits.softmax(-1),
            batch_labels,
        )

        predicted_labels = predicted_logits.detach().argmax(-1)

        # gather metrics
        with torch.no_grad():
            inputs = input
            outputs = predicted_labels
            f1_score = M.f1_score(predicted_labels, batch_labels, task="multiclass", average='macro', num_classes=self.num_class)
            accuracy = M.accuracy(predicted_labels, batch_labels, task="multiclass", average='micro',
                                  num_classes=self.num_class)

            self._stored_labels.append(batch_labels.cpu())
            self._stored_predictions.append(predicted_labels.cpu())

        return SentenceClassificationLossLogs(total_loss=bce_loss,
                                              inputs=inputs,
                                              outputs=outputs,
                                              f1_score=f1_score,
                                              accuracy=accuracy)

    def get_epoch_metrics(self, labels, predictions):
        return SentenceClassificationEpochLogs(
            mcc=M.matthews_corrcoef(predictions,
                                    labels,
                                    num_classes=self.num_class,
                                    task="multiclass")
        )

    def display_sample(self, sample, num_samples):
        batch_sents, batch_labels = sample.inputs
        sample_sents = batch_sents[:num_samples]
        sample_labels = batch_labels[:num_samples]
        sample_predictions = sample.outputs[:num_samples]

        sample_sents_decoded = self.tokenizer.batch_decode(sample_sents, skip_special_tokens=True)

        zipped_iter = enumerate(zip(sample_sents_decoded, sample_labels, sample_predictions))
        pretty_sents = []

        for sample_idx, (sent, lab, pred_lab) in zipped_iter:
            pred_tag = '✅' if pred_lab == lab else '❌'
            pretty_sents.append(f'{sent}  {pred_tag} ({lab}, {pred_lab})')

        return pretty_sents

    def init_heads():
        return
