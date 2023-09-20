from engine.tasks.base.sentence_to_scalar import SentenceToScalar
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as M


@dataclass
class SentencePairClassificationLossLogs:
    total_loss: torch.Tensor
    inputs: tuple
    outputs: torch.Tensor
    f1_score: torch.Tensor
    accuracy: torch.Tensor


class SentencePairClassification(SentenceToScalar):
    def __init__(self, tokenizer, backbone, config=None):
        super().__init__(tokenizer, backbone)

        self.num_class = 2
        self.text_keys = ['sentence1', 'sentence2']

        self.load_config(config)

        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(backbone.config.hidden_size, self.num_class,
                      device=self.device)
        )

    def extract_input_info(self, input):
        return self.extract_input_sentences(input)

    def loss(self, input, **kwargs):
        batch_sent, batch_labels, init_sents = input

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
            inputs = (list(zip(*init_sents)), batch_labels)
            outputs = predicted_labels
            f1_score = M.f1_score(predicted_labels.flatten(), batch_labels.flatten(), task="multiclass", average='macro', num_classes=self.num_class)
            accuracy = M.accuracy(predicted_labels, batch_labels, task="multiclass", average='micro',
                                  num_classes=self.num_class)

        return SentencePairClassificationLossLogs(total_loss=bce_loss,
                                                  inputs=inputs,
                                                  outputs=outputs,
                                                  f1_score=f1_score,
                                                  accuracy=accuracy)

    def display_sample(self, sample, num_samples):
        batch_sents_pairs, batch_labels = sample.inputs
        sample_sents_pairs = batch_sents_pairs[:num_samples]
        sample_labels = batch_labels[:num_samples]
        sample_predictions = sample.outputs[:num_samples]

        zipped_iter = zip(sample_sents_pairs, sample_labels, sample_predictions)
        pretty_sents = []

        for (sent1, sent2), lab, pred_lab in zipped_iter:
            pred_tag = '✅' if pred_lab == lab else '❌'
            pretty_sents.append(f'1 - {sent1}<br>2 - {sent2}<br>{pred_tag}({lab}, {pred_lab})')

        return pretty_sents

    def init_heads():
        return
