from engine.tasks.tasks import Task
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import engine.tasks.pretraining.utils as loss_fns
from functools import partial
from engine.tasks.pretraining.utils import prob_mask_like, mask_with_tokens, get_mask_subset_with_prob, markup_curate, params_except


@dataclass
class GptLossLogs:
    total_loss: torch.Tensor
    lm_loss: torch.Tensor
    emb_loss: torch.Tensor
    inputs: torch.Tensor
    outputs: torch.Tensor
    lm_accuracy: torch.Tensor
    optimization: dict


class GptHeadlessPretraining(Task):
    def __init__(self, tokenizer, lm_model, config=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.lm_model = lm_model

        if hasattr(lm_model, "transformer"):
            self.hidden_dim = lm_model.transformer.get_input_embeddings().weight.shape[1]
        elif hasattr(lm_model, "gpt_neox"):
            self.hidden_dim = lm_model.gpt_neox.get_input_embeddings().weight.shape[1]
        else:
            raise AttributeError(f"Could not find transformer layer in model:\n {lm_model}")
        
        self.emb_predictor = nn.Identity()

        self.device = self.lm_model.device

        self.text_key = "text"
        self.tokenized = False

        self.max_seq_len = 512

        self.learning_rate = 6e-4
        self.weight_decay = 1e-1
        self.warmup_steps = 1430
        self.total_nb_steps = 1e6

        self.emb_loss_weight = 1.
        self.lm_loss_weight = 1.
        self.loss_name = "cwt"
        self.contrastive_temperature = 1.

        self.adam_eps = 1e-8
        self.adam_betas = [0.9, 0.95]

        self.schedule_type = "cosine"

        self.load_config(config)
        self.contrastive_loss_fn = getattr(loss_fns, f"{self.loss_name}_loss")
        if "cwt" in self.loss_name:
            self.contrastive_loss_fn = partial(self.contrastive_loss_fn, temperature=self.contrastive_temperature)
        if self.lm_loss_weight == 0.:
            if hasattr(self.lm_model, "lm_head"):
                self.lm_model.lm_head = nn.Identity()
            elif hasattr(self.lm_model, "embed_out"):
                self.lm_model.embed_out = nn.Identity()

    def preprocess(self, input, **kwargs):
        self.device = self.lm_model.device
        if self.tokenized:
            return torch.stack(input[self.text_key]).T
        return self.tokenizer(input[self.text_key], return_tensors='pt', padding=True,
                              truncation=True, max_length=self.max_seq_len,
                              **kwargs).input_ids.to(self.device)

    def loss(self, input, global_step, **kwargs):
        labels = input[..., 1:]

        # get generator output and get lm loss
        lm_result = self.lm_model(input, output_hidden_states=True)

        total_loss = 0.
        emb_loss = 0.
        lm_loss = 0.

        if self.emb_loss_weight != 0.:
            if hasattr(self.lm_model, "transformer"):
                embs = self.lm_model.transformer.get_input_embeddings()
            elif hasattr(self.lm_model, "gpt_neox"):
                embs = self.lm_model.gpt_neox.get_input_embeddings()
            else:
                raise AttributeError(f"Could not find transformer layer in model:\n {self.lm_model}")
            
            target_input_embeddings = embs(labels)

            last_hidden_state = lm_result.hidden_states[-1][:, :-1]
            emb_prediction = self.emb_predictor(last_hidden_state)

            emb_loss = self.contrastive_loss_fn(
                emb_prediction.flatten(0, 1),
                target_input_embeddings.flatten(0, 1)
            )
            total_loss += self.emb_loss_weight * emb_loss
        
        if self.lm_loss_weight != 0:
            logits = lm_result.logits[:, :-1]
            lm_loss = F.cross_entropy(
                logits.transpose(1, 2),
                labels
            )
            total_loss += self.lm_loss_weight * lm_loss

        # gather metrics

        with torch.no_grad():
            lm_predictions = []
            if self.lm_loss_weight != 0:
                lm_acc = (labels == torch.argmax(logits, dim=-1)).float().mean()
            else:
                lm_acc = 0.
            
            cosines = []
            for representations in lm_result.hidden_states:
                if self.trainer.state.stage == "validate":
                    representations = representations.flatten(0, 1)
                    rand = torch.rand(len(representations))
                    indicA = rand.topk(50).indices
                    indicB = (-rand).topk(50).indices
                    all_cosine = torch.nn.functional.cosine_similarity(representations[indicA], representations[indicB])
                    cosines.append(all_cosine.mean())
                else:
                    cosines.append(0.)

        return GptLossLogs(
            total_loss=total_loss,
            lm_loss=lm_loss,
            emb_loss=emb_loss,
            inputs=input,
            outputs=lm_predictions,
            lm_accuracy=lm_acc,
            optimization={f"avg_cosine_l{i}": cosines[i] for i in range(len(cosines))}
        )

    def display_sample(self, sample, num_samples):
        sample_tokens_ids = sample.outputs[:num_samples]
        sample_input_ids = sample.inputs[:num_samples]

        sample_tokens_decoded = [self.tokenizer.convert_ids_to_tokens(di, skip_special_tokens=True) for di in sample_tokens_ids]
        sample_input_decoded = [self.tokenizer.convert_ids_to_tokens(di, skip_special_tokens=True) for di in sample_input_ids]

        for sample_idx in range(len(sample_tokens_decoded)):
            zipped_iter = zip(sample_tokens_decoded[sample_idx], sample_input_decoded[sample_idx])
            for idx, (pred, label) in enumerate(zipped_iter):
                if idx >= len(sample_tokens_decoded[sample_idx]) or label == 0 and pred == 0:
                    continue
                label_tag = '≠' if label else '='
                pred_tag = '✅' if pred == label else '❌'
                sample_tokens_decoded[sample_idx][idx] = markup_curate(sample_tokens_decoded[sample_idx][idx])
                sample_tokens_decoded[sample_idx][idx] += f'({label_tag} / {pred_tag})'

        disc_input_decoded = [self.tokenizer.convert_tokens_to_string(di) for di in sample_tokens_decoded]
        return disc_input_decoded

    def init_heads():
        return

    def configure_optimizers(self):
        lm_params_wd, lm_params_no_wd = params_except(self.lm_model, ['bias', 'LayerNorm'])

        self.param_groups = [{"params": lm_params_wd,
                              'name': 'lm_wd',
                              'weight_decay': self.weight_decay},
                              {"params": self.emb_predictor.parameters(),
                              'name': 'emb_pred',
                              'weight_decay': self.weight_decay},
                             {"params": lm_params_no_wd,
                              'name': 'lm_no_wd',
                              'weight_decay': 0.},]

        adam_opt = torch.optim.AdamW(self.param_groups,
                                     lr=self.learning_rate,
                                     betas=self.adam_betas,
                                     eps=self.adam_eps)

        schedule_type = getattr(self, "schedule_type", "cosine")
        if schedule_type == "cosine":
            lr_scheduler_func = get_cosine_schedule_with_warmup(
                adam_opt,
                self.warmup_steps,
                self.total_nb_steps
            )
        elif schedule_type == "constant":
            lr_scheduler_func = get_constant_schedule_with_warmup(
                adam_opt,
                self.warmup_steps,
            )
        else:
            raise NotImplementedError(f"LR schedule named '{schedule_type}' is not implemented.")
        lr_scheduler = {
            'scheduler': lr_scheduler_func,
            'interval': 'step',
            'frequency': 1
        }
        return [adam_opt], [lr_scheduler]
