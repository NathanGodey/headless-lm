from engine.tasks.tasks import Task
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import engine.tasks.pretraining.utils as loss_fns
from functools import partial
from engine.tasks.pretraining.utils import prob_mask_like, mask_with_tokens, get_mask_subset_with_prob, markup_curate


@dataclass
class MlmLossLogs:
    total_loss: torch.Tensor
    mlm_loss: torch.Tensor
    emb_loss: torch.Tensor
    inputs: torch.Tensor
    mlm_mask: torch.Tensor
    outputs: torch.Tensor
    mlm_accuracy: torch.Tensor
    optimization: dict


class MlmHeadlessPretraining(Task):
    def __init__(self, tokenizer, mlm_model, config=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.mlm_model = mlm_model
        self.hidden_dim = mlm_model.get_input_embeddings().weight.shape[1]
        self.emb_predictor = nn.Identity()

        self.device = self.mlm_model.device

        # mlm related probabilities
        self.mask_prob = 0.15
        self.replace_prob = 0.8

        self.num_tokens = len(tokenizer.vocab)
        self.random_token_prob = 0.1

        # token ids
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_ignore_token_ids = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        ]

        self.text_key = "text"
        self.tokenized = False

        self.max_seq_len = 128

        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.warmup_steps = 1e4
        self.total_nb_steps = 1e6
        self.adam_betas = [0.9, 0.999]

        self.emb_loss_weight = 1.
        self.mlm_loss_weight = 1.
        self.loss_name = "cwt"
        self.contrastive_temperature = 1.

        self.load_config(config)

        self.mask_ignore_token_ids = set([*self.mask_ignore_token_ids, self.pad_token_id])
        self.contrastive_loss_fn = getattr(loss_fns, f"{self.loss_name}_loss")
        if self.loss_name == "cwt":
            self.contrastive_loss_fn = partial(self.contrastive_loss_fn, temperature=self.contrastive_temperature)
        if self.mlm_loss_weight == 0.:
            self.mlm_model.cls = nn.Identity()
        
        print(self.mlm_model)


    def _randomize_mask_input(self, input):
        random_token_prob = prob_mask_like(input, self.random_token_prob)
        random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
        random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
        random_token_prob &= ~random_no_mask
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)
        input[random_indices] = random_tokens[random_indices]

    def _mask_input(self, input):
        replace_prob = prob_mask_like(input, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # set inverse of mask to ignored index
        mlm_labels = input.masked_fill(~mask, -100)

        input = input.masked_fill(mask * replace_prob, self.mask_token_id)

        return input, mlm_labels, mask, mask_indices

    def _sample(self, logits, mask_indices):
        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = sample_logits.argmax(-1)

        return sampled

    def _get_embeddings(sampled, embeddings):
        return torch.mm(sampled, embeddings)

    def preprocess(self, input, **kwargs):
        self.device = self.mlm_model.device
        if self.tokenized:
            return torch.stack(input[self.text_key]).T
        return self.tokenizer(input[self.text_key], return_tensors='pt', padding=True,
                              truncation=True, max_length=self.max_seq_len,
                              **kwargs).input_ids.to(self.device)

    def loss(self, input, global_step, **kwargs):
        b, t = input.shape
        if t > self.max_seq_len:
            ratio = t//self.max_seq_len
            input = input.reshape((b*ratio, t//ratio))

        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'
            self._randomize_mask_input(masked_input)

        masked_input, mlm_labels, mask, _ = self._mask_input(masked_input)

        # get generator output and get mlm loss
        mlm_result = self.mlm_model(masked_input, output_hidden_states=True)

        if self.emb_loss_weight != 0:
            last_hidden_state = mlm_result.hidden_states[-1][mask]
            emb_prediction = self.emb_predictor(last_hidden_state)

            embs = self.mlm_model.get_input_embeddings()
            target_input_embeddings = embs(input)[mask]
            emb_loss = self.contrastive_loss_fn(
                emb_prediction,
                target_input_embeddings
            )
        else:
            emb_loss = 0.
        
        if self.mlm_loss_weight != 0:
            logits = mlm_result.logits
            mlm_loss = F.cross_entropy(
                logits.transpose(1, 2),
                mlm_labels
            )
        else:
            mlm_loss = 0.

        # gather metrics

        with torch.no_grad():
            mlm_predictions = []
            if self.mlm_loss_weight != 0:
                mlm_predictions = torch.argmax(logits, dim=-1)
                mlm_acc = (mlm_labels[mask] == mlm_predictions[mask]).float().mean()
            else:
                mlm_acc = 0.
            
            cosines = []
            for representations in mlm_result.hidden_states:
                if self.trainer.state.stage == "validate":
                    representations = representations.flatten(0, 1)
                    rand = torch.rand(len(representations))
                    indicA = rand.topk(50).indices
                    indicB = (-rand).topk(50).indices
                    all_cosine = torch.nn.functional.cosine_similarity(representations[indicA], representations[indicB])
                    cosines.append(all_cosine.mean())
                else:
                    cosines.append(0.)

        return MlmLossLogs(
            total_loss=self.mlm_loss_weight*mlm_loss+self.emb_loss_weight*emb_loss,
            mlm_loss=mlm_loss,
            emb_loss=emb_loss,
            inputs=input,
            mlm_mask=mask,
            outputs=mlm_predictions,
            mlm_accuracy=mlm_acc,
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
        self.param_groups = [{"params": self.mlm_model.parameters(),
                              'name': 'mlm_wd',
                              'weight_decay': self.weight_decay},
                              {"params": self.emb_predictor.parameters(),
                              'name': 'emb_pred',
                              'weight_decay': self.weight_decay}]

        adam_opt = torch.optim.AdamW(self.param_groups,
                                     lr=self.learning_rate,
                                     betas=self.adam_betas,
                                     weight_decay=self.weight_decay)
        lr_scheduler_func = get_linear_schedule_with_warmup(
            adam_opt,
            self.warmup_steps,
            self.total_nb_steps
        )
        lr_scheduler = {
            'scheduler': lr_scheduler_func,
            'interval': 'step',
            'frequency': 1
        }
        return [adam_opt], [lr_scheduler]
