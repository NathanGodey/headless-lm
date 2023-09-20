import torch
import math
from functools import reduce
import torchmetrics.functional as M
from transformers import get_linear_schedule_with_warmup as _get_linear_schedule_with_warmup


def cwt_loss(A, B, temperature=1.):
    exp_cosine_sim = torch.exp(torch.mm(A, B.T)/temperature)
    self_dist = exp_cosine_sim.diagonal()
    neg_dist = exp_cosine_sim.sum(-1)
    return - (self_dist/(neg_dist + 1e-9)).log().mean()


def params_except(module, filter = None):
    if not filter:
        filter = []
    remaining_parameters = [p for n, p in module.named_parameters() if not any(nd in n for nd in filter)]
    filtered_parameters = [p for n, p in module.named_parameters() if any(nd in n for nd in filter)]
    return remaining_parameters, filtered_parameters


def get_linear_schedule_with_warmup(*args, shutdown_every_n_steps=None, shutdown_offsets=None, **kwargs):
    linear_schedule_with_warmup = _get_linear_schedule_with_warmup(*args, **kwargs)
    if shutdown_every_n_steps:
        lr_lambdas = linear_schedule_with_warmup.lr_lambdas
        for i, (alternate_freq, shutdown_offset) in enumerate(zip(shutdown_every_n_steps, shutdown_offsets)):
            lr_lambdas[i] = alternate_lambda(lr_lambdas[i], alternate_freq, shutdown_offset)
        linear_schedule_with_warmup.lr_lambdas = lr_lambdas
    return linear_schedule_with_warmup


def markup_curate(token_str):
    if token_str[0] == '[' and token_str[-1] == ']':
        token_str = token_str[1:-1]
    return token_str


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()
