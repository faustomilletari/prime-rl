from dataclasses import dataclass

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor
import torch.distributed as dist
from zeroband.training.config import ClippingConfig, GRPOVariantsConfig, KlCovConfig, RatioConfig


@dataclass
class RatioInfo:
    ratio_sum: Float[Tensor, "1"]
    clipped_token_count: Float[Tensor, "1"]

    raw_ratio_sum: Float[Tensor, "1"]
    raw_ratio_max: Float[Tensor, "1"]
    raw_ratio_min: Float[Tensor, "1"]

    raw_ratio_abs_sum: Float[Tensor, "1"]


@jaxtyped(typechecker=typechecker)
def grpo_loss(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    grpo_loss_config: GRPOVariantsConfig,
) -> tuple[Tensor, RatioInfo]:
    if isinstance(grpo_loss_config, ClippingConfig):
        return grpo_loss_clip(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            grpo_loss_config.epsilon_low,
            grpo_loss_config.epsilon_high,
            grpo_loss_config.clip_ratio,
            max_tokens,
            grpo_loss_config.highest_entropy_ratio_loss,
        )
    elif isinstance(grpo_loss_config, RatioConfig):
        return grpo_loss_ratio(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            max_tokens,
            grpo_loss_config.clip_ratio,
            grpo_loss_config.highest_entropy_ratio_loss,
        )

    elif isinstance(grpo_loss_config, KlCovConfig):
        return grpo_loss_kl_cov(
            logits,
            input_ids,
            advantages,
            original_logprobs,
            loss_mask,
            temperature,
            max_tokens,
            grpo_loss_config.kl_coef,
            grpo_loss_config.k_percent,
            grpo_loss_config.highest_entropy_ratio_loss,
        )
    else:
        raise ValueError(f"Invalid grpo_loss_type: {grpo_loss_config.type}")


@jaxtyped(typechecker=typechecker)
def grpo_loss_clip(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    epsilon_low: float,
    epsilon_high: float,
    clip_ratio: float,
    max_tokens: int,
    highest_entropy_percentage: float,
) -> tuple[Tensor, RatioInfo]:
    """
    DeepSeek Math Loss: https://arxiv.org/abs/2402.03300

    Args:
        policy_logprobs: Log probabilities from the policy model
        ref_logprobs: Log probabilities from the reference model
        advantages: Advantages for each token
        beta: KL penalty coefficient
        epsilon: Clipping parameter for PPO
        ignore_index: Specifies a target value that is ignored and does not contribute to the loss
    """
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    raw_ratio = torch.exp(per_token_logps - original_logprobs)
    coef_1 = torch.clamp(raw_ratio, 0, clip_ratio)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = -coef_1 * advantages
    per_token_loss2 = -coef_2 * advantages
    per_token_loss = torch.max(per_token_loss1, per_token_loss2)

    is_clipped = (per_token_loss1 < per_token_loss2).float()
    clipped_token_count = (is_clipped * loss_mask).sum()

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    raw_ratio = (raw_ratio.detach() - 1) * loss_mask
    ratio = (coef_2.detach() - 1) * loss_mask

    return loss, RatioInfo(
        ratio_sum=ratio.sum().float(),
        clipped_token_count=clipped_token_count.float(),
        raw_ratio_sum=raw_ratio.sum().float(),
        raw_ratio_max=raw_ratio.max().float() + 1,
        raw_ratio_min=raw_ratio.min().float() + 1,
        raw_ratio_abs_sum=raw_ratio.abs().sum().float(),
    )


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss_ratio(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    clip_ratio: float,
    highest_entropy_percentage: float,
) -> tuple[Tensor, RatioInfo]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    raw_ratio = torch.exp(per_token_logps - original_logprobs)

    is_clipped = (raw_ratio > clip_ratio).float()
    clipped_token_count = (is_clipped * loss_mask).sum()

    ratio = torch.clamp(raw_ratio, 0, clip_ratio)
    per_token_loss = -ratio * advantages

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    loss = _apply_mask(per_token_loss, loss_mask, max_tokens)

    raw_ratio = (raw_ratio.detach() - 1) * loss_mask
    ratio = (ratio.detach() - 1) * loss_mask

    return loss, RatioInfo(
        ratio_sum=ratio.sum().float(),
        clipped_token_count=clipped_token_count.float(),
        raw_ratio_sum=raw_ratio.sum().float(),
        raw_ratio_max=raw_ratio.max().float() + 1,
        raw_ratio_min=raw_ratio.min().float() + 1,
        raw_ratio_abs_sum=raw_ratio.abs().sum().float(),
    )


# beartype here just make sure we have the correct shape
@jaxtyped(typechecker=typechecker)
def grpo_loss_kl_cov(
    logits: Float[Tensor, "batch seq vocab"],
    input_ids: Int[Tensor, "batch seq"],
    advantages: Float[Tensor, "batch seq"],
    original_logprobs: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    temperature: float,
    max_tokens: int,
    kl_coef_cov: float,
    k_percent: float,
    highest_entropy_percentage: float,
) -> tuple[Tensor, RatioInfo]:
    # we start by dropping the bos token because it does not have a corresponding logit
    input_ids = input_ids[:, 1:]
    advantages = advantages[:, 1:]
    loss_mask = loss_mask[:, 1:]

    # from the logits we drop the last logits because it corresponds to the next token that will be sample but is not here yet
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction

    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    per_token_logps = selective_log_softmax(logits, input_ids)

    negative_approx_kl = per_token_logps - original_logprobs

    abs_kl = negative_approx_kl.abs()

    ratio = torch.exp(negative_approx_kl)

    ppo_kl_abs = (abs_kl * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    pg_losses1 = -advantages * ratio

    pg_losses_kl = -advantages * ratio + kl_coef_cov * abs_kl

    pg_losses = pg_losses1

    all_valid = loss_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = per_token_logps[all_valid].detach().reshape(-1).cpu()

    k = min(k_percent, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k / 100))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    if highest_entropy_percentage < 1.0:
        loss_mask = highest_entropy_mask(logits, loss_mask, highest_entropy_percentage)

    pg_loss = _apply_mask(pg_losses, loss_mask, max_tokens)

    # For kl_cov variant, we track the ratio for monitoring
    raw_ratio = (ratio.detach() - 1) * loss_mask
    
    # No clipping is done in kl_cov variant, so clipped_token_count is 0
    return pg_loss, RatioInfo(
        ratio_sum=raw_ratio.sum().float(),
        clipped_token_count=torch.tensor(0.0, device=logits.device),
        raw_ratio_sum=raw_ratio.sum().float(),
        raw_ratio_max=raw_ratio.max().float() + 1,
        raw_ratio_min=raw_ratio.min().float() + 1,
        raw_ratio_abs_sum=raw_ratio.abs().sum().float(),
    )


def selective_log_softmax(logits, index):
    """
    credits to https://github.com/huggingface/trl/blob/07cfe1677e552b7d5c92b7740e5b2f0b057661d8/trl/trainer/utils.py#L1659

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


@jaxtyped(typechecker=typechecker)
def entropy_loss(
    logits: Float[Tensor, "batch seq vocab"], loss_mask: Int[Tensor, "batch seq"], temperature: float, max_tokens: int
) -> Tensor:
    return _compile_entropy_loss(logits=logits, loss_mask=loss_mask, temperature=temperature, max_tokens=max_tokens)


# @torch.compile
def _compile_entropy_loss(logits: torch.Tensor, loss_mask: torch.Tensor, temperature: float, max_tokens: int):
    logits = logits[:, :-1, :]
    logits = logits / temperature

    loss_mask = loss_mask[:, 1:]
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

    return _apply_mask(entropy, loss_mask, max_tokens)


@jaxtyped(typechecker=typechecker)
def kl_penalty(
    logprob: Float[Tensor, "batch seq_minus_1"],
    ref_logprob: Float[Tensor, "batch seq_minus_1"],
    loss_mask: Int[Tensor, "batch seq"],
    max_tokens: int,
) -> Float[Tensor, ""]:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py#L351

    Args:
        logprob:
        ref_logprob:

    Returns:

    """

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    loss_mask = loss_mask[:, 1:]

    kl = ref_logprob - logprob
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    kl = torch.clamp(kld, min=-10, max=10)
    return _apply_mask(kl, loss_mask, max_tokens)


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor, max_tokens: int) -> torch.Tensor:
    return (tensor * mask).sum() / max_tokens


@jaxtyped(typechecker=typechecker)
def highest_entropy_mask(
    logits: Float[Tensor, "batch seq vocab"],
    loss_mask: Int[Tensor, "batch seq"],
    percent: float,
) -> Tensor:
    """
    Returns a mask (batch, seq) where the top `percent` of masked tokens (loss_mask==1)
    with the highest entropy are 1, others 0.
    Args:
        logits: Tensor of shape (batch, seq, vocab)
        loss_mask: Tensor of shape (batch, seq), 1 for valid tokens, 0 for padding
        percent: float in (0, 1), e.g., 0.2 for top 20%
        temperature: float, temperature for softmax (default 1.0)
    Returns:
        mask: Tensor of shape (batch, seq), dtype=torch.bool
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)  # (batch, seq)

    valid_entropy = entropy[loss_mask.bool()]
    k = int(percent * valid_entropy.numel())
    if k < 1:
        k = 1
    if k == valid_entropy.numel():
        threshold = valid_entropy.min() - 1  # all True
    else:
        threshold = torch.kthvalue(valid_entropy, valid_entropy.numel() - k + 1).values

    mask = (entropy >= threshold) & (loss_mask.bool())
    return mask




class ImportanceRatioMetrics:
    """
    This class is used to compute the importance ratio metrics

    The importance ratio metrics are computed as follows:
    - error_sum: sum of the importance ratio error. Error is above or below 1
    - raw_error_sum: sum of the raw importance ratio error
    - max: max of the raw importance ratio
    - min: min of the raw importance ratio
    - clipped: clipped percentage of the importance ratio. This is the percentage of tokens that were clipped
    - ratio: ratio of the importance ratio. This is the ratio after clipping
    - raw_ratio: raw ratio of the importance ratio. This is the ratio before clipping
    """

    def __init__(self):
        self.error_sum = torch.tensor(0.0).to("cuda")
        self.raw_error_sum = torch.tensor(0.0).to("cuda")
        self.max = torch.tensor(0.0).to("cuda")
        self.min = torch.tensor(float("inf")).to("cuda")
        self.clipped = torch.tensor(0.0).to("cuda")
        self.ratio = torch.tensor(0.0).to("cuda")
        self.raw_ratio = torch.tensor(0.0).to("cuda")

        self.raw_abs_error_sum = torch.tensor(0.0).to("cuda")

    def update(self, ratio_info: RatioInfo):
        self.error_sum += ratio_info.ratio_sum.detach().float()
        self.raw_error_sum += ratio_info.raw_ratio_sum.detach().float()
        self.raw_abs_error_sum += ratio_info.raw_ratio_abs_sum.detach().float()
        self.max = torch.max(self.max, ratio_info.raw_ratio_max.detach().float())
        self.min = torch.min(self.min, ratio_info.raw_ratio_min.detach().float())
        self.clipped += ratio_info.clipped_token_count.detach().float()

    def sync(self, total_non_masked_tokens: Tensor, loss_scale: float):
        """
        Sync the importance ratio metrics across all ranks.
        """
        self.clipped = self.clipped / loss_scale
        dist.all_reduce(self.clipped, op=dist.ReduceOp.AVG)
        dist.all_reduce(self.max, op=dist.ReduceOp.MAX)
        dist.all_reduce(self.min, op=dist.ReduceOp.MIN)
        dist.all_reduce(self.error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.raw_error_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.raw_abs_error_sum, op=dist.ReduceOp.SUM)

        self.ratio = (total_non_masked_tokens + self.error_sum) / total_non_masked_tokens
        self.raw_ratio = (total_non_masked_tokens + self.raw_error_sum) / total_non_masked_tokens

    def to_dict(self) -> dict[str, float]:
        """
        return a dict of float values (could be used to log to wandb)
        """
        return {
            "importance_ratio/error_sum": self.error_sum.item(),
            "importance_ratio/raw_error_sum": self.raw_error_sum.item(),
            "importance_ratio/max": self.max.item(),
            "importance_ratio/min": self.min.item() if self.min != float("inf") else 0.0,
            "importance_ratio/clipped": self.clipped.item(),
            "importance_ratio/ratio": self.ratio.item(),
            "importance_ratio/raw_ratio": self.raw_ratio.item(),
            "importance_ratio/raw_abs_error_sum": self.raw_abs_error_sum.item(),
        }
