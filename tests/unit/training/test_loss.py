import pytest
import torch

from prime_rl.trainer.rl.config import LossConfig
from prime_rl.trainer.rl.loss import compute_entropy, compute_loss

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    logprobs = torch.randn(100, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(100, dtype=torch.float32).cuda()
    advantages = torch.randn(100).cuda()

    loss, _ = compute_loss(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask=torch.ones(100, dtype=torch.bool).cuda(),
        position_ids=torch.arange(100, dtype=torch.int32).cuda(),
        loss_config=LossConfig(type="grpo", clip_ratio_low=0.0, clip_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == (100,)


def test_gspo_loss():
    logprobs = torch.randn(100, dtype=torch.float32).cuda()
    old_logprobs = torch.randn(100, dtype=torch.float32).cuda()
    advantages = torch.randn(100).cuda()

    loss, _ = compute_loss(
        logprobs,
        old_logprobs,
        advantages,
        loss_mask=torch.ones(100, dtype=torch.bool).cuda(),
        position_ids=torch.arange(100, dtype=torch.int32).cuda(),
        loss_config=LossConfig(type="gspo", clip_ratio_low=0.0, clip_ratio_high=10.0),
        loss_scale=1.0,
    )
    assert loss.shape == (100,)


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)
