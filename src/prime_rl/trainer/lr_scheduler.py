from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import TrainerConfig


def create_lr_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LRScheduler | None:
    """Create learning rate scheduler based on config."""
    if config.optim.scheduler == "constant":
        return None

    warmup_steps = config.optim.n_warmup_steps
    n_final_decay = config.optim.n_final_decay

    if config.max_steps is None:
        raise ValueError("Must specify max_steps when using a scheduler")

    if n_final_decay is None:
        # Fallback: decay for remaining steps after warmup
        n_final_decay = config.max_steps - warmup_steps

    if n_final_decay <= 0:
        raise ValueError(f"n_final_decay must be positive, got {n_final_decay}")

    if warmup_steps + n_final_decay > config.max_steps:
        raise ValueError(
            f"Warmup steps ({warmup_steps}) + final decay steps ({n_final_decay}) exceeds max_steps ({config.max_steps})"
        )

    # Calculate when final decay starts
    decay_start_step = config.max_steps - n_final_decay
    constant_steps = decay_start_step - warmup_steps

    # Create schedulers for each phase
    schedulers = []
    milestones = []

    # Phase 1: Warmup (if any)
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        schedulers.append(warmup_scheduler)
        milestones.append(warmup_steps)

    # Phase 2: Constant (if any)
    if constant_steps > 0:
        constant_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=constant_steps)
        schedulers.append(constant_scheduler)
        milestones.append(decay_start_step)

    # Phase 3: Final decay
    if config.optim.scheduler == "linear":
        decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=n_final_decay)
    elif config.optim.scheduler == "cosine":
        decay_scheduler = CosineAnnealingLR(optimizer, T_max=n_final_decay, eta_min=0.0)
    else:
        raise ValueError(f"Unknown scheduler type: {config.optim.scheduler}")

    schedulers.append(decay_scheduler)

    # Return single scheduler if only one phase, otherwise combine with SequentialLR
    if len(schedulers) == 1:
        return schedulers[0]

    return SequentialLR(optimizer, schedulers, milestones=milestones)
