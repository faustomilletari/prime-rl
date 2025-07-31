from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from prime_rl.trainer.config import TrainerConfig


def create_lr_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LRScheduler | None:
    """Create learning rate scheduler based on config."""
    if config.optim.scheduler == "constant":
        return None

    warmup_steps = config.optim.n_warmup_steps
    n_final_decay = config.optim.n_final_decay

    # If n_final_decay is not specified, fall back to using max_steps for total schedule length
    if n_final_decay is None:
        if config.max_steps is None:
            raise ValueError("Must specify either n_final_decay or max_steps for non-constant scheduler")
        # Use max_steps as total length, subtract warmup to get decay portion
        n_final_decay = config.max_steps - warmup_steps

    if n_final_decay <= 0:
        raise ValueError(f"n_final_decay must be positive, got {n_final_decay} (warmup_steps={warmup_steps})")

    # Create the main decay scheduler
    if config.optim.scheduler == "linear":
        main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=n_final_decay)
    elif config.optim.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(optimizer, T_max=n_final_decay, eta_min=0.0)
    else:
        raise ValueError(f"Unknown scheduler type: {config.optim.scheduler}")

    # If no warmup, return the main scheduler directly
    if warmup_steps == 0:
        return main_scheduler

    # If warmup, combine warmup + main scheduler
    # LinearLR requires start_factor > 0, so we use a very small value instead of 0
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])
