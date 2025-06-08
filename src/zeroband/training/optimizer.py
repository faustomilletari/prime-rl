import torch
from muon_fsdp2 import Muon

from zeroband.training.config import AdamConfig, MuonConfig, OptimizerConfig
from zeroband.utils.models import ModelType


def setup_optimizer(config: OptimizerConfig, model: ModelType) -> torch.optim.Optimizer:
    if isinstance(config, AdamConfig):
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.wd,
            betas=(config.betas1, config.betas2),
        )
        return optimizer

    elif isinstance(config, MuonConfig):
        hidden_matrix_params = []
        rest_params = []

        for n, p in model.named_parameters():
            if p.ndim >= 2 and "embed" not in n:
                hidden_matrix_params.append(p)
            else:
                rest_params.append(p)

        adam_groups = [
            dict(params=rest_params, lr=5e-6, use_muon=False, betas=(0.8, 0.95), eps=1e-10),
        ]

        # Muon parameter group
        muon_group = dict(
            params=hidden_matrix_params,
            lr=config.lr,
            momentum=config.momentum,
            ns_steps=config.ns_steps,
            wd=config.wd,
            use_muon=True,
        )

        # Single unified optimizer
        optimizer = Muon([*adam_groups, muon_group])
        return optimizer
    else:
        raise ValueError(f"Invalid optimizer config: {config}")
