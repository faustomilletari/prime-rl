import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns

from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature, get_optimizer_signature
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class AppState(Stateful):
    """
    A wrapper for checkpointing the trainer with sharded weights and optimizer
    to allow resuming in any world size using torch.distributed.checkpoint
    utilities.
    """

    def __init__(
        self,
        model: Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
    ):
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.progress = progress

    def state_dict(self) -> dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizers)
        scheduler_state_dict = self.scheduler.state_dict()
        progress_state_dict = asdict(self.progress)
        state_dict = {
            "model": model_state_dict,
            "optimizers": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "progress": progress_state_dict,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.model, self.optimizers, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizers"]
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        for key, value in state_dict["progress"].items():
            setattr(self.progress, key, value)


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig, save_hf: bool = True):
        self.config = config
        self.save_hf = save_hf  # NEW: option to save HF format
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.is_master
        self.ckpt_steps: list[int] = []

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"
    
    def get_hf_ckpt_path(self, step: int) -> Path:
        """Get path for HuggingFace checkpoint."""
        return self.ckpt_dir / f"step_{step}" / "hf"

    def get_latest_step(self) -> int:
        step_dirs = list(self.ckpt_dir.glob("step_*"))
        if len(step_dirs) == 0:
            raise ValueError(f"No checkpoints found in {self.ckpt_dir}")
        steps = sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])
        latest_step = steps[-1]
        self._logger.info(f"Found latest checkpoint in {self.ckpt_dir}: {latest_step}")
        return latest_step

    def _gather_hf_weights(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
        """Gather distributed weights into a consolidated HuggingFace-compatible state dict."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            cpu_state = {}
            for key, value in model.state_dict().items():
                if isinstance(value, DTensor):
                    value = value.to(dtype)
                    value = value.full_tensor()

                if self._is_master:
                    key = get_fqns(model, key)
                    assert len(key) == 1
                    key = next(iter(key))
                    cpu_state[key] = value.to("cpu", non_blocking=False)

            torch.distributed.barrier()

        return cpu_state

    def _save_hf_checkpoint(self, model: nn.Module, ckpt_step: int):
        """Save HuggingFace-compatible checkpoint (master rank only)."""
        if not self._is_master:
            torch.distributed.barrier()
            return

        hf_path = self.get_hf_ckpt_path(ckpt_step)
        hf_path.mkdir(parents=True, exist_ok=True)

        self._logger.debug(f"Saving HuggingFace checkpoint to {hf_path}")
        start_time = time.time()

        # Gather weights from all ranks
        cpu_state = self._gather_hf_weights(model, dtype=torch.bfloat16)

        # Save using HuggingFace's save_pretrained
        try:
            # model.config should exist for HF models
            model.config.save_pretrained(hf_path)
            if hasattr(model, 'generation_config') and model.generation_config:
                model.generation_config.save_pretrained(hf_path)
        except Exception as e:
            self._logger.warning(f"Could not save model config: {e}")

        # Save state dict as safetensors
        from safetensors.torch import save_file
        save_file(cpu_state, hf_path / "model.safetensors", metadata={"format": "pt"})

        # Mark as complete
        (hf_path / "STABLE").touch()

        self._logger.debug(f"HuggingFace checkpoint saved in {time.time() - start_time:.2f} seconds")
        torch.distributed.barrier()

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ):
        self._logger.debug(f"Saving training checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        state_dict = {"app": AppState(model, optimizers, scheduler, progress)}

        # Checkpoint the local dataloader
        if dataloader is not None:
            dataloader_dir = ckpt_path / "dataloader"
            dataloader_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self._world.rank}.pt")

        # Save sharded state
        dcp.save(state_dict, checkpoint_id=ckpt_path)

        # NEW: Also save HuggingFace checkpoint
        if self.save_hf:
            self._save_hf_checkpoint(model, ckpt_step)

        # Append to list of saved steps
        if self._is_master:
            self.ckpt_steps.append(ckpt_step)

        self._logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(
        self,
        ckpt_path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading training checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load sharded state
        app_state = AppState(model, optimizers, scheduler, progress)
        state_dict = {"app": app_state}
        dcp.load(state_dict=state_dict, checkpoint_id=ckpt_path)

        # Load the dataloader
        if self.config.skip_dataloader:
            get_logger().warning("Skipping dataloader checkpointing")

        if dataloader is not None and not self.config.skip_dataloader:
            dataloader_path = ckpt_path / "dataloader" / f"rank_{self._world.rank}.pt"
            if not dataloader_path.exists():
                self._logger.warning(
                    f"Did not find local dataloader checkpoint at path {dataloader_path}. This might be because you tried restarting the trainer with a different world size. Falling back to using the master rank's dataloader checkpoint. Note, that this may cause training inconsistencies."
                )
                dataloader_path = ckpt_path / "dataloader" / "rank_0.pt"
                if not dataloader_path.exists():
                    raise RuntimeError(
                        f"Couldn't fallback to using the master rank's dataloader checkpoint, because dataloder checkpoint was not found at path {dataloader_path}. Cannot resume training."
                    )
            dataloader.load_state_dict(torch.load(dataloader_path))

        self._logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Loads a checkpoint from a given path in-place."""
        if step == -1:
            step = self.get_latest_step()

        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        self._logger.debug(
            f"Signatures after loading training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )

    def save(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            f"Signatures before saving training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )
        self._save_to_path(ckpt_path, step, model, optimizers, scheduler, progress, dataloader)

    def maybe_clean(self) -> None:
        """Deletes past local checkpoints beyond the most recent `config.keep` steps. No-op if `config.keep` is None."""
        if self.config.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_delete = self.ckpt_steps[: -self.config.keep]
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self.get_ckpt_path(ckpt_step)
            if ckpt_path.exists():
                self._logger.debug(f"Removing past trainer checkpoint for step {ckpt_step} ({ckpt_path})")
                shutil.rmtree(ckpt_path)
            
            # NEW: Also clean HF checkpoints
            if self.save_hf:
                hf_path = self.get_hf_ckpt_path(ckpt_step)
                if hf_path.exists():
                    self._logger.debug(f"Removing past HF checkpoint for step {ckpt_step} ({hf_path})")
                    shutil.rmtree(hf_path)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.config.keep :]


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None, save_hf: bool = True) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config, save_hf=save_hf)