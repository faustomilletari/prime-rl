import time
from pathlib import Path
from typing import TypedDict
import os
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.trainer.rl.config import DataLoaderConfig, FakeDataLoaderConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_rollout_dir, wait_for_path
from prime_rl.utils.zmq_store import SyncDataStoreClient, wait_for_data_sync
from prime_rl.utils.logger import get_logger


class MicroBatch(TypedDict):
    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    logprobs: Float[Tensor, "batch seq"]
    temperature: float
    total_tokens: int


class FakeDataLoader:
    def __init__(self, config: FakeDataLoaderConfig):
        self.batch_size = config.batch_size
        self.micro_batch_size = config.micro_batch_size
        self.num_micro_batches = self.batch_size // self.micro_batch_size // get_world().world_size
        self.seq_len = config.seq_len

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[MicroBatch]:
        return [self._get_micro_batch() for _ in range(self.num_micro_batches)]

    def _get_micro_batch(self) -> MicroBatch:
        return {
            "input_ids": torch.randint(0, 100, (self.micro_batch_size, self.seq_len)),
            "position_ids": torch.stack([torch.arange(self.seq_len)] * self.micro_batch_size, dim=0),
            "advantages": torch.randn(self.micro_batch_size, self.seq_len),
            "logprobs": torch.randn(self.micro_batch_size, self.seq_len),
            "temperature": 1.0,
            "loss_mask": torch.ones(self.micro_batch_size, self.seq_len, dtype=torch.bool),
            "total_tokens": self.micro_batch_size * self.seq_len,
        }


class DataLoader:
    """Data loader for RL training that loads rollouts from disk or ZeroMQ store."""

    def __init__(self, output_dir: Path, step: int, zmq_config=None):
        self.output_dir = output_dir
        self.step = step
        self.zmq_config = zmq_config
        self.world = get_world()
        self.logger = get_logger()
        
        # Initialize ZeroMQ client if enabled
        self.zmq_client = None
        if zmq_config and zmq_config.enabled:
            self.zmq_client = SyncDataStoreClient(
                server_address=zmq_config.client_connect_address,
                server_port=zmq_config.port
            )
        else:
            self._logger.info("DataLoader using file system")

    def get_rollout_path(self) -> Path:
        """Get rollout path for file system approach."""
        return self.rollout_dir / f"step_{self.step}" / f"rank_{self.world.rank}.pt"

    def get_rollout_key(self) -> str:
        """Get rollout key for ZeroMQ approach."""
        return f"step_{self.step}_rank_{self.world.rank}"

    def wait_for_batch(self):
        """Wait for the batch to be available."""
        if self.zmq_client:
            # Wait for all rank batches to be available
            for rank in range(self.world.world_size):
                rollout_key = f"step_{self.step}_rank_{rank}"
                self.logger.debug(f"Waiting for rollout {rollout_key}")
                wait_for_data_sync(self.zmq_client, rollout_key)
        else:
            rollout_path = self.get_rollout_path()
            self._logger.debug(f"Waiting for rollout file {rollout_path}")
            wait_for_path(rollout_path)

    def get_batch(self):
        """Load the batch for the current step."""
        if self.zmq_client:
            # Load from ZeroMQ store
            rollout_key = self.get_rollout_key()
            self.logger.debug(f"Loading rollout {rollout_key} from ZeroMQ store")
            batches = self.zmq_client.retrieve_data(rollout_key)
            if batches is None:
                raise RuntimeError(f"Failed to retrieve rollout {rollout_key}")
        else:
            rollout_path = self.get_rollout_path()
            self._logger.debug(f"Loading rollout from file {rollout_path}")
            batches = torch.load(rollout_path)
        
        self.current_step += 1
        return batches

    def delete_rollout(self, rollout_key: str):
        """Delete rollout from ZeroMQ store."""
        if self.zmq_client:
            self.logger.debug(f"Deleting rollout {rollout_key} from ZeroMQ store")
            self.zmq_client.delete_data(rollout_key)
        else:
            os.rmtree(self.rollout_dir / f"step_{rollout_key}")
            

    def close(self):
        """Close the data loader."""
        if self.zmq_client:
            self.zmq_client.close()
