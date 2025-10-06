import threading
import time
import socket
from pathlib import Path
from typing import Callable

import pytest
import torch

from prime_rl.orchestrator.batch import BatchSample
from prime_rl.trainer.rl.data import MicroBatch
from prime_rl.utils.variable_store import VariableStoreServer
from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

ENV = {"CUDA_VISIBLE_DEVICES": "1"}
CMD = ["uv", "run", "trainer", "@", "configs/debug/rl/train.toml"]


def get_free_port() -> int:
    """Find and return a free port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def create_sample(seq_len: int) -> BatchSample:
    return {
        "input_ids": torch.randint(0, 100, (seq_len,)).long(),
        "position_ids": torch.zeros(seq_len).long(),
        "advantages": torch.randn(seq_len).float(),
        "loss_mask": torch.ones(seq_len).bool(),
        "logprobs": torch.randn(seq_len).float(),
        "total_tokens": seq_len,
    }


def create_dummy_batch(batch_size: int, seq_len: int) -> MicroBatch:
    micro_batch = {}
    samples = [create_sample(seq_len) for _ in range(batch_size)]
    for key in ["input_ids", "advantages", "loss_mask", "logprobs", "position_ids"]:
        micro_batch[key] = torch.cat([sample[key] for sample in samples]).unsqueeze(0)
    micro_batch["temperature"] = 1.0
    micro_batch["total_tokens"] = batch_size * seq_len
    return micro_batch


@pytest.fixture(scope="module")
def variable_store_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[VariableStoreServer, Path, int]:
    """Create a variable store server with dummy batches."""
    output_dir = tmp_path_factory.mktemp("outputs")
    
    # Get a free port to avoid conflicts
    port = get_free_port()
    
    # Start variable store server
    server = VariableStoreServer(host="localhost", port=port, steps_to_preserve=2)
    server.start()
    
    # Give server time to start
    time.sleep(0.1)
    
    yield server, output_dir, port
    
    # Clean up server
    server.stop()


@pytest.fixture(scope="module")
def train_process(
    run_process: Callable[[Command, Environment], ProcessResult],
    variable_store_server: tuple[VariableStoreServer, Path, int],
):
    server, output_dir, port = variable_store_server
    
    # Populate variable store with dummy batches
    steps = list(range(5))
    batch_size = 16
    micro_batch_size = 8
    seq_len = 16
    
    for step in steps:
        batches = []
        assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"
        for _ in range(batch_size // micro_batch_size):
            micro_batch = create_dummy_batch(micro_batch_size, seq_len)
            batches.append(micro_batch)
        
        # Store batches for rank 0
        key = f"step_{step}_rank_0"
        server.put(key, batches)
    
    # Run trainer with variable store configuration
    return run_process(
        CMD + [
            "--output-dir", output_dir.as_posix(), 
            "--data.fake", "None", 
            "--log.level", "debug",
            "--variable-store.host", "localhost",
            "--variable-store.port", str(port),
        ], 
        ENV
    )


def test_no_error(train_process: ProcessResult):
    assert train_process.returncode == 0, f"Train process failed with return code {train_process.returncode}"
