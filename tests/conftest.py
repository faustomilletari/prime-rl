import concurrent.futures
import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from loguru import logger

from zeroband.training.config import AttnImplementation
from zeroband.training.data import MicroBatch
from zeroband.training.orchestrator.data import Sample
from zeroband.training.world import reset_world
from zeroband.utils.logger import reset_logger, set_logger

TIMEOUT = 120


Environment = dict[str, str]
Command = list[str]


@pytest.fixture(autouse=True)
def setup_logger():
    """
    Fixture to set and reset the logger after each test.
    """
    set_logger(logger)  # Use the default loguru.logger
    yield
    reset_logger()


@pytest.fixture(autouse=True)
def setup_env():
    """
    Fixture to reset environment variables after each test.
    """
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def setup_world():
    """
    Fixture to reset the world info after each test.
    """
    yield
    reset_world()


@pytest.fixture(params=["eager", "sdpa", "flash_attention_2"])
def attn_impl(request) -> AttnImplementation:
    """
    Fixture to test different attention implementations.
    """
    try:
        # ruff: noqa: F401
        import flash_attn
    except ImportError:
        pytest.skip("Flash Attention not available")
    return request.param


@pytest.fixture(scope="session")
def model_name() -> str:
    """Main model to use for tests."""
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def hf_api() -> HfApi:
    """Hugging Face API to use for tests."""
    return HfApi()


@pytest.fixture(scope="module")
def llm(model_name: str):
    """
    vLLM LLM instance to use for tests. Incurs significant startup time, hence reused across tests.
    """
    from vllm import LLM

    yield LLM(
        model=model_name,
        enforce_eager=True,
        disable_async_output_proc=True,
        dtype="bfloat16",
    )

    if dist.is_initialized():
        dist.destroy_process_group()


def create_sample(seq_len: int) -> Sample:
    return {
        "input_ids": torch.randint(0, 100, (seq_len,)),
        "advantages": torch.randn(seq_len),
        "loss_mask": torch.ones(seq_len),
        "position_ids": torch.zeros(seq_len),
        "logprobs": torch.randn(seq_len - 1),
        "total_tokens": seq_len,
    }


def create_dummy_batch(batch_size: int, seq_len: int) -> MicroBatch:
    micro_batch = {}
    samples = [create_sample(seq_len) for _ in range(batch_size)]
    for key in ["input_ids", "advantages", "loss_mask", "logprobs", "position_ids"]:
        micro_batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    micro_batch["temperature"] = 1.0
    micro_batch["total_tokens"] = batch_size * seq_len
    return micro_batch


@pytest.fixture(scope="module")
def fake_rollout_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[list[int], int, int, int], Path]:
    """Create a temporary directory with dummy batches."""
    path = tmp_path_factory.mktemp("fake-rollouts")

    def write_dummy_batches(
        steps: list[int] = [1],
        batch_size: int = 1,
        micro_batch_size: int = 1,
        seq_len: int = 10,
    ) -> Path:
        for step in steps:
            step_path = path / f"step_{step}"
            step_path.mkdir(parents=True, exist_ok=True)
            batch_path = step_path / "rank_0.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            batches = []
            assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"
            for _ in range(batch_size // micro_batch_size):
                micro_batch = create_dummy_batch(micro_batch_size, seq_len)
                batches.append(micro_batch)
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        return path

    return write_dummy_batches


class ProcessResult:
    def __init__(self, returncode: int, pid: int):
        self.returncode = returncode
        self.pid = pid


def run_subprocess(command: Command, env: Environment, timeout: int = TIMEOUT) -> ProcessResult:
    """Run a subprocess with given command and environment with a timeout"""
    try:
        process = subprocess.Popen(command, env={**os.environ, **env})
        process.wait(timeout=timeout)
        return ProcessResult(process.returncode, process.pid)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)  # Give it 10 seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    except Exception as e:
        raise e


def run_subprocesses_in_parallel(
    commands: list[Command], envs: list[Environment], timeout: int = TIMEOUT
) -> list[ProcessResult]:
    """Start multiple processes in parallel using ProcessPoolExecutor and wait for completion."""
    assert len(commands) == len(envs), "Should have an environment for each command"
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(run_subprocess, cmd, env, timeout) for cmd, env in zip(commands, envs)]
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Process {i} did not complete within {timeout} seconds")

    return results


@pytest.fixture(scope="module")
def run_process() -> Callable[[Command, Environment], ProcessResult]:
    """Factory fixture for running a single process."""
    return run_subprocess


@pytest.fixture(scope="module")
def run_processes() -> Callable[[list[Command], list[Environment]], list[ProcessResult]]:
    """Factory fixture for running multiple processes in parallel. Used for parallel inference tests and RL training tests."""
    return run_subprocesses_in_parallel
