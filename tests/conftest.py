import concurrent.futures
import os
import subprocess
from typing import Callable

import pytest
from huggingface_hub import HfApi
from loguru import logger
from openai import AsyncOpenAI

from zeroband.training.config import AttnImplementation
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
    """Factory fixture for running multiple processes in parallel. Used for parallel RL tests"""
    return run_subprocesses_in_parallel
