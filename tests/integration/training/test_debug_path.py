from pathlib import Path
from typing import Callable

import pytest

from tests import Command, Environment, ProcessResult

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

CMD = ["uv", "run", "train", "@configs/training/debug.toml"]


@pytest.fixture(scope="module")
def output_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("test_rollout_run")


@pytest.fixture(scope="module")
def process(
    run_process: Callable[[Command, Environment], ProcessResult],
    fake_rollout_dir: Callable[[list[int], int, int], Path],
):
    rollout_path = fake_rollout_dir(steps=list(range(1, 6)), batch_size=16, micro_batch_size=8, seq_len=16)
    return run_process(CMD + ["--data.path", rollout_path.as_posix(), "--data.fake", "None"], {})


def test_no_error(process: ProcessResult):
    assert process.returncode == 0, f"Process failed with return code {process.returncode}"
