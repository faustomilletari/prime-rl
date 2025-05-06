"""
Tests all of the config file. usefull to catch mismatch key after a renaming of a arg name
Need to be run from the root folder
"""

from zeroband.infer import Config as InferenceConfig
import torch

from pydantic import ValidationError
import pytest
import tomli

from utils import get_all_toml_files


@pytest.mark.parametrize("config_file_path", get_all_toml_files("configs/inference"))
def test_load_inference_config(config_file_path):
    with open(f"{config_file_path}", "rb") as f:
        content = tomli.load(f)
    config = InferenceConfig(**content)
    assert config is not None


def test_throw_error_for_dp_and_pp():
    with pytest.raises(ValidationError):
        config = InferenceConfig(**{"dp": 2, "pp": {"world_size": 2}})
        print(config)


@pytest.mark.parametrize("tp", ["auto", 1, 2])
@pytest.mark.parametrize("pp_rank, pp_world_size", [(0, 2), (1, 2)])
def test_tp_and_pp(tp: int | str, pp_rank: int, pp_world_size: int):
    if pp_rank >= pp_world_size:
        pytest.skip("pp_rank must be less than pp_world_size")
    config = InferenceConfig(**{"tp": tp, "pp": {"rank": pp_rank, "world_size": pp_world_size}})
    if tp == "auto":
        tp = torch.cuda.device_count()
    assert config.tp == tp
    assert config.pp.rank == pp_rank
    assert config.pp.world_size == pp_world_size
    assert config.rank == pp_rank
    assert config.world_size == tp * config.pp.world_size
    assert config.local_world_size == tp
    assert config.local_rank == 0


@pytest.mark.parametrize("tp", ["auto", 1, 2])
def test_tp_and_dp_ok(tp: int | str):
    config = InferenceConfig(**{"tp": tp, "dp": 2})
    if tp == "auto":
        tp = torch.cuda.device_count() // 2
    assert config.tp == tp
    assert config.dp == 2
    assert config.world_size == tp * 2
    assert config.local_world_size == tp
    assert config.local_rank == 0
