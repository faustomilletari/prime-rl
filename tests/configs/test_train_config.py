import pytest
import tomli

from zeroband.train import Config as TrainingConfig

from utils import get_all_toml_files


@pytest.mark.parametrize("config_file_path", get_all_toml_files("configs/training"))
def test_load_config(config_file_path):
    with open(f"{config_file_path}", "rb") as f:
        content = tomli.load(f)
    config = TrainingConfig(**content)
    assert config is not None
