import pytest
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.sft.config import DataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture()
def fake_dataset(tokenizer: PreTrainedTokenizer) -> FakeDataset:
    config = DataConfig(fake="fixed")
    return FakeDataset(tokenizer, config)


@pytest.fixture()
def sft_dataset(tokenizer: PreTrainedTokenizer) -> SFTDataset:
    config = DataConfig(num_examples=2)
    return SFTDataset(tokenizer, config)


def test_init_fake_dataset(fake_dataset: FakeDataset):
    assert fake_dataset is not None


def test_init_sft_dataset(sft_dataset: SFTDataset):
    assert sft_dataset is not None


def test_sft_dataset_state(sft_dataset: SFTDataset):
    dataiter = iter(sft_dataset)
    assert sft_dataset.state_dict() == {"step": -1, "epoch": -1}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 2, "epoch": 1}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 3, "epoch": 1}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 4, "epoch": 2}
    next(dataiter)
    assert sft_dataset.state_dict() == {"step": 5, "epoch": 2}


def test_sft_dataset_state_resume(sft_dataset: SFTDataset):
    assert sft_dataset.state_dict() == {"step": -1, "epoch": -1}
    sft_dataset.load_state_dict({"step": 876, "epoch": 3})
    assert sft_dataset.state_dict() == {"step": 876, "epoch": 3}
