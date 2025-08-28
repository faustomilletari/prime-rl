from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig, RealDataConfig
from prime_rl.trainer.sft.data import FakeDataset, SFTDataset


def test_init_fake_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed")
    fake_dataset = FakeDataset(tokenizer, config)
    assert fake_dataset is not None


def test_fake_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", num_examples=2)
    dataset = FakeDataset(tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 1}


def test_init_sft_dataset():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = RealDataConfig(num_examples=2)
    dataset = SFTDataset(tokenizer, config)
    assert dataset is not None


def test_sft_dataset_state():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = RealDataConfig(num_examples=2)
    dataset = SFTDataset(tokenizer, config)
    dataiter = iter(dataset)
    assert dataset.state_dict() == {"step": 0, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 1, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 2, "epoch": 0}
    next(dataiter)
    assert dataset.state_dict() == {"step": 3, "epoch": 1}
