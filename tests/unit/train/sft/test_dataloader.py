import os

import pytest
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import FakeDataConfig
from prime_rl.trainer.sft.data import setup_dataloader, setup_dataset
from prime_rl.trainer.world import reset_world

pytestmark = [pytest.mark.gpu]


def test_stateful_dataloader_single_rank():
    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=2, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1
    assert dataloader.state_dict()["dataset_state"] == {"step": 1, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 2
    assert dataloader.state_dict()["dataset_state"] == {"step": 2, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3
    assert dataloader.state_dict()["dataset_state"] == {"step": 3, "epoch": 1}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 4
    assert dataloader.state_dict()["dataset_state"] == {"step": 4, "epoch": 1}


@pytest.mark.parametrize("rank", [0, 1], ids=["rank0", "rank1"])
def test_stateful_dataloader_multi_rank(rank: int):
    # Setup world
    reset_world()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(2)

    # Setup stateful dataloader
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 1 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 1 + rank, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 3 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 3 + rank, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 5 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 5 + rank, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 7 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 7 + rank, "epoch": 0}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 9 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 9 + rank, "epoch": 1}
    micro_batch = next(dataiter)
    assert micro_batch["input_ids"].unique().item() == 11 + rank
    assert dataloader.state_dict()["dataset_state"] == {"step": 11 + rank, "epoch": 1}


def test_stateful_dataloader_resume():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    # First epoch
    for i in range(num_examples):
        step = i + 1
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"step": step, "epoch": 0}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second epoch
    for i in range(num_examples):
        step = i + num_examples + 1
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert micro_batch["epoch"] == 1
        assert dataloader.state_dict()["dataset_state"] == {"step": step, "epoch": 1}


def test_stateful_dataloader_resume_mid_epoch():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = FakeDataConfig(length="fixed", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    # First 1/2 epoch
    for i in range(num_examples // 2):
        step = i + 1
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"step": step, "epoch": 0}

    # Reload dataloader
    state_dict = dataloader.state_dict()
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataloader.load_state_dict(state_dict)
    dataiter = iter(dataloader)

    # Second epoch
    for i in range(num_examples // 2):
        step = i + num_examples // 2 + 1
        micro_batch = next(dataiter)
        assert micro_batch["input_ids"].shape == (1, 128)
        assert micro_batch["input_ids"].unique().item() == step
        assert micro_batch["epoch"] == 0
        assert dataloader.state_dict()["dataset_state"] == {"step": step, "epoch": 0}


def test_dataloader_ckpt_with_packing():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    num_examples = 8
    config = FakeDataConfig(length="variable", input_ids="increasing", num_examples=8, batch_size=1, micro_batch_size=1)
    dataset = setup_dataset(tokenizer, config)
    dataloader = setup_dataloader(dataset, tokenizer, config)
    dataiter = iter(dataloader)

    step = 0
    for _ in range(num_examples):
        micro_batch = next(dataiter)
        num_packed_examples = len(micro_batch["input_ids"].unique())
        step += num_packed_examples
        epoch = (step - 1) // num_examples  # -1 because for modulo to work we need zero based counting
        assert micro_batch["input_ids"].shape == (1, 128)
        assert dataloader.state_dict()["dataset_state"] == {"step": step, "epoch": epoch}
