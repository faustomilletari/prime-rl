from typing import TypedDict

import torch
from datasets import load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import DataConfig


class Sample(TypedDict):
    input_ids: Int[Tensor, "seq"]
    position_ids: Int[Tensor, "seq"]
    loss_mask: Bool[Tensor, "seq"]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]


class FakeDataset(IterableDataset):
    """An infinite dataset of fake tokens"""

    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.config = config
        self.vocab_size = tokenizer.vocab_size

    def fake_sample(self) -> Sample:
        input_ids = torch.randint(0, self.vocab_size, (self.config.seq_len + 1,)).long()
        position_ids = torch.arange(len(input_ids)).long()
        loss_mask = torch.ones(len(input_ids)).bool()
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __iter__(self):
        while True:
            yield self.fake_sample()


class HFDataset(Dataset):
    """Standard PyTorch dataset which wraps a HF dataset."""

    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset: Dataset = load_dataset(config.path, split=config.split)

        # Assert that the dataset has a 'text' column
        if "text" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'text' column for SFT")

        # Tokenize dataset
        self.samples = self.dataset.map(self._tokenize, input_columns=["text"], batched=True).to_list()

    def _tokenize(self, text: str):
        return self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.config.seq_len, return_tensors="pt"
        )

    def __getitem__(self, index: int) -> Sample:
        input_ids = self.samples[index]["input_ids"]
        position_ids = torch.arange(len(input_ids)).long()
        loss_mask = torch.ones(len(input_ids)).bool()

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def __len__(self) -> int:
        return len(self.samples)


def get_dataset(tokenizer, config: DataConfig) -> Dataset:
    """Returns the PyTorch dataset to train on."""
    if config.fake:
        return FakeDataset(tokenizer, config)
    return HFDataset(tokenizer, config)


def get_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    def collate_fn(batch: list[Sample]) -> Batch:
        batch_input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch], dim=0)
        batch_position_ids = torch.stack([torch.tensor(item["position_ids"]) for item in batch], dim=0)
        batch_loss_mask = torch.stack([torch.tensor(item["loss_mask"]) for item in batch], dim=0)

        return {
            "input_ids": batch_input_ids[:, :-1].contiguous(),
            "target_ids": batch_input_ids[:, 1:].contiguous(),
            "position_ids": batch_position_ids[:, :-1].contiguous(),
            "loss_mask": batch_loss_mask[:, :-1].contiguous(),
        }

    return iter(DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn))
