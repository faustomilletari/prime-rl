from typing import TypedDict

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from jaxtyping import Bool, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from prime_rl.trainer.sft.config import DataConfig
from prime_rl.utils.logger import get_logger


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[int]


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]


class FakeDataset(Dataset):
    """A dataset of fake tokens"""

    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.config = config
        self.vocab_size = tokenizer.vocab_size

    def __len__(self) -> int:
        return self.config.fake.n // self.config.batch_size * self.config.batch_size  # We drop the last batch

    def __getitem__(self, index: int) -> Sample:
        input_ids = torch.randint(0, self.vocab_size, (self.config.seq_len + 1,)).long()
        position_ids = torch.arange(len(input_ids)).long()
        loss_mask = torch.ones(len(input_ids)).bool()
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class SFTDataset(Dataset):
    """A dataset wrapping a HF SFT dataset with prompt + completion format."""

    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        assert not config.fake, "HFDataset does not support fake data"
        self.config = config
        self.tokenizer = tokenizer
        self._logger = get_logger()

        # Load dataset
        self.dataset: HFDataset = load_dataset(config.name, split=config.split)

        # Assert that the dataset has a 'text' column
        if "prompt" not in self.dataset.column_names or "completion" not in self.dataset.column_names:
            raise ValueError("HF dataset must have a 'prompt' and 'completion' column for SFT")

        # Preprocess dataset (tokenize, truncate and pad)
        self.samples = self.dataset.map(self._preprocess, with_indices=True).to_list()
        self.index = 0

    def _preprocess(self, example: dict, index: int) -> Sample:
        """
        Tokenize, truncate and pad a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
        """
        assert "prompt" in example and "completion" in example, "Prompt and completion must be present in the example"
        assert isinstance(example["prompt"], list) and isinstance(example["completion"], list), (
            "Prompt and completion must be lists"
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            example["prompt"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        prompt_completion_ids = self.tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )

        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
            self._logger.warning(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                "token handling. Verify that the tokenizer is processing text consistently."
            )

        # Create sample
        sample = {
            "input_ids": prompt_completion_ids,
            "position_ids": list(range(len(prompt_completion_ids))),
            "loss_mask": [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids)),
        }

        # Truncate and pad sample
        seq_len = self.config.seq_len + 1  # Because we shift and lose one token
        if len(sample["input_ids"]) > seq_len:  # Truncate
            self._logger.warning(f"Truncated sample {index} from {len(sample['input_ids'])} to {seq_len} tokens")
            sample["input_ids"] = sample["input_ids"][:seq_len]
            sample["loss_mask"] = sample["loss_mask"][:seq_len]
            sample["position_ids"] = sample["position_ids"][:seq_len]
        if len(sample["input_ids"]) < seq_len:  # Pad
            num_pad_tokens = seq_len - len(sample["input_ids"])
            sample["input_ids"] += [self.tokenizer.pad_token_id] * num_pad_tokens
            sample["loss_mask"] += [0] * num_pad_tokens
            sample["position_ids"] += [0] * num_pad_tokens

        return sample

    def __len__(self) -> int:
        return len(self.samples) // self.config.batch_size * self.config.batch_size  # We drop the last batch

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def get_dataset(tokenizer, config: DataConfig) -> Dataset:
    """Returns the PyTorch dataset to train on."""
    if config.fake:
        return FakeDataset(tokenizer, config)
    return SFTDataset(tokenizer, config)


def get_dataloader(dataset: Dataset, config: DataConfig) -> DataLoader:
    def collate_fn(batch: list[Sample]) -> Batch:
        batch_input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch]).long()
        batch_position_ids = torch.stack([torch.tensor(item["position_ids"]) for item in batch]).long()
        batch_loss_mask = torch.stack([torch.tensor(item["loss_mask"]) for item in batch]).bool()

        return {
            "input_ids": batch_input_ids[:, :-1].contiguous(),
            "target_ids": batch_input_ids[:, 1:].contiguous(),
            "position_ids": batch_position_ids[:, :-1].contiguous(),
            "loss_mask": batch_loss_mask[:, :-1].contiguous(),
        }

    # Initialize rank-aware sampler
    sampler = DistributedSampler(dataset, shuffle=config.shuffle, drop_last=True)

    return iter(DataLoader(dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn, sampler=sampler))
