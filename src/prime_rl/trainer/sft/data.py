from typing import TypedDict
from functools import partial

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
        return self.config.fake.n

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

        # Preprocess dataset (apply chat template and tokenize)
        self.samples = self.dataset.map(self._preprocess).to_list()

        # Optionally, pack samples
        if config.collate_mode == "packing":
            self.samples = self._pack_samples(self.samples)

    def _preprocess(self, example: dict) -> Sample:
        """
        Apply chat template and tokenize a single example in prompt + completion format (https://github.com/huggingface/trl/blob/de27d612b026526ba39b88eee348994d7636e033/trl/trainer/sft_trainer.py#L661)
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

        return sample

    def _pack_samples(self, samples: list[Sample]) -> list[Sample]:
        """Offline sample packing using `First Fit Decreasing` algorithm."""
        # Sort samples in reverse order of length
        sorted_samples = sorted(samples, key=lambda x: len(x["input_ids"]), reverse=True)

        # Create packed samples
        packed_samples : list[Sample] = []
        for sample in sorted_samples:
            # Try to find a packed sample that can fit this sequence
            packed_sample_found = False
            for packed_sample in packed_samples:
                # Check if current sample fits in packed sample
                if len(packed_sample["input_ids"]) + len(sample["input_ids"]) <= self.config.seq_len:
                    packed_sample["input_ids"].extend(sample["input_ids"])
                    packed_sample["loss_mask"].extend(sample["loss_mask"])
                    packed_sample["position_ids"].extend(sample["position_ids"])
                    packed_sample_found = True
                    break

            # If no suitable packed sample found, create a new packed sample
            if not packed_sample_found:
                packed_samples.append(sample)

        return packed_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def get_dataset(tokenizer, config: DataConfig) -> Dataset:
    """Returns the PyTorch dataset to train on."""
    if config.fake:
        return FakeDataset(tokenizer, config)
    return SFTDataset(tokenizer, config)

def collate_padding(samples: list[Sample], seq_len: int, tokenizer: AutoTokenizer) -> Batch:
    seq_len += 1  # One more token because we lose one
    for sample in samples: 
        if len(sample["input_ids"]) > seq_len:  # Truncate
            sample["input_ids"] = sample["input_ids"][:seq_len]
            sample["loss_mask"] = sample["loss_mask"][:seq_len]
            sample["position_ids"] = sample["position_ids"][:seq_len]
        if len(sample["input_ids"]) < seq_len:  # Pad
            num_pad_tokens = seq_len - len(sample["input_ids"])
            sample["input_ids"] += [tokenizer.pad_token_id] * num_pad_tokens
            sample["loss_mask"] += [0] * num_pad_tokens
            sample["position_ids"] += [0] * num_pad_tokens

    # Stack tensors into tensors of size (batch_size, seq_len)
    batch_input_ids = torch.stack([torch.tensor(sample["input_ids"]) for sample in samples]).long()
    batch_position_ids = torch.stack([torch.tensor(sample["position_ids"]) for sample in samples]).long()
    batch_loss_mask = torch.stack([torch.tensor(sample["loss_mask"]) for sample in samples]).bool()

    return {
        "input_ids": batch_input_ids[:, :-1].contiguous(),
        "target_ids": batch_input_ids[:, 1:].contiguous(),
        "position_ids": batch_position_ids[:, :-1].contiguous(),
        "loss_mask": batch_loss_mask[:, :-1].contiguous(),
    }


def get_dataloader(dataset: Dataset, tokenizer: AutoTokenizer, config: DataConfig) -> DataLoader:
    # Initialize padding collate function
    collate_fn = partial(collate_padding, seq_len=config.seq_len, tokenizer=tokenizer)

    # Initialize rank-aware sampler
    sampler = DistributedSampler(dataset, shuffle=config.shuffle, drop_last=True)
    return iter(DataLoader(dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn, sampler=sampler))
