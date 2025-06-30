import time
from pathlib import Path
from typing import Any, Generator, TypedDict

import pyarrow.parquet as pq
import torch
from jaxtyping import Float, Int
from pyarrow import dataset as ds
from torch.utils.data import DataLoader, IterableDataset

from zeroband.training.config import CollateMode, DataConfig
from zeroband.training.parquet import SCHEMA
from zeroband.training.world_info import get_world_info
from zeroband.utils.logger import get_logger

STABLE_FILE = "stable"


class DatasetOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "seq"]
    advantages: Float[torch.Tensor, "seq"]
    loss_mask: Int[torch.Tensor, "seq"]
    logprobs: Float[torch.Tensor, "seq"]

    temperature: float


class FakeTokenizedDataset(IterableDataset[DatasetOutput]):
    """A dummy dataset that generates random sequences with the full schema including new columns."""

    def __init__(self, seq_len: int, vocab_size: int):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        assert vocab_size > 3, "Vocab size must be greater than 3"
        self.step = 0

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
        while True:
            world_info = get_world_info()

            # we divide by local world rank to simulate imbalanced in the data
            seq_len = self.seq_len // (1 + world_info.local_rank)

            len_ = torch.randint(1, seq_len + 1, (1,)).item()
            input_ids = torch.randint(3, self.vocab_size, (len_,))
            advantages = torch.randn(len_)
            self.step += 1
            logprobs = -torch.abs(torch.randn(len_))  # Negative values for log probs

            yield {
                "input_ids": input_ids,
                "advantages": advantages,
                "loss_mask": torch.ones(len_).int(),
                "logprobs": logprobs,
                "temperature": 1.0,
            }


class ParquetDataset(IterableDataset[DatasetOutput]):
    """
    This dataset iterate over a parquet files using parquet dataset.
    It will yield a single sample at a time.
    """

    def __init__(
        self,
        path: Path,
        step_count_init: int,
        pq_read_bs: int = 64,
    ):
        self._logger = get_logger()
        self._path = path
        self._pq_read_bs = pq_read_bs

        self._world_info = get_world_info()

        self._step_count = step_count_init  # we immediately bump the step count by one later

    def __iter__(self) -> Generator[DatasetOutput, Any, None]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        while True:
            self._step_count += 1

            self._logger.debug(f"Data processing step {self._step_count}")

            file = self._path / f"step_{self._step_count}.parquet"
            while not file.exists():
                time.sleep(0.1)

            if not self.validate_schema_pa_file(file):
                raise ValueError(f"Schema of file {file} is not the same as the schema of the pa_schema")

            dataset = ds.dataset(file, format="parquet")

            required_columns = [
                "input_tokens",
                "output_tokens",
                "advantages",
                "input_logprobs",
                "output_logprobs",
                "temperature",
            ]

            scanner = dataset.scanner(columns=required_columns, batch_size=self._pq_read_bs)
            counter = 0

            for batch in scanner.to_batches():
                for i in range(len(batch["input_tokens"])):
                    counter += 1
                    if self.should_skip_index(
                        index=counter,
                        world_size=self._world_info.world_size,
                        rank=self._world_info.rank,
                        num_workers=num_workers,
                        workers_id=worker_id,
                    ):
                        continue

                    input_ids = torch.tensor(batch["input_tokens"][i].as_py())
                    output_ids = torch.tensor(batch["output_tokens"][i].as_py())

                    ids = torch.cat([input_ids, output_ids], dim=0)
                    loss_mask = torch.cat(
                        [torch.zeros(len(input_ids)), torch.ones(len(output_ids))],
                        dim=0,
                    ).int()

                    adv_value = batch["advantages"][i].as_py()

                    adv = torch.tensor([adv_value] * len(ids))  # advantage

                    input_logprobs = torch.tensor(batch["input_logprobs"][i].as_py())
                    output_logprobs = torch.tensor(batch["output_logprobs"][i].as_py())
                    # Concatenate and remove the first token (BOS)
                    logprobs = torch.cat([input_logprobs, output_logprobs], dim=0)
                    assert logprobs.shape == ids.shape, (
                        f"logprobs: {logprobs.shape} should be the same as ids: {ids.shape}"
                    )

                    data = {
                        "input_ids": ids,
                        "advantages": adv,
                        "loss_mask": loss_mask,
                        "logprobs": logprobs,
                        "temperature": batch["temperature"][i].as_py(),
                    }
                    yield data

    @staticmethod
    def validate_schema_pa_file(file: Path):
        """Check if the schema of the parquet file is the same as the schema of the pa_schema"""
        try:
            parquet_schema = pq.read_schema(file)
            return parquet_schema.equals(SCHEMA)
        except Exception as e:
            print(f"Error reading schema for file {file}: {e}")
            return False

    @staticmethod
    def should_skip_index(index: int, world_size: int, rank: int, num_workers: int, workers_id: int) -> bool:
        """
        This function is used to skip the index if it is not the responsibility of the current worker.
        It take into account the number of workers as well as rank.

        Its equivalent to checking if index is in samples[rank::world_size][workers_id::num_workers]

        Returns:
            True if the index should be skipped
            False if the index should be processed

        PS: would love to remove this function and use samples[rank::world_size][workers_id::num_workers] but not sure how it would work across pq dataset
        """
        # First, check if the index belongs to this rank (distributed across world_size)
        if (index % world_size) != rank:
            return True

        # Next, compute the position within the rank's subset
        rank_position = index // world_size

        # Check if this position belongs to this worker (distributed across num_workers)
        if (rank_position % num_workers) != workers_id:
            return True

        # If we passed both checks, this index should be processed by this worker
        return False


class BatchOutput(TypedDict):
    # token level
    input_ids: Int[torch.Tensor, "batch seq"]
    advantages: Float[torch.Tensor, "batch seq"]
    loss_mask: Int[torch.Tensor, "batch seq"]
    position_ids: Int[torch.Tensor, "batch seq"]
    logprobs: Float[torch.Tensor, "batch seq_minus_1"]

    # batch level
    temperature: float
    total_tokens: int


# def collate_fn(
#     samples: list[DatasetOutput], max_seq_len: int, pad_token_id: int
# ) -> BatchOutput:
#     """
#     This take a list of samples that should be packed together along the sequence dimension. Will add padding at the end if needed and
#     clipped to max_seq_len
#     """

#     total_len = sum(len(sample["input_ids"]) for sample in samples)

#     inputs_ids = [sample["input_ids"] for sample in samples]
#     advantages = [sample["advantages"] for sample in samples]
#     loss_masks = [sample["loss_mask"] for sample in samples]

#     # Handle logprobs if available
#     all_logprobs = [sample["logprobs"] for sample]


#     position_ids = [torch.arange(0, len(sample["input_ids"]), dtype=torch.int32) for sample in samples]

#     temperature = samples[0]["temperature"]
#     assert all(temperature == sample["temperature"] for sample in samples), "all temperatures must be the same"

#     if total_len < max_seq_len:
#         padding_len = max_seq_len - total_len

#         inputs_ids.append(
#             torch.full(
#                 (padding_len,), fill_value=pad_token_id, dtype=inputs_ids[0].dtype
#             )
#         )
#         advantages.append(torch.zeros(padding_len, dtype=advantages[0].dtype))
#         loss_masks.append(torch.zeros(padding_len, dtype=loss_masks[0].dtype).int())
#         position_ids.append(torch.arange(0, padding_len, dtype=torch.int32))

#         if has_logprobs:
#             # For logprobs, we pad with zeros (these will be masked out anyway)
#             logprobs.append(torch.zeros(padding_len, dtype=logprobs[0].dtype))

#     # Concatenate logprobs if available
#     concat_logprobs = None
#     if has_logprobs:
#         # we remove the first logprob because it corresponds to the bos token
#         concat_logprobs = torch.cat(logprobs, dim=0)[1:max_seq_len].unsqueeze(0)

#     return {
#         # token level
#         "input_ids": torch.cat(inputs_ids, dim=0)[:max_seq_len].unsqueeze(0),
#         "advantages": torch.cat(advantages, dim=0)[:max_seq_len].unsqueeze(0),
#         "loss_mask": torch.cat(loss_masks, dim=0)[:max_seq_len].unsqueeze(0),
#         "position_ids": torch.cat(position_ids, dim=0)[:max_seq_len].unsqueeze(0),
#         "logprobs": concat_logprobs,
#         "temperature": temperature,
#         "total_tokens": total_len,
#     }


class PaddingDataset(IterableDataset[BatchOutput]):
    """
    This dataset will pad each entry in the batch to the max_seq_len and return a batch of size micro_bs.
    """

    def __init__(
        self,
        dataset: IterableDataset[DatasetOutput],
        max_seq_len: int,
        pad_token_id: int,
        micro_bs: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.micro_bs = micro_bs
        self.batch_size = batch_size

        assert batch_size % micro_bs == 0, "batch_size must be divisible by micro_bs"

    def __iter__(self) -> Generator[list[BatchOutput], Any, None]:
        """
        What this function does is:
            * iterate over **single** sample from the dataset
            * for each sample pad it to the max_seq_len
            * once it reach micro_bs, add the batch to the list of batches
            * once the list of batches reach batch_size // micro_bs, yield the list of batches
            * repeat
        """

        current_micro_batch = []
        current_batch = []
        for sample in self.dataset:
            current_micro_batch.append(self.pad_sample(sample))
            if len(current_micro_batch) == self.micro_bs:
                current_batch.append(self.collate_fn(current_micro_batch))
                current_micro_batch = []

            if len(current_batch) == self.batch_size // self.micro_bs:
                yield current_batch
                current_batch = []
                current_micro_batch = []

    def pad_sample(self, sample: DatasetOutput):
        """
        This function will pad the batch to the max_seq_len
        """
        seq_len = len(sample["input_ids"])
        if seq_len < self.max_seq_len:
            padding_len = self.max_seq_len - seq_len
            sample["input_ids"] = torch.cat(
                [sample["input_ids"], torch.full((padding_len,), self.pad_token_id, dtype=sample["input_ids"].dtype)]
            )
            sample["advantages"] = torch.cat(
                [sample["advantages"], torch.zeros(padding_len, dtype=sample["advantages"].dtype)]
            )
            sample["loss_mask"] = torch.cat(
                [sample["loss_mask"], torch.zeros(padding_len, dtype=sample["loss_mask"].dtype).int()]
            )
        else:
            sample["input_ids"] = sample["input_ids"][: self.max_seq_len]
            sample["advantages"] = sample["advantages"][: self.max_seq_len]
            sample["loss_mask"] = sample["loss_mask"][: self.max_seq_len]

        sample["position_ids"] = torch.arange(0, self.max_seq_len, dtype=torch.int32)

        return sample

    def collate_fn(self, samples: list[DatasetOutput]) -> BatchOutput:
        """
        This function will collate the samples into a batch.
        """
        batch = {}
        for key in ["input_ids", "advantages", "loss_mask", "position_ids", "logprobs"]:
            batch[key] = torch.stack([sample[key] for sample in samples], dim=0)

        batch["temperature"] = samples[0]["temperature"]
        assert all(sample["temperature"] == batch["temperature"] for sample in samples), (
            "all temperatures must be the same"
        )

        batch["total_tokens"] = sum(len(sample["input_ids"]) for sample in samples)

        return batch


class PackingDataset(IterableDataset[BatchOutput]):
    """
    This dataset will pack the batch into a single batch in a efficient manner
    it will return a batch where tensor are shaped [1, micro_bs * seq_len]
    """

    def __init__(
        self,
        dataset: IterableDataset[DatasetOutput],
        max_seq_len: int,
        micro_bs: int,
        pad_token_id: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len * micro_bs
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Generator[BatchOutput, Any, None]:
        # current_batch = []
        # for sample in self.dataset:
        raise NotImplementedError("Not implemented")


def get_dataloader(
    tokenizer,
    batch_size: int,
    micro_bs: int,
    data_config: DataConfig,
    step_count_init: int,
    collate_mode: CollateMode,
) -> IterableDataset[list[BatchOutput]]:
    """Get a dataloader for the training dataset"""

    path = data_config.path

    if data_config.fake:
        train_dataset = FakeTokenizedDataset(data_config.seq_length, len(tokenizer))
    else:
        train_dataset = ParquetDataset(Path(path), step_count_init=step_count_init)

    match collate_mode:
        case "padding":
            dataset = PaddingDataset(
                train_dataset, data_config.seq_length, tokenizer.pad_token_id, micro_bs, batch_size
            )
            return DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)

        case "packing":
            dataset = PackingDataset(
                train_dataset, data_config.seq_length, micro_bs, tokenizer.pad_token_id, batch_size
            )
            raise NotImplementedError("Not implemented")
        case _:
            raise ValueError(f"Invalid collate mode: {collate_mode}")


# def pack_datatset_outputs_efficiently(
#     batch_optim: list[DatasetOutput], max_seq_len: int
# ) -> list[list[DatasetOutput]]:
#     """
#     This function will pack the batch into a single batch in a efficient manner
#     """
#     ## we sorted by inputs_ids

#     batch_with_len = [(len(sample["input_ids"]), sample) for sample in batch_optim]

#     sorted_batch = sorted(batch_with_len, key=lambda x: x[0], reverse=True)

#     ## we create bins
#     batches: list[list[DatasetOutput]] = []

#     ## we pack the bins

#     for seq_len, sample in sorted_batch:
#         # Try to find a bin that can fit this sequence
#         bin_found = False
#         for bin_idx, bin_content in enumerate(batches):
#             # Calculate current bin length
#             bin_len = sum(len(s["input_ids"]) for s in bin_content)
#             # Check if sequence fits in this bin
#             if bin_len + seq_len <= max_seq_len:
#                 batches[bin_idx].append(sample)
#                 bin_found = True
#                 break

#         # If no suitable bin found, create a new bin
#         if not bin_found:
#             batches.append([sample])

#     return batches


# def data_parallel_rebalancing(micro_batches: list[BatchOutput]) -> list[BatchOutput]:
#     """
#     This function will duplicate the first micro_batch to match the number of grad acc steps on each gpu
#     Otherwise will block FSDP forward and backward all gather.
#     """
#     num_grad_acc_steps = len(micro_batches)

#     max_grad_acc_step = num_grad_acc_steps
#     if dist.is_initialized():
#         max_grad_acc_step = torch.tensor(num_grad_acc_steps, dtype=torch.int32).to(
#             "cuda"
#         )
#         dist.all_reduce(max_grad_acc_step, op=dist.ReduceOp.MAX, group=None)
#         max_grad_acc_step = int(max_grad_acc_step.item())

#     empty_batch_count = max_grad_acc_step - num_grad_acc_steps

#     for _ in range(empty_batch_count):
#         empty_batch = {}

#         for key, value in micro_batches[0].items():
#             if isinstance(value, torch.Tensor):
#                 empty_batch[key] = value.clone()
#             else:
#                 empty_batch[key] = value

#         micro_batches.append(empty_batch)

#     return micro_batches


# def packed_batch_packing(
#     batch_optim: list[DatasetOutput], max_seq_len: int, pad_token_id: int, micro_bs: int
# ) -> list[BatchOutput]:
#     """
#     this function will pack the batch into [1, seq_len] microbatch tensors with positions ids for calling fa2 with sequence packing
#     """
#     max_seq_len = max_seq_len * micro_bs

#     batches = pack_datatset_outputs_efficiently(batch_optim, max_seq_len=max_seq_len)

#     micro_batches = [
#         collate_fn(bin, pad_token_id=pad_token_id, max_seq_len=max_seq_len)
#         for bin in batches
#     ]

#     return data_parallel_rebalancing(micro_batches)


# def merge_batches_padding(batches: list[BatchOutput]) -> BatchOutput:
#     # Check if any batch has logprobs
#     has_logprobs = any(b["logprobs"] is not None for b in batches)
#     merged_logprobs = None
#     if has_logprobs:
#         # If some batches have logprobs, all should have them
#         merged_logprobs = torch.cat(
#             [b["logprobs"] for b in batches if b["logprobs"] is not None], dim=0
#         )

#     # All batches should have the same temperature
#     temperatures = [b["temperature"] for b in batches]
#     assert all(temp == temperatures[0] for temp in temperatures), (
#         "all temperatures must be the same"
#     )

#     return {
#         # token level
#         "input_ids": torch.cat([b["input_ids"] for b in batches], dim=0),
#         "advantages": torch.cat([b["advantages"] for b in batches], dim=0),
#         "loss_mask": torch.cat([b["loss_mask"] for b in batches], dim=0),
#         "position_ids": torch.cat([b["position_ids"] for b in batches], dim=0),
#         "logprobs": merged_logprobs,
#         # batch level
#         "temperature": temperatures[0],
#         "total_tokens": sum(b["total_tokens"] for b in batches),
#     }
