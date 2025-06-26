# save_ckpt_demo.py
import hashlib
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from zeroband.utils.models import ModelType


def compute_state_dict_hash(state_dict: dict) -> str:
    """Compute SHA256 hash of a state dict."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode())
        tensor = state_dict[key]
        # Convert bf16 to float32 for numpy compatibility
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        hasher.update(tensor.cpu().numpy().tobytes())
    return hasher.hexdigest()


def apply_fsdp(model: ModelType, reshard_after_forward: bool = False):
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    for layer_id, transformer_block in enumerate(model.model.layers):
        if reshard_after_forward:
            layer_reshard_after_forward = layer_id < len(model.model.layers) - 1
        else:
            layer_reshard_after_forward = False
        fully_shard(transformer_block, mp_policy=mp_policy, reshard_after_forward=layer_reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def save_ckpt_for_rollout_fast(model, path: Path, dtype: torch.dtype = torch.bfloat16):
    """
    Save the checkpoint for rollout as one unified safetensors file.
    """
    world_info = dist.get_rank(), dist.get_world_size()
    rank, _ = world_info

    path.mkdir(parents=True, exist_ok=True)
    print(f"[rank={rank}] Saving rollout ckpt at {path}")

    cpu_state = {}
    copy_stream = torch.cuda.Stream()  # private stream for DMA

    for key, value in model.state_dict().items():
        if isinstance(value, torch.distributed.tensor.DTensor):
            value = value.to(dtype)
            # only gather after the downcast to dtype as it will be faster
            value = value.full_tensor()  # ideally would only be gathered on rank 0
        else:
            value = value.to(dtype)

        if rank == 0:
            # we use pin and shared memory for avoiding multi processing data duplication
            host_buf = torch.empty_like(value, device="cpu", pin_memory=True)
            with torch.cuda.stream(copy_stream):
                host_buf.copy_(value, non_blocking=True)

            cpu_state[key] = host_buf

    torch.cuda.synchronize()
    torch.distributed.barrier()

    if rank == 0:
        torch.save(cpu_state, path / "model.pt")


def save_ckpt_for_rollout_slow(model, path: Path, dtype: torch.dtype = torch.bfloat16):
    """
    Save the checkpoint for rollout as one unified safetensors file.
    """
    world_info = dist.get_rank(), dist.get_world_size()
    rank, _ = world_info

    path.mkdir(parents=True, exist_ok=True)
    print(f"[rank={rank}] Saving rollout ckpt at {path}")

    cpu_state = {}
    for key, value in model.state_dict().items():
        if isinstance(value, torch.distributed.tensor.DTensor):
            value = value.to(dtype).full_tensor()  # all-gather
        else:
            value = value.to(dtype)

        if rank == 0:
            host_buf = torch.empty_like(value, device="cpu")
            host_buf.copy_(value)
            cpu_state[key] = host_buf

    if rank == 0:
        torch.save(cpu_state, path / "model.pt")


def hash_ckpt(ckpt_dir: Path):
    # Load checkpoint and compute hash
    ckpt_path = ckpt_dir / "model.pt"
    state_dict = torch.load(ckpt_path, map_location="cpu")

    return compute_state_dict_hash(state_dict)


def main():
    # env vars are filled in by torchrun
    rank = int(os.environ["RANK"])
    # world_sz = int(os.environ["WORLD_SIZE"])
    # device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(rank)

    # Load Qwen3 0.6B model
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-0.6B")

    # Apply FSDP sharding
    apply_fsdp(model, reshard_after_forward=False)

    ckpt_dir = Path("demo_ckpt")
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    save_ckpt_for_rollout_fast(model, ckpt_dir)
    if rank == 0:
        print(f"Fast ckpt hash: {hash_ckpt(ckpt_dir)}")

    shutil.rmtree(ckpt_dir, ignore_errors=True)

    save_ckpt_for_rollout_slow(model, ckpt_dir)
    if rank == 0:
        print(f"Slow ckpt hash: {hash_ckpt(ckpt_dir)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
