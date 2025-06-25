# (Jack): This is an umerged patch to fix a bug in vllm https://github.com/vllm-project/vllm/pull/19940
# This can be removed once the patch is merged and vllm is updated.
import zeroband.inference.monkeypatch_sampling_metadata  # noqa: F401

import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
import uuid

# Import environment before any other imports
# ruff: noqa: I001
from zeroband.inference import envs

import pyarrow.parquet as pq
import requests
import torch
from openai import OpenAI
from toploc.utils import sha256sum
from transformers import AutoTokenizer

from zeroband.utils.pydantic_config import parse_argv
from zeroband.inference.config import Config as InferenceConfig
from zeroband.inference.parquet import get_parquet_table
from zeroband.inference.rewards import compute_vllm_rewards
from zeroband.utils.monitor import setup_monitor
from zeroband.inference.utils import (
    get_inference_input_output_flops,
)
from zeroband.training.mp import EnvWrapper
from zeroband.utils.utils import clean_exit
from zeroband.inference.logger import setup_logger

from zeroband.environments.registry import get_environment
from zeroband.inference.model_client import ModelClient


@clean_exit
def inference(config: InferenceConfig):
    # Initialize the logger
    dp_rank = int(os.environ.get("DP_RANK", 0))
    logger = setup_logger(config.log, parallel_config=config.parallel, dp_rank=dp_rank)
    logger.info("Starting inference")

    # Optionally, clean the rollout path
    if config.clean_rollout_path and config.rollout_path is not None:
        logger.info(f"Cleaning rollout path ({config.rollout_path})")
        shutil.rmtree(config.rollout_path, ignore_errors=True)

    # Initialize metrics
    monitor = setup_monitor(config.monitor, config.task_id, config)

    # TODO: spawn model server process directly (or wait if launched separately)
    # TODO: pre-fetch model weights
    # TODO: re-add tensor/pipeline parallelism (patch_model_load)
    # TODO: pipeline parallel communication and hook (setup_comm / setup_hooks)
    # TODO: set up TOPLOC

    # Connect to model server
    model_name = config.model.name
    model_server_url = "http://0.0.0.0:8000"  # TODO: config
    oai_client = OpenAI(base_url=model_server_url)
    model_client = ModelClient(model_server_url)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize dataset
    logger.info(f"Initializing environment (name={config.data.name})")

    vf_env = get_environment(config.data.name)

    dataset = vf_env.get_dataset()

    # optionally, get eval dataset (for periodic evals w/ diff sampling params, n=1)
    eval_dataset = vf_env.get_eval_dataset()

    # TODO: length penalty module in Environments
    # TODO: prompt length filtering (can be added to Environments)
    # TODO: optional difficulty filtering module
    # TODO: optionally shuffle dataset (can be done in Environments, get_dataset supports seed arg)

    start_time = time.time()

    # initialize sampling args (Environment.generate takes dict, not vLLM-specific SamplingParams)
    sampling_args = config.sampling

    if model_client is not None:
        max_batch_size = model_client.get_max_batch_size()
    else:
        max_batch_size = config.max_batch_size

    # TODO: num_generations as arg, separate from sampling.n -- we should duplicate prompts directly
    # Compute the true batch size
    problems_per_batch = max_batch_size // config.num_generations  # type: ignore
    batch_size = problems_per_batch * config.num_generations  # type: ignore
    logger.info(
        f"Problems per batch: {max_batch_size} // {config.num_generations} = {problems_per_batch}, batch size: {problems_per_batch} * {config.num_generations} = {batch_size} (missing: {max_batch_size % config.num_generations})"  # type: ignore
    )

    # TODO: resume from checkpoint / step_path
    ckpt_step = 0
    real_step = ckpt_step

    # This is used by the seeding logic to make sure we dont generate the same samples twice if we do multiple batches for a step
    current_step_batch_counter = 1
    total_problems = 0
    total_samples = 0
    total_tokens = 0

    dataset_offset = 0
    while True:
        if config.rl and config.rl.step_endpoint is not None:
            # We get the step from the endpoint at the start of each batch to know what to work on
            try:
                new_real_step = requests.get(config.rl.step_endpoint).json()
            except Exception as e:
                logger.warning(f"Failed to get step from endpoint {config.rl.step_endpoint}: {e}")
                time.sleep(10)
                continue

            if new_real_step != real_step:
                real_step = new_real_step
                current_step_batch_counter = 1
            else:
                current_step_batch_counter += 1

        logger.info(f"Inference step {real_step} (Checkpoint step: {ckpt_step})")
        if config.rl and real_step - ckpt_step > config.rl.async_level:
            logger.info(f"Required to reload model weights for step {ckpt_step} from {config.rl.ckpt_path}")
            ckpt_step = real_step - config.rl.async_level
            attempt_count = 0
            while True:
                stable_file = Path(config.rl.ckpt_path) / f"step_{ckpt_step}/stable"
                if stable_file.exists():
                    logger.info(f"Reloading model weights for step {ckpt_step} from {stable_file}")
                    model_client.reload_weights(Path(config.rl.ckpt_path) / f"step_{ckpt_step}/model.safetensors")
                    total_problems = 0
                    total_tokens = 0
                    logger.success(f"Reloaded model weights for step {ckpt_step} from {stable_file}")
                    break
                if attempt_count % 30 == 0:
                    logger.info(f"No stable file found at {stable_file}, waiting for new checkpoint")
                time.sleep(1)
                attempt_count += 1

        if config.step_path is not None:
            logger.info(f"Writing current inference step ({real_step}) to {config.step_path}")
            if not config.step_path.exists():
                config.step_path.parent.mkdir(parents=True, exist_ok=True)
            config.step_path.write_text(str(real_step))

        # Get batch indices
        # TODO: if node_address_int is not None case
        indices = [(dataset_offset + j) % len(dataset) for j in range(problems_per_batch)]  # type: ignore
        # TODO: handle collisions, looping

        logger.debug(f"Sampling batch with indices [{' '.join(map(str, indices[:3]))}...{' '.join(map(str, indices[-3:]))}]")
        inputs = dataset.select(indices)  # type: ignore
        # problems = Dataset with prompt, answer, info, task

        generate_start_time = time.time()
        outputs = vf_env.generate(
            inputs=inputs,
            client=oai_client,
            model=model_name,
            sampling_args=sampling_args,  # type: ignore
            max_concurrent=max_batch_size,  # type: ignore
        )
        generation_time = time.time() - generate_start_time

        # post-process outputs
        prompts = outputs["prompts"]
        completions = outputs["completions"]
        states = outputs["states"]
        rewards = outputs["rewards"]
        task_types = outputs["task_types"]
        verification_infos = outputs["verification_infos"]
        proofs = outputs["proofs"]
        target_lengths = outputs["target_lengths"]

        # TODO: extract proofs from outputs

        # Compute progress metrics
        batch_problems = len(problems)
        batch_samples = sum(len(req.outputs) for req in request_outputs)
        batch_input_tokens = sum(len(req.prompt_token_ids) for req in request_outputs)
        batch_output_tokens = sum(sum(len(output.token_ids) for output in req.outputs) for req in request_outputs)
        batch_tokens = batch_input_tokens + batch_output_tokens
        total_tokens += batch_tokens
        total_problems += batch_problems
        total_samples += batch_samples
        logger.success(f"Generated {batch_samples} samples for {batch_problems} problems in {generation_time:.2f}s")

        # Print example
        first_prompt = tokenizer.decode(request_outputs[0].prompt_token_ids)
        first_completion = tokenizer.decode(request_outputs[0].outputs[0].token_ids)
        logger.debug(f"Showing example (first completion):\n{first_prompt}{first_completion}")

        # Log progress metrics
        progress_metrics = {
            "progress/batch_problems": batch_problems,
            "progress/batch_samples": batch_samples,
            "progress/batch_tokens": batch_tokens,
            "step": real_step,
        }
        monitor.log(progress_metrics)

        # Compute performance metrics
        batch_tokens_per_second = batch_tokens / generation_time
        batch_samples_per_minute = batch_samples / generation_time * 60
        batch_avg_seq_length = batch_tokens / batch_size
        logger.info(
            f"Batch throughput: {batch_tokens_per_second:.2f} tokens/sec, {batch_samples_per_minute:.2f} samples/min ({batch_tokens} tokens in {generation_time:.2f}s, avg seq len: {batch_avg_seq_length:.1f})"
        )

        # Log performance metrics
        perf_metrics = {
            "performance/batch_tokens_per_second": batch_tokens_per_second,
            "performance/batch_samples_per_minute": batch_samples_per_minute,
            "performance/batch_avg_seq_length": batch_avg_seq_length,
            "step": real_step,
        }
        monitor.log(perf_metrics)

        # Compute and log rewards and advantages
        logger.info("Computing rewards and advantages")
        request_rewards = compute_vllm_rewards(request_outputs, verification_infos, task_types, config.rewards)
        batch_rewards = sum(sum(r.reward for r in req.rewards) for req in request_rewards) / batch_samples
        logger.info(f"Average reward of the batch: {batch_rewards:.2f}")
        monitor.log({"rewards/batch_rewards": batch_rewards, "step": real_step})

        if sampling_params.seed is not None:
            sampling_seeds = [sampling_params.seed + i for i in range(sampling_params.n)] * problems_per_batch
        else:
            sampling_seeds = [None] * batch_samples

        # Get parquet table
        table = get_parquet_table(
            request_outputs,
            request_rewards,
            prompts,
            proofs,
            ckpt_step,
            target_lengths,
            problems,
            enable_logprobs=config.sampling.logprobs is not None,
            seeds=sampling_seeds,
            temperature=sampling_params.temperature,
        )

        # Save outputs to parquet file
        step_path = Path(config.rollout_path) / f"step_{real_step}"
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"{uuid.uuid4()}.parquet"
        logger.info(f"Saving batch outputs to {save_path}")
        pq.write_table(table, save_path)

        # Log file metadata
        sha256 = sha256sum(save_path)
        flop_counts = [
            get_inference_input_output_flops(config.model.name, len(input_tokens), len(output_tokens))
            for input_tokens, output_tokens in zip(table.column("input_tokens").to_pylist(), table.column("output_tokens").to_pylist())
        ]

        work_submission = {
            "output/output_flops": sum(output_flops for _, output_flops in flop_counts) // config.parallel.pp.world_size,
            "output/input_flops": sum(input_flops for input_flops, _ in flop_counts) // config.parallel.pp.world_size,
            "output/save_path": save_path.as_posix(),
            "output/sha256": sha256,
            "output/step": real_step,
        }
        monitor.log(work_submission, exclude=["wandb"])

        real_step += 1

        if config.max_steps is not None and real_step > config.max_steps:
            logger.info(f"Reached max steps {config.max_steps}, stopping inference")
            break

        dataset_offset += problems_per_batch

    logger.success(f"Inference finished! Generated {total_samples} samples for {total_problems} problems")


def main(config: InferenceConfig) -> list[mp.Process]:
    processes = []

    if config.parallel.dp > 1:
        if config.parallel.tp == "auto":
            assert torch.cuda.device_count() % config.parallel.dp == 0, "Number of GPUs must be divisible by DP"
            config.parallel.tp = torch.cuda.device_count() // config.parallel.dp
        gpu_ids = envs.CUDA_VISIBLE_DEVICES
        gpu_ids_per_rank = [gpu_ids[i : i + config.parallel.tp] for i in range(0, len(gpu_ids), config.parallel.tp)]
        for rank, gpu_ids in enumerate(gpu_ids_per_rank):
            env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)), "DP_RANK": str(rank)}
            process = mp.Process(target=EnvWrapper(inference, env), args=(config,))
            processes.append(process)
    else:
        if config.parallel.tp == "auto":
            config.parallel.tp = torch.cuda.device_count()
        inference(config)

    # Start all processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":
    # Set spawn method before any other multiprocessing code
    mp.set_start_method("spawn")

    config = parse_argv(InferenceConfig)

    if config.rl and config.rl.step_endpoint is not None:
        current_step = requests.get(config.rl.step_endpoint).json()
        assert isinstance(current_step, int), "Current step must be an integer"

    # Maybe start shardcast downloader
    from zeroband.inference import envs as inference_envs

    if inference_envs.SHARDCAST_SERVERS is not None:
        assert config.rl is not None, "RL config is required when SHARDCAST_SERVERS is set"
        from zeroband.inference.shardcast_downloader import run_main_bg

        shardcast_process = run_main_bg(
            inference_envs.SHARDCAST_SERVERS,
            config.rl.ckpt_path,
            config.rl.async_level + 1,
            # TODO: maybe +1 because we most likely won't download the current step in time?
            # We could deadlock though.
            max(current_step - config.rl.async_level, 1),
        )
    else:
        shardcast_process = None

    try:
        main(config)

    finally:
        if shardcast_process is not None:
            import os
            import signal

            # SIGTERM is not working, so we use SIGKILL
            os.kill(shardcast_process.pid, signal.SIGKILL)
            shardcast_process.join()
