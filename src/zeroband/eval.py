# Import environment before any other imports
# ruff: noqa
import time
from huggingface_hub import snapshot_download

from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.inference.eval.config import Config as EvalConfig
from zeroband.inference.eval.utils import run_benchmark
from zeroband.inference.utils import setup_model, reload_checkpoint
from zeroband.inference.eval.logger import setup_logger
from zeroband.inference.eval.utils import run_benchmark
from zeroband.utils.utils import clean_exit




@clean_exit
def main(config: EvalConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting evaluation")
    logger.info(f"Evaluation config: {config.eval}")

    # Initialize the monitor
    setup_monitor(config.monitor, None, config)

    # Pre-download the model weights
    logger.info(f"Downloading model weights for {config.model.name}")
    start_time = time.time()
    snapshot_download(config.model.name)
    logger.success(f"Downloaded model weights in {time.time() - start_time:.2f}s")

    # Initializing the model and tokenizer
    logger.info(f"Initializing model and tokenizer ({config.model} tensor_parallel_size={config.parallel.tp} seed={config.seed})")
    start_time = time.time()
    llm = setup_model(config.model, tp=config.parallel.tp, seed=config.seed)
    logger.success(f"Initialized model and tokenizer in {time.time() - start_time:.2f}s")

    # Run benchmarks on base model
    logger.info(f"Running benchmarks on base model {config.model.name}")
    for benchmark in config.eval.benchmarks:
        run_benchmark(llm, benchmark, config.model, config.sampling, config.eval, seed=config.seed)

    # If specified, run online evaluation
    if config.eval.online:
        logger.info(
            f"Running online evaluation on {config.model.name} every {config.eval.online.interval} steps from checkpoint directory {config.eval.online.ckpt_path}"
        )
        while True:
            # Reload model weights from checkpoint once available
            step = config.eval.online.interval
            llm = reload_checkpoint(llm, config.eval.online.ckpt_path, step)

            # Run benchmarks on new checkpoint
            logger.info(f"Running benchmarks for checkpoint step {step}")
            for benchmark in config.eval.benchmarks:
                run_benchmark(llm, benchmark, config.model, config.sampling, config.eval, seed=config.seed)

            # Update eval step to next checkpoint step
            step += config.eval.online.interval

            if config.eval.online.max_steps and step > config.eval.online.max_steps:
                logger.info(f"Reached maximum number of steps ({config.eval.online.max_steps}). Stopping online evaluation.")
                break

    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main(parse_argv(EvalConfig))
