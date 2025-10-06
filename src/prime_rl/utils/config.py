from typing import Annotated

from pydantic import Field

from prime_rl.utils.pydantic_config import BaseConfig


class ModelConfig(BaseConfig):
    """Configures the model."""

    name: Annotated[str, Field(description="Name or path of the HF model to use.")] = "Qwen/Qwen3-0.6B"

    trust_remote_code: Annotated[
        bool,
        Field(
            description="Whether to trust remote code for tokenizer initialization.",
        ),
    ] = False


class LogConfig(BaseConfig):
    """Configures the logger."""

    level: Annotated[
        str,
        Field(description="Logging level for the process. Will determine the logging verbosity and format."),
    ] = "info"

    file: Annotated[
        bool,
        Field(
            description="Whether to log to a file. If True, will log to a file in the output directory.",
        ),
    ] = True

    log_data: Annotated[
        bool,
        Field(
            description="Whether to log the first data sample to the logger.",
        ),
    ] = False


class LogExtrasConfig(BaseConfig):
    """Configures extra logging for W&B tables."""

    samples: Annotated[
        bool,
        Field(
            description="Whether to log prompt/response samples to W&B tables.",
        ),
    ] = True

    distributions: Annotated[
        bool,
        Field(
            description="Whether to log distributions (like rewards, advantages, etc.) to W&B tables.",
        ),
    ] = True

    interval: Annotated[
        int,
        Field(
            ge=1,
            description="Step interval at which to log extras to W&B table.",
        ),
    ] = 10


class WandbMonitorConfig(BaseConfig):
    """Configures logging to Weights and Biases."""

    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: Annotated[str, Field(description="The W&B project to log to.")] = "prime-rl"

    name: Annotated[
        str | None,
        Field(
            description="The W&B name to to use for logging.",
        ),
    ] = None

    offline: Annotated[bool, Field(description="Whether to run W&B in offline mode.")] = False

    # Individual configs (can only be specified on trainer or orchestrator)
    id: Annotated[
        str | None,
        Field(
            description="The W&B run ID to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.",
        ),
    ] = None

    log_extras: Annotated[
        LogExtrasConfig | None,
        Field(
            description="Configuration for logging extras to W&B tables. If None, no extras are logged.",
        ),
    ] = LogExtrasConfig()


class VariableStoreServerConfig(BaseConfig):
    """Configures the ZeroMQ variable store server co-located with the orchestrator."""

    host: Annotated[
        str,
        Field(
            description="Host address to bind the variable store server to. Use 'localhost' for single-node setups, or '0.0.0.0' for multi-node setups."
        ),
    ] = "localhost"

    port: Annotated[
        int,
        Field(
            ge=1024,
            le=65535,
            description="Port for the variable store server to bind to."
        ),
    ] = 22222

    steps_to_preserve: Annotated[
        int,
        Field(
            ge=1,
            description="Number of recent steps to preserve in the variable store. Older steps will be automatically cleaned up."
        ),
    ] = 2


class VariableStoreClientConfig(BaseConfig):
    """Configures the ZeroMQ variable store client used by trainers."""

    host: Annotated[
        str,
        Field(
            description="Host address of the variable store server. Use 'localhost' for single-node setups."
        ),
    ] = "localhost"

    port: Annotated[
        int,
        Field(
            ge=1024,
            le=65535,
            description="Port of the variable store server."
        ),
    ] = 22222

    timeout: Annotated[
        int,
        Field(
            ge=1,
            description="Timeout in seconds for variable store operations."
        ),
    ] = 300