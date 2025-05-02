from typing import TYPE_CHECKING, Any, Dict, List
import os

if TYPE_CHECKING:
    PRIME_LOG_LEVEL: str = "INFO"

# Shared environment variables between training and inference
_BASE_ENV: Dict[str, Any] = {
    "PRIME_LOG_LEVEL": lambda: os.getenv("PRIME_LOG_LEVEL", "INFO"),
}


def get_env_value(envs: Dict[str, Any], name: str) -> Any:
    if name not in envs:
        raise AttributeError(f"Invalid environment variable: {name}")
    return envs[name]()


def get_dir(envs: Dict[str, Any]) -> List[str]:
    return list(envs.keys())
