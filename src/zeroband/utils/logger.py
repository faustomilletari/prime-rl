import logging
from logging import Logger, Formatter
import os

from zeroband.utils.world_info import get_world_info, WorldInfo


class PrimeFormatter(Formatter):
    def __init__(self, world_info: WorldInfo):
        super().__init__()
        self.world_info = world_info

    def format(self, record):
        record.local_rank = self.world_info.local_rank
        log_format = "{asctime} [{name}] [{levelname}] [Rank {local_rank}] {message}"
        formatter = logging.Formatter(log_format, style="{", datefmt="%m-%d %H:%M:%S")
        return formatter.format(record)


ALLOWED_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "CRITICAL": logging.CRITICAL}


def get_logger(name: str) -> Logger:
    # Get logger from Python's built-in registry
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        # Get world info (will instantiate if not already)
        world_info = get_world_info()

        # Set log level
        if world_info.local_rank == 0:
            # On first rank, set log level from env var
            level = os.environ.get("PRIME_LOG_LEVEL", "INFO")
            logger.setLevel(ALLOWED_LEVELS.get(level.upper(), logging.INFO))
        else:
            # Else, only log critical messages by default
            logger.setLevel(logging.CRITICAL)

        # Add handler with custom formatter
        handler = logging.StreamHandler()
        formatter = PrimeFormatter(world_info)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent the log messages from being propagated to the root logger
        logger.propagate = False

    return logger


def reset_logger(name: str) -> None:
    logger = logging.getLogger(name)
    logger.handlers.clear()


if __name__ == "__main__":
    logger = get_logger("TEST")
    logger.info(f"Hi from logger {logger.name}")
    logger.debug("I cannot see this.")
