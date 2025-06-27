from pprint import pprint

from zeroband.training.config import Config as TrainingConfig
from zeroband.training.orchestrator.config import OrchestratorConfig
from zeroband.utils.pydantic_config import BaseConfig, parse_argv


class RLConfig(BaseConfig):
    """Configures the RL training."""

    orchestrator: OrchestratorConfig = OrchestratorConfig()
    train: TrainingConfig = TrainingConfig()


def main(config: RLConfig):
    pprint(config)


if __name__ == "__main__":
    config = parse_argv(RLConfig)
    main(config)
