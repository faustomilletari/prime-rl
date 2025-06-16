from pathlib import Path
from typing import Annotated, ClassVar

import tomli
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseSettings(PydanticBaseSettings):
    """
    Base settings class for all configs.
    """

    # These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
    _GLOBAL_TOML_FILES: ClassVar[list[str]] = []

    toml_files: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="List of extra TOML files to load. If provided, will override all other config files. Note: This field is only read from within configuration files - setting --toml-files from CLI has no effect.",
        ),
    ]

    @classmethod
    def set_global_toml_files(cls, toml_files: list[str]) -> None:
        """
        Set the global TOML files to be used for this config.
        These are two somewhat hacky workarounds inspired by https://github.com/pydantic/pydantic-settings/issues/259 to ensure backwards compatibility with our old CLI system `pydantic_config`
        """
        cls._GLOBAL_TOML_FILES = toml_files

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["BaseSettings"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # This is a hacky way to dynamically load TOML file paths from CLI
        # https://github.com/pydantic/pydantic-settings/issues/259
        return (
            TomlConfigSettingsSource(settings_cls, toml_file=cls._GLOBAL_TOML_FILES),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_prefix="PRIME_",
        env_nested_delimiter="__",
        # By default, we do not parse CLI. To activate, set `_cli_parse_args` to true or a list of arguments at init time.
        cli_parse_args=False,
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_implicit_flags=True,
        cli_use_class_docs_for_groups=True,
    )


class FileMonitorConfig(BaseConfig):
    """Configures logging to a file."""

    path: Annotated[Path | None, Field(default=None, description="The file path to log to")]

    @model_validator(mode="after")
    def validate_path(self):
        if self.path is None:
            raise ValueError("File path must be set when FileMonitor is enabled. Try setting --monitor.file.path")
        return self


class SocketMonitorConfig(BaseConfig):
    """Configures logging to a Unix socket."""

    path: Annotated[Path | None, Field(default=None, description="The socket path to log to")]

    @model_validator(mode="after")
    def validate_path(self):
        if self.path is None:
            raise ValueError("Socket path must be set when SocketMonitor is enabled. Try setting --monitor.socket.path")
        return self


class APIMonitorConfig(BaseConfig):
    """Configures logging to an API via HTTP."""

    url: Annotated[str | None, Field(default=None, description="The API URL to log to")]

    auth_token: Annotated[str | None, Field(default=None, description="The API auth token to use")]

    @model_validator(mode="after")
    def validate_url(self):
        if self.url is None:
            raise ValueError("URL must be set when APIMonitor is enabled. Try setting --monitor.api.url")
        return self

    @model_validator(mode="after")
    def validate_auth_token(self):
        if self.auth_token is None:
            raise ValueError("Auth token must be set when APIMonitor is enabled. Try setting --monitor.api.auth_token")
        return self


class MultiMonitorConfig(BaseConfig):
    """Configures the monitoring system."""

    # All possible monitors (currently only supports one instance per type)
    file: Annotated[FileMonitorConfig, Field(default=None)]
    socket: Annotated[SocketMonitorConfig, Field(default=None)]
    api: Annotated[APIMonitorConfig, Field(default=None)]

    system_log_frequency: Annotated[
        int, Field(default=0, ge=0, description="Interval in seconds to log system metrics. If 0, no system metrics are logged)")
    ]

    def __str__(self) -> str:
        file_str = "disabled" if self.file is None else f"path={self.file.path}"
        socket_str = "disabled" if self.socket is None else f"path={self.socket.path}"
        api_str = "disabled" if self.api is None else f"url={self.api.url}"
        return f"file={file_str}, socket={socket_str}, api={api_str}, system_log_frequency={self.system_log_frequency}"


def check_path_and_handle_inheritance(path: str, seen_files: list[str]):
    """
    Recursively look for inheritance in a toml file. Return a list of all toml files to load.

    Example:
        If config.toml has `toml_files = ["base.toml"]` and base.toml has
        `toml_files = ["common.toml"]`, this returns ["config.toml", "base.toml", "common.toml"]
    """
    if path in seen_files:
        return

    seen_files.append(path)
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomli.load(f)

        if "toml_files" in data:
            maybe_new_files = [path.parent / file for file in data["toml_files"]]

            files = [file for file in maybe_new_files if str(file).endswith(".toml") and file.exists()]
            # todo which should probably look for infinite inheritance loops here
            for file in files:
                check_path_and_handle_inheritance(str(file), seen_files)

    except Exception as e:
        print(f"Error reading {path}: {e}")


# Extract config file paths from CLI to pass to pydantic-settings as toml source
# This enables the use of `@` to pass config file paths to the CLI
def extract_toml_paths(args: list[str]) -> tuple[list[str], list[str]]:
    toml_paths = []
    remaining_args = args.copy()
    for arg, next_arg in zip(args, args[1:] + [""]):
        if arg.startswith("@"):
            toml_path: str
            if arg == "@":  # We assume that the next argument is a toml file path
                toml_path = next_arg
                remaining_args.remove(arg)
                remaining_args.remove(next_arg)
            else:  # We assume that the argument is a toml file path
                remaining_args.remove(arg)
                toml_path = arg.replace("@", "")

            check_path_and_handle_inheritance(toml_path, toml_paths)

    return toml_paths, remaining_args


def to_kebab_case(args: list[str]) -> list[str]:
    """
    Converts CLI argument keys from snake case to kebab case.

    For example, `--max_batch_size 1` will be transformed `--max-batch-size 1`.
    """
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            args[i] = arg.replace("_", "-")
    return args
