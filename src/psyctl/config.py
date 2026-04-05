"""Simple configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar, overload

T = TypeVar("T")


# Overloads for different types (bool before int since bool is subclass of int)
@overload
def get_env(key: str, default: str, cast_type: type[str] = ...) -> str: ...


@overload
def get_env(key: str, default: bool, cast_type: type[bool]) -> bool: ...


@overload
def get_env(key: str, default: int, cast_type: type[int]) -> int: ...


@overload
def get_env(key: str, default: float, cast_type: type[float]) -> float: ...


@overload
def get_env(key: str, default: Path, cast_type: type[Path]) -> Path: ...


# Overloads for None defaults
@overload
def get_env(
    key: str, default: None = ..., cast_type: type[str] = ...
) -> str | None: ...


@overload
def get_env(key: str, default: None, cast_type: type[Path]) -> Path | None: ...


# Implementation
def get_env(key: str, default=None, cast_type: type = str):
    """Get environment variable with type casting."""
    value = os.getenv(key)
    if value is None:
        return default
    if cast_type is bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif cast_type is int:
        return int(value)
    elif cast_type is float:
        return float(value)
    elif cast_type is Path:
        return Path(value)
    return value


# Model settings
DEFAULT_MODEL: str = get_env("PSYCTL_DEFAULT_MODEL", "gemma-3-270m-it")
DEFAULT_DEVICE: str = get_env("PSYCTL_DEFAULT_DEVICE", "auto")

# Hugging Face settings
HF_TOKEN: str | None = get_env("HF_TOKEN")

# OpenRouter settings
OPENROUTER_API_KEY: str | None = get_env("OPENROUTER_API_KEY")
OPENROUTER_DEFAULT_MODEL: str = get_env(
    "OPENROUTER_DEFAULT_MODEL", "qwen/qwen3-next-80b-a3b-instruct"
)

# Dataset settings
DEFAULT_DATASET_SIZE: int = get_env("PSYCTL_DEFAULT_DATASET_SIZE", 1000, int)
DEFAULT_BATCH_SIZE: int = get_env("PSYCTL_DEFAULT_BATCH_SIZE", 8, int)
INFERENCE_BATCH_SIZE: int = get_env("PSYCTL_INFERENCE_BATCH_SIZE", 16, int)
MAX_WORKERS: int = get_env("PSYCTL_MAX_WORKERS", 4, int)
CHECKPOINT_INTERVAL: int = get_env("PSYCTL_CHECKPOINT_INTERVAL", 100, int)

# Steering settings
DEFAULT_LAYER: str = get_env("PSYCTL_DEFAULT_LAYER", "model.layers[13].mlp.down_proj")
STEERING_STRENGTH: float = get_env("PSYCTL_STEERING_STRENGTH", 1.0, float)

# Directory settings
OUTPUT_DIR: Path = get_env("PSYCTL_OUTPUT_DIR", Path("./output"), Path)
DATASET_DIR: Path = get_env("PSYCTL_DATASET_DIR", Path("./dataset"), Path)
STEERING_VECTOR_DIR: Path = get_env(
    "PSYCTL_STEERING_VECTOR_DIR", Path("./steering_vector"), Path
)
RESULTS_DIR: Path = get_env("PSYCTL_RESULTS_DIR", Path("./results"), Path)
CACHE_DIR: Path = get_env("PSYCTL_CACHE_DIR", Path("./temp"), Path)

# Logging settings
LOG_LEVEL: str = get_env("PSYCTL_LOG_LEVEL", "INFO")
LOG_FILE: Path | None = get_env("PSYCTL_LOG_FILE", None, Path)


def create_directories():
    """Create necessary directories."""
    directories: list[Path] = [
        OUTPUT_DIR,
        DATASET_DIR,
        STEERING_VECTOR_DIR,
        RESULTS_DIR,
        CACHE_DIR,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}")
            raise
