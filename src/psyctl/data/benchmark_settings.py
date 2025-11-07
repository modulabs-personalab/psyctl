"""Benchmark configuration loader.

This module provides utilities to load settings from benchmark_config.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Config file location
BENCHMARK_CONFIG_FILE = Path(__file__).parent / "benchmark_config.json"

# Global config cache
_config_cache: dict[str, Any] | None = None


def load_benchmark_config(force_reload: bool = False) -> dict[str, Any]:
    """
    Load benchmark configuration from JSON file.
    
    Args:
        force_reload: Force reload from file (ignore cache)
        
    Returns:
        Full configuration dictionary
    """
    global _config_cache
    
    if _config_cache is not None and not force_reload:
        return _config_cache
    
    with open(BENCHMARK_CONFIG_FILE, encoding="utf-8") as f:
        _config_cache = json.load(f)
    
    return _config_cache


def get_setting(
    path: str, 
    default: Any = None, 
    value_type: type | None = None
) -> Any:
    """
    Get configuration value by path with environment variable override.
    
    Args:
        path: Dot-separated path (e.g., "system_settings.device.default")
        default: Default value if not found
        value_type: Type to cast the value to
        
    Returns:
        Configuration value (env var takes precedence)
        
    Example:
        >>> get_setting("system_settings.device.default", "cpu", str)
        "cuda"  # if PSYCTL_BENCHMARK_DEVICE=cuda is set
    """
    # Check environment variable first
    env_key = f"PSYCTL_BENCHMARK_{path.replace('.', '_').upper()}"
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        if value_type is not None:
            if value_type == bool:
                return env_value.lower() in ("true", "1", "yes")
            elif value_type == Path:
                return Path(env_value)
            return value_type(env_value)
        return env_value
    
    # Get from config file
    cfg = load_benchmark_config()
    parts = path.split(".")
    value = cfg
    
    try:
        for part in parts:
            value = value[part]
        
        if value_type is not None:
            if value_type == Path:
                return Path(value).expanduser()
            return value_type(value)
        
        if isinstance(value, str) and value.startswith("~/"):
            return Path(value).expanduser()
        
        return value
    except (KeyError, TypeError):
        return default


# Convenience accessors
def get_results_dir() -> Path:
    """Get results directory path."""
    return get_setting("system_settings.paths.results_dir", "./results", Path)


def get_default_device() -> str:
    """Get default device setting."""
    return get_setting("system_settings.device.default", "auto", str)


def get_judge_models() -> dict[str, Any]:
    """Get all judge model configurations."""
    return load_benchmark_config().get("judge_models", {})


def get_layer_groups() -> dict[str, Any]:
    """Get all layer group configurations."""
    return load_benchmark_config().get("layer_groups", {})


def get_prompts() -> dict[str, str]:
    """Get all prompt templates."""
    return load_benchmark_config().get("prompts", {})


def get_default_questions() -> dict[str, list[str]]:
    """Get default questions for each trait."""
    return load_benchmark_config().get("default_questions", {})


def get_inventories() -> dict[str, Any]:
    """Get all inventory configurations."""
    return load_benchmark_config().get("inventories", {})


def get_evaluation_settings() -> dict[str, Any]:
    """Get all evaluation settings."""
    return load_benchmark_config().get("evaluation_settings", {})


def get_judge_default_model() -> str:
    """Get default judge model name."""
    return get_setting("evaluation_settings.judge.default_model", "local-default", str)


def get_judge_max_tokens() -> int:
    """Get judge max tokens."""
    return get_setting("evaluation_settings.judge.max_tokens", 150, int)


def get_judge_temperature() -> float:
    """Get judge temperature."""
    return get_setting("evaluation_settings.judge.temperature", 0.7, float)


def get_default_num_questions() -> int:
    """Get default number of questions."""
    return get_setting("evaluation_settings.judge.default_num_questions", 8, int)


