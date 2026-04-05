"""Utility functions for psyctl."""

import json
from pathlib import Path
from typing import Any

from psyctl.core.logger import get_logger

logger = get_logger("utils")


def save_json(data: dict[str, Any], filepath: Path) -> None:
    """Save data to JSON file."""
    logger.debug(f"Saving JSON data to: {filepath}")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with Path(filepath).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"JSON data saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON data: {e}")
        raise


def load_json(filepath: Path) -> dict[str, Any]:
    """Load data from JSON file."""
    logger.debug(f"Loading JSON data from: {filepath}")

    try:
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file does not exist: {filepath}")

        with Path(filepath).open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"JSON data loaded successfully from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data: {e}")
        raise


def parse_personality_traits(personality_str: str) -> list[str]:
    """Parse personality traits string into list."""
    logger.debug(f"Parsing personality traits: {personality_str}")

    traits = [trait.strip() for trait in personality_str.split(",")]
    logger.debug(f"Parsed traits: {traits}")

    return traits


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    logger.debug(f"Validating model name: {model_name}")

    # Basic validation - can be extended
    is_valid = "/" in model_name and len(model_name) > 0

    if is_valid:
        logger.debug(f"Model name validation passed: {model_name}")
    else:
        logger.warning(f"Model name validation failed: {model_name}")

    return is_valid


def validate_hf_token() -> str:
    """
    Validate HuggingFace token from environment.

    Returns:
        str: Valid HF_TOKEN

    Raises:
        click.ClickException: If HF_TOKEN is not set with helpful message

    Example:
        >>> from psyctl.core.utils import validate_hf_token
        >>> token = validate_hf_token()
        >>> print(f"Token: {token[:4]}...")
    """
    import os

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is required for uploading to HuggingFace Hub.\n\n"
            "To set up your token:\n"
            "  1. Get token from https://huggingface.co/settings/tokens\n"
            "  2. Set environment variable:\n"
            '     - Windows PowerShell: $env:HF_TOKEN="hf_xxxx"\n'
            '     - Linux/macOS: export HF_TOKEN="hf_xxxx"\n'
            "  3. Or use: huggingface-cli login"
        )

    logger.debug(
        f"HF_TOKEN found: {token[:4]}...{token[-4:] if len(token) > 8 else '***'}"
    )
    return token


def validate_tokenizer_padding(tokenizer) -> str:
    """
    Validate and report tokenizer padding configuration.

    Checks the padding direction and logs information about compatibility
    with the activation extraction system.

    Args:
        tokenizer: HuggingFace tokenizer to validate

    Returns:
        Padding side: 'left' or 'right'

    Example:
        >>> from transformers import AutoTokenizer
        >>> from psyctl.core.utils import validate_tokenizer_padding
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        >>> padding_side = validate_tokenizer_padding(tokenizer)
        Tokenizer uses LEFT padding
          Position -1 always points to last real token (safe)
    """
    padding_side = getattr(tokenizer, "padding_side", "right")

    if padding_side == "left":
        logger.info(
            "Tokenizer uses LEFT padding\n"
            "  Position -1 always points to last real token (safe)"
        )
    else:
        logger.info(
            "Tokenizer uses RIGHT padding\n"
            "  This project uses attention masks to handle this correctly.\n"
            "  Activation extraction will find the last real token automatically."
        )

    return padding_side
