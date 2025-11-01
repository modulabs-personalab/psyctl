"""LLM loader and manager."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from psyctl.core.logger import get_logger

# Disable PyTorch compiler to avoid Triton issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


class LLMLoader:
    """Load and manage LLM models."""

    def __init__(self):
        self.models: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}
        self.logger = get_logger("llm_loader")

    def load_model(self, model_name: str, device: str | None = None, dtype: str | None = None) -> tuple:
        """Load model and tokenizer."""
        self.logger.info(f"Loading model: {model_name}")

        if model_name in self.models:
            self.logger.debug(
                f"Model {model_name} already loaded, returning cached version"
            )
            return self.models[model_name], self.tokenizers[model_name]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Auto-detected device: {device}")
        else:
            self.logger.info(f"Using specified device: {device}")

        if dtype is None:
            dtype = 'auto'
            trust_remote_code = False
        else:
            trust_remote_code = True

        try:
            # Load tokenizer
            self.logger.debug("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                self.logger.debug("Set pad_token to eos_token")

            # Load model
            self.logger.debug("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                device_map=device if device == "cuda" else None,
            )

            # Cache the loaded model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

            self.logger.info(f"Successfully loaded model: {model_name}")
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
