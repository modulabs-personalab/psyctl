"""Base class for steering vector extraction methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from psyctl.config import INFERENCE_BATCH_SIZE
from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger
from psyctl.core.steer_dataset_loader import SteerDatasetLoader


class BaseVectorExtractor(ABC):
    """Base class for steering vector extraction methods.

    Provides shared infrastructure for dataset loading, layer validation,
    and vector normalization. Subclasses implement extract() with their
    specific algorithm.
    """

    def __init__(self, logger_name: str | None = None):
        """Initialize shared components.

        Args:
            logger_name: Custom logger name. Defaults to class name.
        """
        self.dataset_loader = SteerDatasetLoader()
        self.layer_accessor = LayerAccessor()
        self.logger = get_logger(logger_name or type(self).__name__)

    def _validate_dataset_params(
        self,
        dataset: list[dict] | None,
        dataset_path: Path | str | None,
    ) -> None:
        """Validate mutual exclusion of dataset and dataset_path."""
        if dataset is not None and dataset_path is not None:
            raise ValueError(
                "Cannot provide both 'dataset' and 'dataset_path'. Choose one."
            )
        if dataset is None and dataset_path is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_path'.")

    def _validate_layers(self, model: nn.Module, layers: list[str]) -> None:
        """Validate that all layer paths exist in the model."""
        self.logger.info("Validating layer paths...")
        if not self.layer_accessor.validate_layers(model, layers):
            raise ValueError("Some layer paths are invalid")

    def _resolve_layer_modules(
        self, model: nn.Module, layers: list[str]
    ) -> dict[str, nn.Module]:
        """Resolve layer string paths to nn.Module objects."""
        layer_modules: dict[str, nn.Module] = {}
        for layer_str in layers:
            layer_modules[layer_str] = self.layer_accessor.get_layer(model, layer_str)
        return layer_modules

    def _load_dataset(
        self,
        dataset: list[dict] | None,
        dataset_path: Path | str | None,
    ) -> list[dict]:
        """Load dataset from path or return pre-loaded dataset."""
        if dataset is not None:
            self.logger.info("Using pre-loaded dataset...")
            return dataset
        self.logger.info("Loading dataset...")
        return self.dataset_loader.load(dataset_path)  # type: ignore[arg-type]

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        """Return provided batch size or default from config."""
        return batch_size if batch_size is not None else INFERENCE_BATCH_SIZE

    def _normalize_vectors(
        self,
        steering_vectors: dict[str, torch.Tensor],
        normalize: bool,
    ) -> dict[str, torch.Tensor]:
        """Optionally normalize vectors to unit length."""
        if not normalize:
            return steering_vectors
        result: dict[str, torch.Tensor] = {}
        for layer_name, vec in steering_vectors.items():
            norm = vec.norm()
            if norm > 1e-8:
                result[layer_name] = vec / norm
                self.logger.debug(f"Normalized vector for '{layer_name}'")
            else:
                result[layer_name] = vec
                self.logger.warning(
                    f"Vector for '{layer_name}' has near-zero norm, "
                    "skipping normalization"
                )
        return result

    @abstractmethod
    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract steering vectors from specified layers.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths to extract from
            dataset_path: Path to dataset
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping layer names to steering vectors
        """
        raise NotImplementedError("Subclasses must implement extract() method")
