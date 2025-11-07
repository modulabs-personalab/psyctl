"""Steering vector extractor using various methods."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch import nn
from transformers import AutoTokenizer

from psyctl.core.extractors import (
    BiPOVectorExtractor,
    DenoisedMeanDifferenceVectorExtractor,
    MeanDifferenceActivationVectorExtractor,
)
from psyctl.core.logger import get_logger
from psyctl.core.utils import validate_tokenizer_padding
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore


class SteeringExtractor:
    """Extract steering vectors using various methods."""

    EXTRACTORS: ClassVar[dict[str, type]] = {
        "mean_diff": MeanDifferenceActivationVectorExtractor,
        "bipo": BiPOVectorExtractor,
        "denoised_mean_diff": DenoisedMeanDifferenceVectorExtractor,
    }

    def __init__(self):
        self.logger = get_logger("steering_extractor")
        self.llm_loader = LLMLoader()
        self.vector_store = VectorStore()

    def extract_steering_vector(
        self,
        layers: list[str],
        output_path: Path,
        model_name: str | None = None,
        model: nn.Module | None = None,
        tokenizer: AutoTokenizer | None = None,
        dataset_path: Path | str | None = None,
        dataset: list[dict] | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
        method: str = "mean_diff",
        **method_params,
    ) -> dict[str, torch.Tensor]:
        """
        Extract steering vectors using various methods.

        Args:
            layers: List of layer paths to extract from
            output_path: Output file path for safetensors
            model_name: Hugging Face model identifier (optional if model provided)
            model: Pre-loaded model (optional if model_name provided)
            tokenizer: Pre-loaded tokenizer (optional if model_name provided)
            dataset_path: Path to steering dataset or HuggingFace dataset name (optional if dataset provided)
            dataset: Pre-loaded dataset as list of dicts (optional if dataset_path provided)
            batch_size: Batch size for inference (optional)
            normalize: Whether to normalize vectors to unit length
            method: Extraction method name (default: "mean_diff")
            **method_params: Additional method-specific parameters

        Returns:
            Dictionary mapping layer names to steering vectors

        Raises:
            ValueError: If neither model_name nor model is provided, or both are provided
            ValueError: If neither dataset_path nor dataset is provided, or both are provided

        Examples:
            >>> # Example 1: Using model_name and dataset_path (original usage)
            >>> extractor = SteeringExtractor()
            >>> vectors = extractor.extract_steering_vector(
            ...     model_name="meta-llama/Llama-3.2-3B-Instruct",
            ...     layers=["model.layers[13].mlp.down_proj"],
            ...     dataset_path=Path("./dataset/steering"),
            ...     output_path=Path("./out.safetensors")
            ... )

            >>> # Example 2: Using pre-loaded model
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
            >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            >>> vectors = extractor.extract_steering_vector(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers.13.mlp.down_proj"],
            ...     dataset_path=Path("./dataset/steering"),
            ...     output_path=Path("./out.safetensors"),
            ...     method="mean_diff"
            ... )

            >>> # Example 3: Using pre-loaded dataset
            >>> from datasets import load_dataset
            >>> hf_dataset = load_dataset("CaveduckAI/steer-personality-rudeness-ko", split="train")
            >>> dataset = [{"question": item["question"], "positive": item["positive"], "neutral": item["neutral"]} for item in hf_dataset]
            >>> vectors = extractor.extract_steering_vector(
            ...     model_name="google/gemma-2-2b-it",
            ...     layers=["model.layers.13.mlp.down_proj"],
            ...     dataset=dataset,
            ...     output_path=Path("./out.safetensors")
            ... )

            >>> # Example 4: Using both pre-loaded model and dataset
            >>> vectors = extractor.extract_steering_vector(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     dataset=dataset,
            ...     layers=["model.layers.13.mlp.down_proj"],
            ...     output_path=Path("./out.safetensors")
            ... )

            >>> # Example 5: Using denoised mean difference method
            >>> vectors = extractor.extract_steering_vector(
            ...     model_name="google/gemma-2-2b-it",
            ...     layers=["model.layers.13.mlp.down_proj"],
            ...     dataset_path=Path("./dataset/steering"),
            ...     output_path=Path("./out.safetensors"),
            ...     method="denoised_mean_diff",
            ...     variance_threshold=0.95
            ... )
        """
        # Validate model parameters
        if model is not None and model_name is not None:
            raise ValueError(
                "Cannot provide both 'model' and 'model_name'. Choose one."
            )
        if model is None and model_name is None:
            raise ValueError("Must provide either 'model' or 'model_name'.")
        if model is not None and tokenizer is None:
            raise ValueError("Must provide 'tokenizer' when providing 'model'.")

        # Validate dataset parameters
        if dataset is not None and dataset_path is not None:
            raise ValueError(
                "Cannot provide both 'dataset' and 'dataset_path'. Choose one."
            )
        if dataset is None and dataset_path is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_path'.")

        # Determine model identifier for logging and metadata
        if model_name:
            model_identifier = model_name
        else:
            # Try to get model name from config
            try:
                model_identifier = model.config._name_or_path  # type: ignore[union-attr]
            except AttributeError:
                model_identifier = "unknown"

        self.logger.info(f"Extracting steering vectors for model: {model_identifier}")
        self.logger.info(f"Target layers: {layers}")
        self.logger.info(f"Dataset: {'pre-loaded' if dataset else dataset_path}")
        self.logger.info(f"Output path: {output_path}")
        self.logger.info(f"Method: {method}")

        try:
            # Validate dataset_path if provided
            if dataset_path is not None:
                if isinstance(dataset_path, str):
                    # Check if it's a HuggingFace dataset name (contains '/')
                    if "/" in dataset_path:
                        self.logger.info(f"Using HuggingFace dataset: {dataset_path}")
                    else:
                        # It's a local path string, convert to Path
                        dataset_path = Path(dataset_path)
                        if not dataset_path.exists():
                            raise FileNotFoundError(
                                f"Dataset path does not exist: {dataset_path}"
                            )
                elif isinstance(dataset_path, Path) and not dataset_path.exists():
                    # Validate local path
                    raise FileNotFoundError(
                        f"Dataset path does not exist: {dataset_path}"
                    )

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_path.parent}")

            # 1. Load model if not provided
            if model is None:
                self.logger.info("Loading model...")
                assert model_name is not None
                model, tokenizer = self.llm_loader.load_model(model_name)
            else:
                self.logger.info("Using pre-loaded model")

            # 1.5. Validate tokenizer padding configuration
            validate_tokenizer_padding(tokenizer)

            # 2. Get extractor
            extractor_class = self.EXTRACTORS.get(method)
            if extractor_class is None:
                raise ValueError(
                    f"Unknown extraction method: {method}. "
                    f"Available methods: {list(self.EXTRACTORS.keys())}"
                )

            extractor = extractor_class()

            # 3. Extract steering vectors
            self.logger.info(f"Extracting steering vectors using {method}...")
            steering_vectors = extractor.extract(
                model=model,
                tokenizer=tokenizer,
                layers=layers,
                dataset_path=dataset_path,
                dataset=dataset,
                batch_size=batch_size,
                normalize=normalize,
                **method_params,
            )

            # 4. Prepare metadata
            metadata = {
                "model": model_identifier,
                "method": method,
                "layers": layers,
                "dataset_path": str(dataset_path) if dataset_path else "pre-loaded",
                "num_layers": len(layers),
                "normalized": normalize,
            }

            # Add dataset info if available
            if dataset:
                metadata["dataset_samples"] = len(dataset)
            elif dataset_path:
                try:
                    from psyctl.core.steer_dataset_loader import SteerDatasetLoader

                    loader = SteerDatasetLoader()
                    dataset_info = loader.get_dataset_info(dataset_path)  # type: ignore[arg-type]
                    metadata["dataset_samples"] = dataset_info["num_samples"]
                except Exception as e:
                    self.logger.debug(f"Could not get dataset info: {e}")

            # 5. Save steering vectors
            self.logger.info("Saving steering vectors...")
            self.vector_store.save_multi_layer(
                vectors=steering_vectors, output_path=output_path, metadata=metadata
            )

            self.logger.info(
                f"Extracted and saved {len(steering_vectors)} steering vectors to {output_path}"
            )

            return steering_vectors

        except Exception as e:
            self.logger.error(f"Failed to extract steering vector: {e}")
            raise
