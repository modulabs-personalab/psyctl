"""Mean Difference Activation Vector extractor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from psyctl.config import INFERENCE_BATCH_SIZE
from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.hook_manager import ActivationHookManager
from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger
from psyctl.core.steer_dataset_loader import SteerDatasetLoader


class MeanDifferenceActivationVectorExtractor(BaseVectorExtractor):
    """
    Extract steering vectors using mean difference of activations.

    This extractor computes steering vectors by taking the mean difference
    between activations from positive and neutral personality prompts.
    The extracted vectors are later applied using CAA (Contrastive Activation Addition).

    Algorithm:
    1. Collect activations from positive prompts → compute mean
    2. Collect activations from neutral prompts → compute mean
    3. Steering vector = mean(positive) - mean(neutral)

    Attributes:
        hook_manager: Manager for forward hooks
        dataset_loader: Loader for steering dataset
        layer_accessor: Accessor for dynamic layer retrieval
        logger: Logger instance
    """

    def __init__(self):
        """Initialize MeanDifferenceActivationVectorExtractor."""
        self.hook_manager = ActivationHookManager()
        self.dataset_loader = SteerDatasetLoader()
        self.layer_accessor = LayerAccessor()
        self.logger = get_logger("mdav_extractor")

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path | str | None = None,
        dataset: list[dict] | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
        use_chat_template: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract steering vectors from multiple layers simultaneously.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths (e.g., ["model.layers[13].mlp.down_proj"])
            dataset_path: Path to steering dataset or HuggingFace dataset name (optional if dataset provided)
            dataset: Pre-loaded dataset as list of dicts (optional if dataset_path provided)
            batch_size: Batch size for inference (default: from config)
            normalize: Whether to normalize vectors to unit length
            use_chat_template: Whether to use chat template for prompt formatting (default: True)
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary mapping layer names to steering vectors
            {
                "model.layers[13].mlp.down_proj": tensor(...),
                "model.layers[14].mlp.down_proj": tensor(...)
            }

        Raises:
            ValueError: If neither dataset_path nor dataset is provided, or both are provided

        Example:
            >>> extractor = MeanDifferenceActivationVectorExtractor()
            >>> # Using dataset_path
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"],
            ...     dataset_path=Path("./dataset/caa"),
            ...     batch_size=16
            ... )
            >>> # Using pre-loaded dataset
            >>> dataset = [{"question": "...", "positive": "...", "neutral": "..."}]
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp.down_proj"],
            ...     dataset=dataset,
            ...     batch_size=16
            ... )
        """
        # Validate dataset parameters
        if dataset is not None and dataset_path is not None:
            raise ValueError(
                "Cannot provide both 'dataset' and 'dataset_path'. Choose one."
            )
        if dataset is None and dataset_path is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_path'.")

        if batch_size is None:
            batch_size = INFERENCE_BATCH_SIZE

        self.logger.info(f"Extracting steering vectors from {len(layers)} layers")
        self.logger.info(f"Dataset: {'pre-loaded' if dataset else dataset_path}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Normalize: {normalize}")
        self.logger.info(f"Use chat template: {use_chat_template}")

        # 1. Validate layers
        self.logger.info("Validating layer paths...")
        if not self.layer_accessor.validate_layers(model, layers):
            raise ValueError("Some layer paths are invalid")

        # 2. Load dataset if not provided
        if dataset is None:
            self.logger.info("Loading dataset...")
            dataset = self.dataset_loader.load(dataset_path)  # type: ignore[arg-type]
        else:
            self.logger.info("Using pre-loaded dataset...")

        positive_prompts, neutral_prompts = self.dataset_loader.create_prompts(
            dataset, tokenizer, format_type="index", use_chat_template=use_chat_template
        )

        self.logger.info(
            f"Loaded {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )

        # 3. Get layer modules
        layer_modules = {}
        for layer_str in layers:
            layer_module = self.layer_accessor.get_layer(model, layer_str)
            layer_modules[layer_str] = layer_module

        # 4. Collect positive activations
        self.logger.info("Collecting positive activations...")
        self._collect_activations(
            model, tokenizer, layer_modules, positive_prompts, batch_size, "positive"
        )
        positive_means = self.hook_manager.get_mean_activations()

        # 5. Collect neutral activations
        self.logger.info("Collecting neutral activations...")
        self.hook_manager.reset()
        self._collect_activations(
            model, tokenizer, layer_modules, neutral_prompts, batch_size, "neutral"
        )
        neutral_means = self.hook_manager.get_mean_activations()

        # 6. Compute steering vectors
        self.logger.info("Computing steering vectors...")
        steering_vectors = {}

        for layer_name in layers:
            positive_key = f"{layer_name}_positive"
            neutral_key = f"{layer_name}_neutral"

            if positive_key not in positive_means or neutral_key not in neutral_means:
                self.logger.warning(
                    f"Missing activations for layer '{layer_name}', skipping"
                )
                continue

            steering_vec = positive_means[positive_key] - neutral_means[neutral_key]

            if normalize:
                norm = steering_vec.norm()
                if norm > 1e-8:
                    steering_vec = steering_vec / norm
                    self.logger.debug(f"Normalized vector for '{layer_name}'")
                else:
                    self.logger.warning(
                        f"Vector for '{layer_name}' has near-zero norm, skipping normalization"
                    )

            steering_vectors[layer_name] = steering_vec

            self.logger.info(
                f"Extracted steering vector for '{layer_name}': "
                f"shape={steering_vec.shape}, norm={steering_vec.norm():.4f}"
            )

        self.logger.info(
            f"Successfully extracted {len(steering_vectors)} steering vectors"
        )
        return steering_vectors

    def _collect_activations(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_modules: dict[str, nn.Module],
        prompts: list[str],
        batch_size: int,
        suffix: str,
    ) -> None:
        """
        Collect activations from multiple layers in one forward pass.

        Uses attention mask to correctly identify the last real token in each
        sequence, handling both left-padded and right-padded models.

        Args:
            model: Language model
            tokenizer: Tokenizer
            layer_modules: Dictionary of layer name → module
            prompts: List of prompts to process
            batch_size: Batch size for inference
            suffix: Suffix for layer names (e.g., "positive" or "neutral")
        """
        # Register hooks with suffix
        suffixed_layers = {
            f"{name}_{suffix}": module for name, module in layer_modules.items()
        }
        self.hook_manager.register_hooks(suffixed_layers)

        try:
            num_batches = (len(prompts) + batch_size - 1) // batch_size

            with torch.inference_mode():
                for batch_prompts in tqdm(
                    self.dataset_loader.get_batch_iterator(prompts, batch_size),
                    desc=f"Collecting {suffix} activations",
                    total=num_batches,
                ):
                    # Tokenize batch with padding
                    inputs = tokenizer(  # type: ignore[call-arg]
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=False,
                    )

                    # Move to model device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Set attention mask before forward pass
                    # This allows hooks to find the last REAL token (not padding)
                    self.hook_manager.set_attention_mask(inputs["attention_mask"])

                    # Forward pass (hooks will collect activations)
                    _ = model(**inputs)

        finally:
            self.hook_manager.remove_all_hooks()

        # Log statistics
        stats = self.hook_manager.get_activation_stats()
        for layer_name, layer_stats in stats.items():
            self.logger.debug(
                f"Collected {layer_stats['count']} activations from '{layer_name}'"
            )
