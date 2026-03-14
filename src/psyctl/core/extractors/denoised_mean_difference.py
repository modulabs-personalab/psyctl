"""Denoised Mean Difference extractor using PCA-based noise reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.hook_manager import ActivationHookManager


class DenoisedMeanDifferenceVectorExtractor(BaseVectorExtractor):
    """Extract steering vectors using PCA-denoised mean difference.

    Algorithm:
    1. Collect activations from positive and neutral prompts
    2. Compute raw steering vector (mean difference)
    3. Apply PCA to combined activations
    4. Project steering vector onto top-K principal components (95% variance)
    5. Reconstruct denoised steering vector
    """

    def __init__(self):
        """Initialize DenoisedMeanDifferenceVectorExtractor."""
        super().__init__(logger_name="denoised_mean_diff")
        self.hook_manager = ActivationHookManager()

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path | str | None = None,
        dataset: list[dict] | None = None,
        batch_size: int | None = None,
        normalize: bool = False,
        variance_threshold: float = 0.95,
        use_chat_template: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Extract PCA-denoised steering vectors from multiple layers.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths (e.g., ["model.layers[13].mlp.down_proj"])
            dataset_path: Path to steering dataset or HuggingFace dataset name (optional if dataset provided)
            dataset: Pre-loaded dataset as list of dicts (optional if dataset_path provided)
            batch_size: Batch size for inference (default: from config)
            normalize: Whether to normalize vectors to unit length
            variance_threshold: PCA variance threshold (default: 0.95 = keep 95% variance)
            use_chat_template: Whether to use chat template for prompt formatting (default: True)
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary mapping layer names to PCA-denoised steering vectors

        Raises:
            ValueError: If neither dataset_path nor dataset is provided, or both are provided

        Example:
            >>> extractor = DenoisedMeanDifferenceVectorExtractor()
            >>> vectors = extractor.extract(
            ...     model=model,
            ...     tokenizer=tokenizer,
            ...     layers=["model.layers[13].mlp.down_proj"],
            ...     dataset_path=Path("./dataset/steering"),
            ...     variance_threshold=0.95
            ... )
        """
        self._validate_dataset_params(dataset, dataset_path)
        batch_size = self._resolve_batch_size(batch_size)

        self.logger.info(
            f"Extracting PCA-enhanced steering vectors from {len(layers)} layers"
        )
        self.logger.info(
            f"Batch size: {batch_size}, Variance threshold: {variance_threshold}"
        )

        self._validate_layers(model, layers)
        dataset = self._load_dataset(dataset, dataset_path)

        positive_prompts, neutral_prompts = self.dataset_loader.create_prompts(
            dataset, tokenizer, format_type="index", use_chat_template=use_chat_template
        )

        self.logger.info(
            f"Loaded {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )

        layer_modules = self._resolve_layer_modules(model, layers)

        # 4. Collect positive activations
        self.logger.info("Collecting positive activations...")
        positive_activations = self._collect_all_activations(
            model, tokenizer, layer_modules, positive_prompts, batch_size, "positive"
        )

        # 5. Collect neutral activations
        self.logger.info("Collecting neutral activations...")
        neutral_activations = self._collect_all_activations(
            model, tokenizer, layer_modules, neutral_prompts, batch_size, "neutral"
        )

        # 6. Apply PCA-enhanced CAA to each layer
        self.logger.info("Computing PCA-denoised steering vectors...")
        steering_vectors = {}

        for layer_name in layers:
            positive_key = f"{layer_name}_positive"
            neutral_key = f"{layer_name}_neutral"

            if (
                positive_key not in positive_activations
                or neutral_key not in neutral_activations
            ):
                self.logger.warning(
                    f"Missing activations for layer '{layer_name}', skipping"
                )
                continue

            # Get all activation samples (not just means)
            pos_acts = positive_activations[positive_key]  # List of tensors [N, D]
            neu_acts = neutral_activations[neutral_key]  # List of tensors [N, D]

            # Apply PCA-enhanced CAA
            steering_vec = self._pca_enhanced_md(
                pos_acts, neu_acts, variance_threshold, layer_name
            )

            steering_vectors[layer_name] = steering_vec

            self.logger.info(
                f"Extracted PCA-denoised steering vector for '{layer_name}': "
                f"shape={steering_vec.shape}, norm={steering_vec.norm():.4f}"
            )

        self.logger.info(
            f"Successfully extracted {len(steering_vectors)} PCA-denoised steering vectors"
        )
        return self._normalize_vectors(steering_vectors, normalize)

    def _pca_enhanced_md(
        self,
        positive_acts: list[torch.Tensor],
        neutral_acts: list[torch.Tensor],
        variance_threshold: float,
        layer_name: str,
    ) -> torch.Tensor:
        """
        Apply PCA-enhanced MD algorithm.

        Args:
            positive_acts: List of positive activation tensors [N, D]
            neutral_acts: List of neutral activation tensors [N, D]
            variance_threshold: Cumulative variance threshold (e.g., 0.95)
            layer_name: Layer name for logging

        Returns:
            PCA-denoised steering vector [D]
        """
        # Stack activations into matrices
        # Convert to float32 first (sklearn doesn't support bfloat16)
        pos_matrix = torch.vstack(positive_acts).float().cpu().numpy()  # [N_pos, D]
        neu_matrix = torch.vstack(neutral_acts).float().cpu().numpy()  # [N_neu, D]

        # Compute raw steering vector (mean difference)
        pos_mean = pos_matrix.mean(axis=0)  # [D]
        neu_mean = neu_matrix.mean(axis=0)  # [D]
        raw_vector = pos_mean - neu_mean  # [D]

        self.logger.debug(
            f"Layer '{layer_name}': Raw vector norm={np.linalg.norm(raw_vector):.4f}"
        )

        # Combine activations for PCA
        combined = np.vstack([pos_matrix, neu_matrix])  # [N_pos + N_neu, D]
        n_samples, n_features = combined.shape

        # Determine number of components
        n_components = min(n_samples, n_features)

        self.logger.debug(
            f"Layer '{layer_name}': Fitting PCA with {n_components} components "
            f"on {n_samples} samples x {n_features} features"
        )

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(combined)

        # Find number of components to explain variance_threshold
        explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(explained_var_cumsum >= variance_threshold) + 1

        # Ensure we have at least a few components
        k = max(k, min(10, n_components))

        self.logger.info(
            f"Layer '{layer_name}': Using {k}/{n_components} PCA components "
            f"(explains {explained_var_cumsum[k - 1]:.3f} variance)"
        )

        # Get top-K principal components
        top_components = pca.components_[:k]  # [K, D]

        # Project raw vector onto top-K components and reconstruct
        # This removes noise in low-variance directions
        projection_coeffs = top_components @ raw_vector  # [K]
        denoised_vector = top_components.T @ projection_coeffs  # [D]

        # Calculate denoising metrics
        noise_removed = np.linalg.norm(raw_vector - denoised_vector)
        signal_preserved = np.linalg.norm(denoised_vector)

        self.logger.debug(
            f"Layer '{layer_name}': Denoised vector norm={signal_preserved:.4f}, "
            f"noise removed={noise_removed:.4f} "
            f"(SNR improvement: {signal_preserved / noise_removed:.2f}x)"
        )

        # Convert back to torch tensor
        device = positive_acts[0].device
        dtype = positive_acts[0].dtype
        return torch.from_numpy(denoised_vector).to(device=device, dtype=dtype)

    def _collect_all_activations(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_modules: dict[str, nn.Module],
        prompts: list[str],
        batch_size: int,
        suffix: str,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Collect ALL activation samples (not just means) from multiple layers.

        Args:
            model: Language model
            tokenizer: Tokenizer
            layer_modules: Dictionary of layer name → module
            prompts: List of prompts to process
            batch_size: Batch size for inference
            suffix: Suffix for layer names (e.g., "positive" or "neutral")

        Returns:
            Dictionary mapping layer names to lists of activation tensors
            Example: {"layer_name_positive": [tensor1, tensor2, ...]}
        """
        # Storage for all activations
        all_activations = {f"{name}_{suffix}": [] for name in layer_modules}

        # Register hooks that collect ALL samples
        hook_handles = []

        for name, module in layer_modules.items():
            key = f"{name}_{suffix}"

            def make_hook(storage_key):
                def hook_fn(module, input, output):
                    # Extract hidden states
                    hidden_states = output[0] if isinstance(output, tuple) else output

                    # Take last token activation for each sample in batch
                    # Shape: [batch_size, seq_len, hidden_dim]
                    batch_acts = (
                        hidden_states[:, -1, :].detach().cpu()
                    )  # [batch_size, D]

                    # Store each sample separately
                    for i in range(batch_acts.size(0)):
                        all_activations[storage_key].append(batch_acts[i])  # [D]

                return hook_fn

            handle = module.register_forward_hook(make_hook(key))
            hook_handles.append(handle)

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

                    # Forward pass (hooks will collect activations)
                    _ = model(**inputs)

        finally:
            # Remove all hooks
            for handle in hook_handles:
                handle.remove()

        # Log statistics
        for key in all_activations:
            n_samples = len(all_activations[key])
            self.logger.debug(f"Collected {n_samples} activation samples from '{key}'")

        return all_activations
