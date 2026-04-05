"""Layer analyzer for finding optimal steering target layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from psyctl.core.analyzers import ConsensusAnalyzer, SVMAnalyzer
from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger
from psyctl.core.steer_dataset_loader import SteerDatasetLoader
from psyctl.core.utils import validate_tokenizer_padding
from psyctl.models.llm_loader import LLMLoader


class LayerAnalyzer:
    """Analyze layers to find optimal steering target layers."""

    ANALYZERS: ClassVar[dict[str, type]] = {
        "svm": SVMAnalyzer,
        "consensus": ConsensusAnalyzer,
    }

    def __init__(self):
        """Initialize LayerAnalyzer."""
        self.logger = get_logger("layer_analyzer")
        self.llm_loader = LLMLoader()
        self.layer_accessor = LayerAccessor()
        self.dataset_loader = SteerDatasetLoader()

    def analyze_layers(
        self,
        layers: list[str],
        output_path: Path | None = None,
        model_name: str | None = None,
        model: nn.Module | None = None,
        tokenizer: AutoTokenizer | None = None,
        dataset_path: Path | str | None = None,
        dataset: list[dict] | None = None,
        batch_size: int | None = None,
        method: str = "svm",
        top_k: int = 5,
        **method_params,
    ) -> dict[str, Any]:
        """
        Analyze layers to find optimal steering target layers.

        Args:
            layers: List of layer paths (supports wildcards)
            output_path: Output JSON file path (optional, skip saving if None)
            model_name: Model name (optional if model provided)
            model: Pre-loaded model (optional if model_name provided)
            tokenizer: Pre-loaded tokenizer (optional if model_name provided)
            dataset_path: Dataset path (optional if dataset provided)
            dataset: Pre-loaded dataset (optional if dataset_path provided)
            batch_size: Batch size for inference
            method: Analysis method name ("svm" or "consensus", default: "svm")
            top_k: Number of top layers to report
            **method_params: Method-specific parameters

        Returns:
            Dictionary with analysis results

        Example:
            >>> analyzer = LayerAnalyzer()
            >>> # Using SVM analyzer
            >>> results = analyzer.analyze_layers(
            ...     layers=["model.layers[*].mlp"],
            ...     output_path=Path("./results/analysis.json"),
            ...     model_name="google/gemma-3-270m-it",
            ...     dataset_path=Path("./dataset/caa"),
            ...     method="svm",
            ...     top_k=5
            ... )
            >>> # Using Consensus analyzer (more robust)
            >>> results = analyzer.analyze_layers(
            ...     layers=["model.layers[*].mlp.down_proj"],
            ...     output_path=Path("./results/analysis.json"),
            ...     model_name="google/gemma-3-270m-it",
            ...     dataset_path=Path("./dataset/caa"),
            ...     method="consensus",
            ...     top_k=5
            ... )
        """
        # Validate parameters
        if model is not None and model_name is not None:
            raise ValueError("Cannot provide both 'model' and 'model_name'")
        if model is None and model_name is None:
            raise ValueError("Must provide either 'model' or 'model_name'")
        if model is not None and tokenizer is None:
            raise ValueError("Must provide 'tokenizer' when providing 'model'")
        if dataset is not None and dataset_path is not None:
            raise ValueError("Cannot provide both 'dataset' and 'dataset_path'")
        if dataset is None and dataset_path is None:
            raise ValueError("Must provide either 'dataset' or 'dataset_path'")

        # Determine model identifier
        if model_name:
            model_identifier = model_name
        else:
            try:
                model_identifier = model.config._name_or_path  # type: ignore[union-attr]
            except AttributeError:
                model_identifier = "unknown"

        self.logger.info(f"Analyzing layers for model: {model_identifier}")
        self.logger.info(f"Analysis method: {method}")

        # Load model if needed
        if model is None:
            self.logger.info("Loading model...")
            assert model_name is not None
            model, tokenizer = self.llm_loader.load_model(model_name)
        else:
            self.logger.info("Using pre-loaded model")
        assert model is not None
        assert tokenizer is not None

        # Validate tokenizer
        validate_tokenizer_padding(tokenizer)

        # Expand wildcard patterns
        self.logger.info(f"Expanding {len(layers)} layer patterns...")
        expanded_layers = self.layer_accessor.expand_layer_patterns(model, layers)
        self.logger.info(f"Expanded to {len(expanded_layers)} concrete layers")

        # Load dataset if needed
        if dataset is None:
            self.logger.info(f"Loading dataset from {dataset_path}...")
            assert dataset_path is not None
            dataset = self.dataset_loader.load(dataset_path)
        else:
            self.logger.info("Using pre-loaded dataset")

        # Get analyzer
        analyzer_class = self.ANALYZERS.get(method)
        if analyzer_class is None:
            raise ValueError(
                f"Unknown analysis method: {method}. "
                f"Available: {list(self.ANALYZERS.keys())}"
            )
        analyzer = analyzer_class(**method_params)

        # Collect activations for all layers at once (efficient!)
        self.logger.info(f"Collecting activations for {len(expanded_layers)} layers...")
        layer_activations = self._collect_all_activations(
            model=model,
            tokenizer=tokenizer,
            layer_paths=expanded_layers,
            dataset=dataset,
            batch_size=batch_size,
        )

        # Analyze each layer
        self.logger.info(f"Analyzing {len(expanded_layers)} layers...")
        results = []

        for layer_path in tqdm(expanded_layers, desc="Analyzing layers"):
            try:
                positive_acts = layer_activations[layer_path]["positive"]
                neutral_acts = layer_activations[layer_path]["neutral"]

                metrics = analyzer.analyze(positive_acts, neutral_acts)
                results.append({"layer": layer_path, "metrics": metrics})
            except Exception as e:
                self.logger.error(f"Failed to analyze layer {layer_path}: {e}")
                results.append({
                    "layer": layer_path,
                    "metrics": {"score": 0.0, "error": str(e)},
                })

        # Sort by score
        results.sort(key=lambda x: x["metrics"].get("score", 0.0), reverse=True)

        # Add rankings
        for rank, result in enumerate(results, 1):
            result["rank"] = rank

        # Prepare output
        output_data = {
            "model": model_identifier,
            "dataset": str(dataset_path) if dataset_path else "pre-loaded",
            "method": method,
            "total_layers": len(expanded_layers),
            "rankings": results,
            "top_k_layers": [r["layer"] for r in results[:top_k]],
        }

        # Save results if output_path is provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import json

            with Path(output_path).open("w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to {output_path}")
        else:
            self.logger.info("Output path not provided, skipping JSON save")

        return output_data

    def _collect_all_activations(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layer_paths: list[str],
        dataset: list[dict],
        batch_size: int | None = None,
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """
        Collect activations for all layers at once (efficient single-pass).

        Args:
            model: Model
            tokenizer: Tokenizer
            layer_paths: List of layer paths
            dataset: Dataset
            batch_size: Batch size for processing (default: 10)

        Returns:
            Dictionary mapping layer paths to {"positive": [...], "neutral": [...]}
        """
        if batch_size is None:
            batch_size = 10  # Default batch size

        self.logger.info(f"Using batch size: {batch_size}")
        model.eval()

        # Get all layer modules
        layer_modules = {}
        for layer_path in layer_paths:
            layer_module = self.layer_accessor.get_layer(model, layer_path)
            layer_modules[layer_path] = layer_module

        # Initialize storage
        layer_activations = {
            path: {"positive": [], "neutral": []} for path in layer_paths
        }

        # Collect positive activations (all layers, all samples at once)
        self.logger.info("Collecting positive activations...")

        # Create a custom hook that stores individual activations per sample
        activations_storage = {path: [] for path in layer_paths}

        def make_hook(layer_path):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                act = output.detach().cpu()  # [B, T, H]
                # Store each sample's last token activation
                for i in range(act.shape[0]):
                    if hasattr(hook, "attention_mask"):  # type: ignore[arg-type]
                        mask = hook.attention_mask[i].cpu()  # type: ignore[union-attr]
                        real_positions = torch.where(mask == 1)[0]
                        if len(real_positions) > 0:
                            last_pos = real_positions[-1].item()
                        else:
                            last_pos = -1
                    else:
                        last_pos = -1
                    activations_storage[layer_path].append(act[i, last_pos, :])

            return hook

        # Register hooks for all layers
        handles = []
        for layer_path, layer_module in layer_modules.items():
            hook = make_hook(layer_path)
            handle = layer_module.register_forward_hook(hook)
            handles.append((handle, hook))

        # Process positive samples in batches
        num_batches = (len(dataset) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(dataset), batch_size),
            total=num_batches,
            desc="Positive batches",
            leave=False,
        ):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            prompts = []
            for item in batch:
                situation = item.get("situation", item.get("question", ""))
                prompt = (
                    tokenizer.apply_chat_template(  # type: ignore[call-arg]
                        [{"role": "user", "content": situation}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + item["positive"]
                )
                prompts.append(prompt)

            inputs = tokenizer(  # type: ignore[call-arg]
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            # Set attention mask on hooks
            for _, hook in handles:
                hook.attention_mask = inputs["attention_mask"]

            with torch.inference_mode():
                _ = model(**inputs)

        # Remove hooks and store results
        for handle, _ in handles:
            handle.remove()

        for layer_path in layer_paths:
            layer_activations[layer_path]["positive"] = activations_storage[layer_path]

        # Reset storage for neutral
        activations_storage = {path: [] for path in layer_paths}

        # Collect neutral activations
        self.logger.info("Collecting neutral activations...")

        # Register hooks again
        handles = []
        for layer_path, layer_module in layer_modules.items():
            hook = make_hook(layer_path)
            handle = layer_module.register_forward_hook(hook)
            handles.append((handle, hook))

        # Process neutral samples in batches
        for i in tqdm(
            range(0, len(dataset), batch_size),
            total=num_batches,
            desc="Neutral batches",
            leave=False,
        ):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            prompts = []
            for item in batch:
                situation = item.get("situation", item.get("question", ""))
                prompt = (
                    tokenizer.apply_chat_template(  # type: ignore[call-arg]
                        [{"role": "user", "content": situation}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    + item["neutral"]
                )
                prompts.append(prompt)

            inputs = tokenizer(  # type: ignore[call-arg]
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            # Set attention mask on hooks
            for _, hook in handles:
                hook.attention_mask = inputs["attention_mask"]

            with torch.inference_mode():
                _ = model(**inputs)

        # Remove hooks and store results
        for handle, _ in handles:
            handle.remove()

        for layer_path in layer_paths:
            layer_activations[layer_path]["neutral"] = activations_storage[layer_path]

        self.logger.info(
            f"Collected activations: {len(dataset)} samples x {len(layer_paths)} layers"
        )

        return layer_activations

