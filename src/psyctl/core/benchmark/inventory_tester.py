"""Inventory tester for measuring personality changes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from psyctl.core.benchmark.layer_resolver import LayerResolver
from psyctl.core.logger import get_logger
from psyctl.core.benchmark.logprob_scorer import LogProbScorer
from psyctl.core.steering_applier import SteeringApplier
from psyctl.data.inventories.ipip_neo import IPIPNEO
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore


class InventoryTester:
    """Test personality changes using psychological inventories."""

    def __init__(self):
        self.logger = get_logger("inventory_tester")
        self.loader = LLMLoader()
        self.vector_store = VectorStore()
        self.applier = SteeringApplier()
        
        # Load layer groups configuration
        from psyctl.data import benchmark_settings
        try:
            self.layer_groups_config = benchmark_settings.get_layer_groups()
        except Exception as e:
            self.logger.warning(f"Could not load layer groups config: {e}")
            self.layer_groups_config = {}

    def test_inventory(
        self,
        model: str,
        steering_vector_path: Path | None,
        inventory_name: str = "ipip_neo_120",
        steering_strength: float = 1.0,
        device: str = "auto",
        target_trait: str | None = None,
        layer_spec: str | None = None,
    ) -> dict[str, Any]:
        """
        Run inventory test and return results.

        Args:
            model: Model name or path
            steering_vector_path: Path to steering vector file (None for baseline only)
            inventory_name: Name of inventory to use
            steering_strength: Steering strength multiplier
            device: Device to use
            target_trait: Specific trait to test (N/E/O/A/C or full name). Tests all if None.
            layer_spec: Layer specification string (e.g., "15", "0-5", "middle", "early,late").
                       Uses all layers if None. Supports:
                       - Direct numbers: "0,5,10"
                       - Ranges: "0-5", "10-20"
                       - Keywords: "all", "early", "middle", "late"
                       - Combinations: "0,10-15,late"

        Returns:
            Dictionary with test results
        """
        # Reduced logging for cleaner output
        self.logger.debug(f"Running inventory test for model: {model}")
        self.logger.debug(f"Inventory: {inventory_name}")
        if target_trait:
            self.logger.debug(f"Target trait: {target_trait}")
        if steering_vector_path:
            self.logger.debug(f"Steering vector path: {steering_vector_path}")
            self.logger.debug(f"Steering strength: {steering_strength}")
            if layer_spec:
                self.logger.debug(f"Layer spec: {layer_spec}")

        try:
            # 1. Load model
            self.logger.info("Loading model and tokenizer...")
            model_obj, tokenizer = self.loader.load_model(model, device=device)

            # 2. Load inventory and questions
            self.logger.info(f"Loading inventory: {inventory_name}")
            version = inventory_name.split("_")[-1] if "_" in inventory_name else "120"
            inventory = IPIPNEO(version=version)
            
            # Get questions (filtered by trait if specified)
            questions = inventory.get_questions(trait=target_trait)
            if target_trait:
                self.logger.info(
                    f"Loaded {len(questions)} questions for trait '{target_trait}'"
                )
            else:
                self.logger.info(f"Loaded {len(questions)} questions (all traits)")

            # 3. Evaluate baseline (no steering)
            self.logger.info("Evaluating baseline responses...")
            baseline_responses = self._evaluate_questions(
                model_obj, tokenizer, questions
            )
            baseline_scores = self._calculate_domain_scores(
                baseline_responses, inventory
            )

            # 4. Apply steering and evaluate (if provided)
            steered_scores = None
            resolved_layers = None
            if steering_vector_path:
                if not steering_vector_path.exists():
                    raise FileNotFoundError(
                        f"Steering vector file does not exist: {steering_vector_path}"
                    )
                
                # Load steering vectors to get available layers
                steering_vectors, metadata = self.vector_store.load_multi_layer(
                    steering_vector_path
                )
                available_layers = list(steering_vectors.keys())
                
                # Resolve layer specification
                if layer_spec:
                    resolved_layers = LayerResolver.resolve_layer_spec(
                        layer_spec, available_layers, self.layer_groups_config
                    )
                    layer_desc = LayerResolver.describe_layer_spec(
                        layer_spec, self.layer_groups_config
                    )
                    self.logger.info(
                        f"Resolved layer spec '{layer_desc}' to {len(resolved_layers)} layers: "
                        f"{[LayerResolver._extract_layer_number(l) for l in resolved_layers]}"
                    )
                
                self.logger.info("Applying steering hooks to model...")
                # Use existing apply method with prompt_length=0 for all tokens
                model_obj, tokenizer = self.applier.get_steering_applied_model(
                    steering_vector_path=steering_vector_path,
                    model=model_obj,
                    tokenizer=tokenizer,
                    strength=steering_strength,
                    prompt_length=0,  # Apply to all tokens
                    orthogonal=False,
                    layers=resolved_layers,  # Filter to specific layers if provided
                )
                self.logger.info("Steering hooks applied successfully")
                
                # Evaluate with steering
                self.logger.info("Evaluating steered responses...")
                steered_responses = self._evaluate_questions(
                    model_obj, tokenizer, questions
                )
                steered_scores = self._calculate_domain_scores(
                    steered_responses, inventory
                )
                
                # Clean up steering hooks
                if hasattr(model_obj, "remove_steering"):
                    model_obj.remove_steering()  # type: ignore[attr-defined]
                    self.logger.info("Removed steering hooks")

            # 6. Compile results
            results = {
                "model": model,
                "inventory": inventory_name,
                "steering_vector": str(steering_vector_path) if steering_vector_path else None,
                "steering_strength": steering_strength,
                "layer_spec": layer_spec,
                "resolved_layers": [LayerResolver._extract_layer_number(l) for l in resolved_layers] if resolved_layers else None,
                "baseline": baseline_scores,
                "steered": steered_scores,
                "comparison": self._compare_scores(baseline_scores, steered_scores)
                if steered_scores
                else None,
            }

            self.logger.success("Inventory test completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Failed to run inventory test: {e}")
            raise

    def _evaluate_questions(
        self,
        model: Any,
        tokenizer: Any,
        questions: list[dict[str, Any]],
    ) -> dict[str, list[float]]:
        """
        Evaluate all questions and return domain-organized scores.

        Args:
            model: Model object
            tokenizer: Tokenizer
            questions: List of question dicts

        Returns:
            Dict mapping domain codes to lists of scores
        """
        domain_scores: dict[str, list[float]] = {}

        for question in tqdm(questions, desc="Evaluating", ncols=80, unit="q"):
            prompt = self._create_prompt(question["text"])
            score, confidence = self._get_score_from_logits(
                model,
                tokenizer,
                prompt,
            )

            # Reverse scoring for minus-keyed items
            if question["keyed"] == "minus":
                score = 6.0 - score

            # Organize by domain
            domain = question["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(score)

        return domain_scores

    def _create_prompt(self, question_text: str) -> str:
        """Create prompt for a single question."""
        prompt = f"""Rate how accurately this statement describes you on a scale of 1 to 5:

Statement: "{question_text}"

Scale:
1 = Very Inaccurate
2 = Moderately Inaccurate
3 = Neither Accurate Nor Inaccurate
4 = Moderately Accurate
5 = Very Accurate

Your rating (1-5):"""
        return prompt

    def _get_score_from_logits(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> tuple[float, float]:
        """
        Get weighted score from model logits using log probability.

        Args:
            model: Model object
            tokenizer: Tokenizer
            prompt: Input prompt

        Returns:
            (weighted_score, confidence)
        """
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Forward pass (steering hooks already applied if needed)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits for next token
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Calculate weighted score
        weighted_score, confidence = self._calculate_weighted_score(
            logits, tokenizer
        )

        return weighted_score, confidence

    def _calculate_weighted_score(
        self, logits: torch.Tensor, tokenizer: Any
    ) -> tuple[float, float]:
        """
        Calculate weighted score from logits using softmax.

        Uses shared LogProbScorer for consistency across all evaluations.

        Args:
            logits: Model output logits (vocab_size,)
            tokenizer: Tokenizer

        Returns:
            (weighted_score, confidence)
        """
        return LogProbScorer.calculate_weighted_score(
            logits, tokenizer, score_range=(1, 5)
        )

    def _calculate_domain_scores(
        self, domain_responses: dict[str, list[float]], inventory: IPIPNEO
    ) -> dict[str, dict[str, float]]:
        """
        Calculate domain scores from responses.

        Args:
            domain_responses: Dict mapping domains to score lists
            inventory: Inventory object

        Returns:
            Dict with domain scores and statistics
        """
        return inventory.calculate_scores(domain_responses)

    def _compare_scores(
        self,
        baseline: dict[str, dict[str, float]],
        steered: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Compare baseline and steered scores.

        Args:
            baseline: Baseline scores
            steered: Steered scores

        Returns:
            Comparison dict with changes
        """
        comparison = {}

        for domain in baseline.keys():
            if domain in steered:
                baseline_raw = baseline[domain]["raw_score"]
                steered_raw = steered[domain]["raw_score"]
                change = steered_raw - baseline_raw
                pct_change = (
                    (change / baseline_raw) * 100 if baseline_raw != 0 else 0.0
                )

                comparison[domain] = {
                    "domain_name": baseline[domain]["domain_name"],
                    "baseline_raw": baseline_raw,
                    "steered_raw": steered_raw,
                    "change": change,
                    "percent_change": pct_change,
                    "baseline_z": baseline[domain]["z_score"],
                    "steered_z": steered[domain]["z_score"],
                    "z_change": steered[domain]["z_score"]
                    - baseline[domain]["z_score"],
                }

        return comparison

