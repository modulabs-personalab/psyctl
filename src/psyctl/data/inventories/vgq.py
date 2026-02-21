"""VGQ (Victim Gaslighting Questionnaire) implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("vgq")
class VGQ(BaseInventory):
    """VGQ (Victim Gaslighting Questionnaire) - Gaslighting victimization.

    A self-report measure of the victim's feelings, beliefs and behaviours
    due to gaslighting. Based on Shuja & Aqeel (2021).

    Single domain:
        VGQ: Total gaslighting victimization score (14 items, range 14-70)

    Note:
        Population norms (mean=42.0, std=10.0) are midpoint-based estimates.
        Update with empirical values when the original paper norms are confirmed.
    """

    def __init__(self, version: str = "14"):
        """Initialize VGQ inventory.

        Args:
            version: Version of the inventory ("14")
        """
        self.name = f"VGQ-{version}"
        self.domain = "Gaslighting Victimization"
        self.license = "Academic"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"vgq_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(f"VGQ version '{self.version}' not found in config")

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported traits.

        Returns:
            List of dicts with trait code and full name
        """
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code.

        Args:
            trait: Trait code or full name

        Returns:
            Normalized trait code

        Raises:
            ValueError: If trait is not recognized
        """
        trait_map: dict[str, str] = {
            "VGQ": "VGQ",
            "Gaslighting Victimization": "VGQ",
            "gaslighting victimization": "VGQ",
        }

        trait_code = trait_map.get(trait)
        if trait_code is None:
            trait_code = trait_map.get(trait.lower())

        if trait_code is None:
            valid_traits = ["VGQ"]
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores from responses.

        Computes the single VGQ total score.

        Args:
            responses: Dict mapping domain codes to list of scores

        Returns:
            Dict with domain scores including raw score, z-score, percentile
        """
        results: dict[str, dict[str, float]] = {}

        for domain, scores in responses.items():
            if domain not in self.config["domains"]:
                continue

            domain_config = self.config["domains"][domain]
            raw_score = sum(scores)
            mean = domain_config["population_mean"]
            std = domain_config["population_std"]

            z_score = (raw_score - mean) / std if std > 0 else 0.0
            percentile = self._z_to_percentile(z_score)

            results[domain] = {
                "domain_name": domain_config["name"],
                "raw_score": raw_score,
                "mean_score": raw_score / len(scores) if scores else 0.0,
                "population_mean": mean,
                "population_std": std,
                "z_score": z_score,
                "percentile": percentile,
                "num_items": float(len(scores)),
            }

        return results

    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile using normal CDF approximation.

        Args:
            z_score: Standard score

        Returns:
            Percentile (0-100)
        """
        if abs(z_score) < 3:
            percentile = 50 + 34.13 * z_score
        elif z_score >= 3:
            percentile = 99.87
        else:
            percentile = 0.13

        return max(0.0, min(100.0, percentile))
