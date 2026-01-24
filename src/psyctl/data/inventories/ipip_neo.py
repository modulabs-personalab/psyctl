"""IPIP-NEO inventory implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("ipip_neo")
class IPIPNEO(BaseInventory):
    """IPIP-NEO (International Personality Item Pool - NEO) inventory."""

    def __init__(self, version: str = "120"):
        """
        Initialize IPIP-NEO inventory.

        Args:
            version: Version of the inventory ("120" or "300")
        """
        self.name = f"IPIP-NEO-{version}"
        self.domain = "Big Five"
        self.license = "MIT"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"ipip_neo_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(f"Inventory version '{self.version}' not found in config")

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_questions_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """
        Get questions for a specific domain.

        Args:
            domain: Domain code (N, E, O, A, C)

        Returns:
            List of questions for that domain
        """
        return [q for q in self.questions if q["domain"] == domain]

    def get_supported_traits(self) -> list[dict[str, str]]:
        """
        Get list of supported personality traits.

        Returns:
            List of dicts with trait code and full name
        """
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """
        Normalize trait name to domain code.

        Args:
            trait: Trait code or full name (e.g., "E", "Extraversion")

        Returns:
            Domain code (N, E, O, A, C)

        Raises:
            ValueError: If trait is not recognized
        """
        trait_map = {
            "N": "N",
            "Neuroticism": "N",
            "neuroticism": "N",
            "E": "E",
            "Extraversion": "E",
            "extraversion": "E",
            "O": "O",
            "Openness": "O",
            "openness": "O",
            "A": "A",
            "Agreeableness": "A",
            "agreeableness": "A",
            "C": "C",
            "Conscientiousness": "C",
            "conscientiousness": "C",
        }

        trait_code = trait_map.get(trait)
        if trait_code is None:
            # Try uppercase
            trait_code = trait_map.get(trait.upper())

        if trait_code is None:
            valid_traits = list(set(trait_map.values()))
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate personality scores from responses.

        Args:
            responses: Dict mapping domain codes to list of scores

        Returns:
            Dict with domain scores including raw score, z-score, percentile
        """
        results = {}

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
                "num_items": len(scores),
            }

        return results

    def _z_to_percentile(self, z_score: float) -> float:
        """
        Convert z-score to percentile using normal CDF approximation.

        Args:
            z_score: Standard score

        Returns:
            Percentile (0-100)
        """
        # Approximation of cumulative distribution function
        # Using the error function approximation

        # Simple approximation: percentile ≈ 50 + 34.13 * z for |z| < 3
        if abs(z_score) < 3:
            percentile = 50 + 34.13 * z_score
        elif z_score >= 3:
            percentile = 99.87
        else:
            percentile = 0.13

        return max(0.0, min(100.0, percentile))
