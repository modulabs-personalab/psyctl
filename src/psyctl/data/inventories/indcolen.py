"""INDCOL (Individualism and Collectivism Scale) implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("indcolen")
class INDCOL(BaseInventory):
    """INDCOL (Individualism and Collectivism Scale) - 32 item form.

    Based on Singelis et al. (1995) 4-factor structure:
        HI: Horizontal Individualism
        VI: Vertical Individualism
        HC: Horizontal Collectivism
        VC: Vertical Collectivism
    """

    def __init__(self, version: str = "1"):
        """Initialize INDCOL inventory."""
        self.name = f"INDCOL-{version}"
        self.domain = "Individualism-Collectivism"
        self.license = "Unknown"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"indcolen_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(f"INDCOL version '{self.version}' not found in config")

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported traits."""
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code."""
        trait_map: dict[str, str] = {
            "HI": "HI",
            "Horizontal Individualism": "HI",
            "horizontal individualism": "HI",
            "VI": "VI",
            "Vertical Individualism": "VI",
            "vertical individualism": "VI",
            "HC": "HC",
            "Horizontal Collectivism": "HC",
            "horizontal collectivism": "HC",
            "VC": "VC",
            "Vertical Collectivism": "VC",
            "vertical collectivism": "VC",
        }

        trait_code = trait_map.get(trait) or trait_map.get(trait.lower())
        if trait_code is None:
            valid_traits = list(self.config["domains"].keys())
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores from responses."""
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
        """Convert z-score to percentile using normal CDF approximation."""
        if abs(z_score) < 3:
            percentile = 50 + 34.13 * z_score
        elif z_score >= 3:
            percentile = 99.87
        else:
            percentile = 0.13

        return max(0.0, min(100.0, percentile))
