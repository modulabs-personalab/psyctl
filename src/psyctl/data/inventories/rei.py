"""REI (Rational Experiential Inventory) implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory

# Mapping from subscale codes to parent scale
_SUBSCALE_TO_PARENT: dict[str, str] = {
    "RA": "R",
    "RE": "R",
    "EA": "E",
    "EE": "E",
}


@register_inventory("rei")
class REI(BaseInventory):
    """REI (Rational Experiential Inventory) - Dual-process thinking styles.

    Measures rational and experiential information processing styles
    based on Pacini & Epstein (1999).

    Subscales:
        RA: Rational Ability
        RE: Rational Engagement
        EA: Experiential Ability
        EE: Experiential Engagement

    Higher-order scales:
        R: Rationality (RA + RE)
        E: Experientiality (EA + EE)
    """

    def __init__(self, version: str = "40"):
        """Initialize REI inventory.

        Args:
            version: Version of the inventory ("40")
        """
        self.name = f"REI-{version}"
        self.domain = "Dual-Process Thinking"
        self.license = "Academic"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"rei_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(f"REI version '{self.version}' not found in config")

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported traits (subscales and parent scales).

        Returns:
            List of dicts with trait code and full name
        """
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code.

        Supports both subscale codes (RA, RE, EA, EE) and parent scale
        codes (R, E) as well as full names.

        Args:
            trait: Trait code or full name

        Returns:
            Normalized trait code

        Raises:
            ValueError: If trait is not recognized
        """
        trait_map: dict[str, str] = {
            "RA": "RA",
            "Rational Ability": "RA",
            "rational ability": "RA",
            "RE": "RE",
            "Rational Engagement": "RE",
            "rational engagement": "RE",
            "EA": "EA",
            "Experiential Ability": "EA",
            "experiential ability": "EA",
            "EE": "EE",
            "Experiential Engagement": "EE",
            "experiential engagement": "EE",
            "R": "R",
            "Rationality": "R",
            "rationality": "R",
            "E": "E",
            "Experientiality": "E",
            "experientiality": "E",
        }

        trait_code = trait_map.get(trait)
        if trait_code is None:
            trait_code = trait_map.get(trait.lower())

        if trait_code is None:
            valid_traits = ["RA", "RE", "EA", "EE", "R", "E"]
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def get_questions(self, trait: str | None = None) -> list[dict[str, Any]]:
        """Get questions, optionally filtered by trait.

        Supports parent scale filtering: trait="R" returns RA+RE questions,
        trait="E" returns EA+EE questions.

        Args:
            trait: Specific trait to filter. Returns all if None.

        Returns:
            List of questions (all or filtered by trait)
        """
        if trait is None:
            return self.questions.copy()

        trait_code = self._normalize_trait(trait)

        # Parent scale: return questions from both subscales
        if trait_code == "R":
            return [q for q in self.questions if q.get("domain") in ("RA", "RE")]
        if trait_code == "E":
            return [q for q in self.questions if q.get("domain") in ("EA", "EE")]

        # Subscale: return questions for that specific domain
        return [q for q in self.questions if q.get("domain") == trait_code]

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores from responses.

        Computes scores for 4 subscales (RA, RE, EA, EE) and
        2 parent scales (R = RA+RE, E = EA+EE).

        Args:
            responses: Dict mapping domain codes to list of scores

        Returns:
            Dict with domain scores including raw score, z-score, percentile
        """
        results: dict[str, dict[str, float]] = {}

        # Calculate subscale scores
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

        # Calculate parent scale scores (R = RA + RE, E = EA + EE)
        parent_scales: dict[str, list[str]] = {
            "R": ["RA", "RE"],
            "E": ["EA", "EE"],
        }

        for parent, subscales in parent_scales.items():
            if parent not in self.config["domains"]:
                continue

            # Combine scores from subscales
            combined_scores: list[float] = []
            for sub in subscales:
                if sub in responses:
                    combined_scores.extend(responses[sub])

            if not combined_scores:
                continue

            parent_config = self.config["domains"][parent]
            raw_score = sum(combined_scores)
            mean = parent_config["population_mean"]
            std = parent_config["population_std"]

            z_score = (raw_score - mean) / std if std > 0 else 0.0
            percentile = self._z_to_percentile(z_score)

            results[parent] = {
                "domain_name": parent_config["name"],
                "raw_score": raw_score,
                "mean_score": raw_score / len(combined_scores),
                "population_mean": mean,
                "population_std": std,
                "z_score": z_score,
                "percentile": percentile,
                "num_items": float(len(combined_scores)),
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
