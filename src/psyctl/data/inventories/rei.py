"""REI (Rational Experiential Inventory) implementation."""

from __future__ import annotations

from typing import Any, ClassVar

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("rei")
class REI(BaseInventory):
    """REI (Rational Experiential Inventory) - Dual-process thinking styles.

    Subscales:
        RA: Rational Ability
        RE: Rational Engagement
        EA: Experiential Ability
        EE: Experiential Engagement

    Higher-order scales:
        R: Rationality (RA + RE)
        E: Experientiality (EA + EE)
    """

    registry_name: ClassVar[str] = "rei"
    default_version: ClassVar[str] = "40"
    display_name: ClassVar[str] = "REI"
    domain_label: ClassVar[str] = "Dual-Process Thinking"
    license_type: ClassVar[str] = "Academic"

    _PARENT_SCALES: ClassVar[dict[str, list[str]]] = {
        "R": ["RA", "RE"],
        "E": ["EA", "EE"],
    }

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code."""
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
        """Get questions with parent scale support.

        trait="R" returns RA+RE questions, trait="E" returns EA+EE questions.
        """
        if trait is None:
            return self.questions.copy()

        trait_code = self._normalize_trait(trait)

        if trait_code == "R":
            return [q for q in self.questions if q.get("domain") in ("RA", "RE")]
        if trait_code == "E":
            return [q for q in self.questions if q.get("domain") in ("EA", "EE")]

        return [q for q in self.questions if q.get("domain") == trait_code]

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores with parent scale aggregation."""
        results = super().calculate_scores(responses)

        for parent, subscales in self._PARENT_SCALES.items():
            if parent not in self.config["domains"]:
                continue

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
