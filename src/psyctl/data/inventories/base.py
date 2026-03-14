"""Base class for personality inventories."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar


class BaseInventory(ABC):
    """Abstract base class for personality inventories.

    Subclasses must define class attributes and implement _normalize_trait().
    All other methods have default implementations that can be overridden.

    Class attributes:
        registry_name: Registry key (e.g., "ipip_neo"). Set automatically by
            @register_inventory decorator if not defined explicitly.
        config_key: Config lookup key override. Defaults to
            "{registry_name}_{version}". Set explicitly for non-standard keys
            (e.g., SD4 uses "sd4" instead of "sd4_28").
        default_version: Default version string (e.g., "120").
        display_name: Human-readable name (e.g., "IPIP-NEO").
        domain_label: Domain description (e.g., "Big Five").
        license_type: License identifier (e.g., "MIT").
    """

    registry_name: ClassVar[str]
    config_key: ClassVar[str | None] = None
    default_version: ClassVar[str]
    display_name: ClassVar[str]
    domain_label: ClassVar[str]
    license_type: ClassVar[str]

    def __init__(self, version: str | None = None):
        """Initialize inventory.

        Args:
            version: Inventory version. Falls back to default_version.
        """
        self.version = version or self.default_version
        self.name = f"{self.display_name}-{self.version}"
        self.domain = self.domain_label
        self.license = self.license_type
        self.config = self._load_config()
        self.questions = self._load_questions()

    def _resolve_config_key(self) -> str:
        """Resolve the config lookup key.

        Returns:
            Config key string for benchmark_config.json lookup.
        """
        if self.config_key is not None:
            return self.config_key
        return f"{self.registry_name}_{self.version}"

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration from benchmark_config.json."""
        config_path = self.get_config_path()
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        key = self._resolve_config_key()
        if "inventories" not in all_configs or key not in all_configs["inventories"]:
            raise ValueError(
                f"{self.display_name} version '{self.version}' not found in config "
                f"(key: '{key}')"
            )

        return all_configs["inventories"][key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file referenced in config."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            return json.load(f)

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported personality traits.

        Returns:
            List of dicts with trait code and full name.
        """
        return [
            {"code": code, "name": info["name"]}
            for code, info in self.config["domains"].items()
        ]

    @abstractmethod
    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code.

        Args:
            trait: Trait code or full name.

        Returns:
            Normalized trait code.

        Raises:
            ValueError: If trait is not recognized.
        """
        ...

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate personality scores from responses.

        Computes raw score, z-score, and percentile for each domain.
        Override for custom scoring logic (e.g., parent scale aggregation).

        Args:
            responses: Dict mapping domain codes to lists of item scores.

        Returns:
            Dict with domain scores and statistics.
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
        """Convert z-score to percentile using normal CDF.

        Args:
            z_score: Standard score.

        Returns:
            Percentile (0-100).
        """
        percentile = 50.0 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        return max(0.0, min(100.0, percentile))

    def get_questions(self, trait: str | None = None) -> list[dict[str, Any]]:
        """Get questions, optionally filtered by trait.

        Args:
            trait: Specific trait to filter. Returns all if None.

        Returns:
            List of questions (all or filtered by trait).
        """
        if trait is None:
            return self.questions.copy()

        trait_code = self._normalize_trait(trait)
        return [
            q
            for q in self.questions
            if q.get("domain") == trait_code or q.get("trait") == trait_code
        ]

    def get_inventory_info(self) -> dict[str, Any]:
        """Get inventory metadata.

        Returns:
            Dict with inventory information.
        """
        traits = self.get_supported_traits()
        total_questions = len(self.questions)
        num_traits = len(traits)

        return {
            "name": self.config.get("name", "Unknown"),
            "version": self.version,
            "total_questions": total_questions,
            "traits": traits,
            "questions_per_trait": (
                total_questions // num_traits if num_traits > 0 else 0
            ),
        }

    @classmethod
    def get_config_path(cls) -> Path:
        """Get path to inventory config file."""
        return Path(__file__).parent.parent / "benchmark_config.json"
