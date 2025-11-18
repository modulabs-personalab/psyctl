"""Base class for personality inventories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseInventory(ABC):
    """
    Abstract base class for personality inventories.

    Provides common interface and functionality for all personality
    inventory implementations (IPIP-NEO, NPI-40, MACH-IV, etc.).
    """

    def __init__(self, version: str | None = None):
        """
        Initialize inventory.

        Args:
            version: Inventory version (e.g., "120", "300")
        """
        self.version = version
        self.config = self._load_config()
        self.questions = self._load_questions()

    @abstractmethod
    def _load_config(self) -> dict[str, Any]:
        """
        Load inventory configuration.

        Returns:
            Configuration dict with inventory metadata
        """
        pass

    @abstractmethod
    def _load_questions(self) -> list[dict[str, Any]]:
        """
        Load questions from data file.

        Returns:
            List of question dicts
        """
        pass

    @abstractmethod
    def get_supported_traits(self) -> list[dict[str, str]]:
        """
        Get list of supported personality traits.

        Returns:
            List of dicts with trait code and full name
            Example: [{"code": "E", "name": "Extraversion"}, ...]
        """
        pass

    @abstractmethod
    def _normalize_trait(self, trait: str) -> str:
        """
        Normalize trait name to domain code.

        Args:
            trait: Trait code or full name

        Returns:
            Normalized trait code

        Raises:
            ValueError: If trait is not recognized
        """
        pass

    @abstractmethod
    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """
        Calculate personality scores from responses.

        Args:
            responses: Dict mapping trait codes to lists of scores

        Returns:
            Dict with trait scores and statistics
        """
        pass

    def get_questions(self, trait: str | None = None) -> list[dict[str, Any]]:
        """
        Get questions, optionally filtered by trait.

        Args:
            trait: Specific trait to filter. Returns all if None.

        Returns:
            List of questions (all or filtered by trait)
        """
        if trait is None:
            return self.questions.copy()

        # Normalize trait and filter
        trait_code = self._normalize_trait(trait)
        return [
            q
            for q in self.questions
            if q.get("domain") == trait_code or q.get("trait") == trait_code
        ]

    def get_inventory_info(self) -> dict[str, Any]:
        """
        Get inventory metadata.

        Returns:
            Dict with inventory information
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
        """
        Get path to inventory config file.

        Returns:
            Path to config JSON file
        """
        return Path(__file__).parent.parent / "benchmark_config.json"

