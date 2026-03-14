"""SD4 (Short Dark Tetrad) inventory implementation."""

from __future__ import annotations

from typing import ClassVar

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("sd4")
class SD4(BaseInventory):
    """SD4 (Short Dark Tetrad) - Dark personality traits."""

    registry_name: ClassVar[str] = "sd4"
    config_key: ClassVar[str | None] = "sd4"
    default_version: ClassVar[str] = "28"
    display_name: ClassVar[str] = "SD4"
    domain_label: ClassVar[str] = "Dark Tetrad"
    license_type: ClassVar[str] = "Academic"

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code."""
        trait_map: dict[str, str] = {
            "M": "M",
            "Machiavellianism": "M",
            "machiavellianism": "M",
            "N": "N",
            "Narcissism": "N",
            "narcissism": "N",
            "P": "P",
            "Psychopathy": "P",
            "psychopathy": "P",
            "S": "S",
            "Sadism": "S",
            "sadism": "S",
        }

        trait_code = trait_map.get(trait)
        if trait_code is None:
            trait_code = trait_map.get(trait.upper())

        if trait_code is None:
            valid_traits = ["M", "N", "P", "S"]
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code
