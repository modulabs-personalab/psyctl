"""VGQ (Victim Gaslighting Questionnaire) implementation."""

from __future__ import annotations

from typing import ClassVar

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("vgq")
class VGQ(BaseInventory):
    """VGQ (Victim Gaslighting Questionnaire) - Gaslighting victimization.

    Single domain:
        VGQ: Total gaslighting victimization score (14 items, range 14-70)

    Note:
        Population norms (mean=42.0, std=10.0) are midpoint-based estimates.
    """

    registry_name: ClassVar[str] = "vgq"
    default_version: ClassVar[str] = "14"
    display_name: ClassVar[str] = "VGQ"
    domain_label: ClassVar[str] = "Gaslighting Victimization"
    license_type: ClassVar[str] = "Academic"

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code."""
        trait_map: dict[str, str] = {
            "VGQ": "VGQ",
            "Gaslighting Victimization": "VGQ",
            "gaslighting victimization": "VGQ",
        }

        trait_code = trait_map.get(trait)
        if trait_code is None:
            trait_code = trait_map.get(trait.lower())

        if trait_code is None:
            raise ValueError(f"Unrecognized trait '{trait}'. Valid traits: ['VGQ']")

        return trait_code
