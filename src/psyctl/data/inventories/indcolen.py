"""INDCOL (Individualism and Collectivism Scale) implementation."""

from __future__ import annotations

from typing import ClassVar

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

    registry_name: ClassVar[str] = "indcolen"
    default_version: ClassVar[str] = "1"
    display_name: ClassVar[str] = "INDCOL"
    domain_label: ClassVar[str] = "Individualism-Collectivism"
    license_type: ClassVar[str] = "Unknown"

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
