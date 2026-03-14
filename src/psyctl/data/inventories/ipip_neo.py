"""IPIP-NEO inventory implementation."""

from __future__ import annotations

from typing import ClassVar

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("ipip_neo")
class IPIPNEO(BaseInventory):
    """IPIP-NEO (International Personality Item Pool - NEO) inventory."""

    registry_name: ClassVar[str] = "ipip_neo"
    default_version: ClassVar[str] = "120"
    display_name: ClassVar[str] = "IPIP-NEO"
    domain_label: ClassVar[str] = "Big Five"
    license_type: ClassVar[str] = "MIT"

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code."""
        trait_map: dict[str, str] = {
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
            trait_code = trait_map.get(trait.upper())

        if trait_code is None:
            valid_traits = list(set(trait_map.values()))
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code
