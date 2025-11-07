"""Layer specification resolver for dynamic layer selection.

This resolver treats layer specifications as list indices, not layer numbers.
This makes it architecture-independent - works with any model structure.
"""

from __future__ import annotations

import math
from typing import Any

from psyctl.core.logger import get_logger

logger = get_logger("layer_resolver")


class LayerResolver:
    """
    Resolve layer specifications to actual layer names using list indices.
    
    Supports:
    - Direct indices: "0,5,10,15" -> layers at indices [0, 5, 10, 15]
    - Range notation: "0-5,10-15" -> layers at indices 0-5 and 10-15
    - Named groups: "early", "middle", "late" -> computed as fractions of total layers
    - All layers: "all" or None -> all available layers
    
    Note: Specifications are interpreted as **list indices**, not layer numbers.
    This means "10-20" selects available_layers[10:21], regardless of what
    the actual layer names are (model.layers[10] vs transformer.h[10] etc).
    """

    @staticmethod
    def _extract_layer_number(layer_name: str) -> str:
        """
        Extract a display label from a layer name.
        
        This is used for logging and display purposes only.
        Returns the layer name itself as it is architecture-independent.
        
        Args:
            layer_name: Full layer name (e.g., "model.layers[13].mlp.down_proj")
            
        Returns:
            A shortened display label (e.g., "model.layers[13]")
        """
        # Extract just the layer identifier for display
        # e.g., "model.layers[13].mlp.down_proj" -> "layer[13]"
        import re
        match = re.search(r'layers?\[(\d+)\]|layers?\.(\d+)|h\.(\d+)|transformer\.(\d+)', layer_name, re.IGNORECASE)
        if match:
            num = next(g for g in match.groups() if g is not None)
            return f"layer[{num}]"
        # If no pattern matches, return the layer name itself (shortened)
        return layer_name.split(".")[-2] if "." in layer_name else layer_name

    @staticmethod
    def resolve_layer_spec(
        layer_spec: str | list[str] | None,
        available_layers: list[str],
        layer_groups_config: dict[str, Any],
    ) -> list[str]:
        """
        Resolve layer specification to actual layer names using indices.
        
        Args:
            layer_spec: Layer specification (indices, ranges, or keywords)
            available_layers: List of available layer names from steering vector
            layer_groups_config: Configuration for named layer groups
            
        Returns:
            List of resolved layer names
            
        Examples:
            >>> # Available layers from vector file (any architecture)
            >>> layers = ["transformer.h[0]", "transformer.h[1]", ..., "transformer.h[31]"]
            >>> 
            >>> # Direct indices
            >>> resolve_layer_spec("0,5,10", layers, {})
            ["transformer.h[0]", "transformer.h[5]", "transformer.h[10]"]
            >>> 
            >>> # Range
            >>> resolve_layer_spec("5-10", layers, {})
            ["transformer.h[5]", ..., "transformer.h[10]"]  # 6 layers
            >>> 
            >>> # Keywords (computed from total layer count)
            >>> resolve_layer_spec("early", layers, {"early": {"fraction": 0.33}})
            ["transformer.h[0]", ..., "transformer.h[10]"]  # First 33%
            >>> 
            >>> # All layers
            >>> resolve_layer_spec("all", layers, {}) == layers
            True
        """
        # Handle None or "all" - return all layers
        if layer_spec is None or layer_spec == "all":
            return available_layers

        # Convert string to list of specs
        if isinstance(layer_spec, str):
            specs = [s.strip() for s in layer_spec.split(",")]
        else:
            specs = layer_spec

        # Resolve each spec to indices
        total_layers = len(available_layers)
        resolved_indices = set()

        for spec in specs:
            spec = spec.strip().lower()
            
            # Check if it's a keyword (early, middle, late, etc.)
            if spec in layer_groups_config:
                indices = LayerResolver._resolve_keyword(
                    spec, total_layers, layer_groups_config
                )
                resolved_indices.update(indices)
            
            # Check if it's a range (e.g., "5-10")
            elif "-" in spec and not spec.startswith("-"):
                indices = LayerResolver._resolve_range(spec, total_layers)
                resolved_indices.update(indices)
            
            # Otherwise treat as direct index
            else:
                try:
                    idx = int(spec)
                    if 0 <= idx < total_layers:
                        resolved_indices.add(idx)
                    else:
                        logger.warning(
                            f"Index {idx} out of range (0-{total_layers-1}), skipping"
                        )
                except ValueError:
                    logger.warning(f"Unrecognized layer specification: '{spec}'")

        # Convert indices to layer names
        sorted_indices = sorted(resolved_indices)
        result = [available_layers[i] for i in sorted_indices]
        
        if not result:
            logger.warning(
                f"No layers resolved for spec: {layer_spec}. Using all layers."
            )
            return available_layers
        
        logger.info(
            f"Resolved '{layer_spec}' to {len(result)} layers (indices: {sorted_indices[:5]}{'...' if len(sorted_indices) > 5 else ''})"
        )
        return result

    @staticmethod
    def _resolve_keyword(
        keyword: str, total_layers: int, config: dict[str, Any]
    ) -> list[int]:
        """Resolve keyword (early, middle, late) to list of indices."""
        group_config = config.get(keyword, {})
        group_type = group_config.get("type", keyword)
        
        if group_type == "all":
            return list(range(total_layers))
        
        elif group_type == "early":
            fraction = group_config.get("fraction", 0.33)
            count = math.ceil(total_layers * fraction)
            return list(range(count))
        
        elif group_type == "middle":
            start_idx = math.floor(total_layers * 0.25)
            end_idx = math.ceil(total_layers * 0.75)
            return list(range(start_idx, end_idx))
        
        elif group_type == "late":
            fraction = group_config.get("fraction", 0.33)
            count = math.ceil(total_layers * fraction)
            return list(range(total_layers - count, total_layers))
        
        else:
            logger.warning(f"Unknown keyword type: {group_type}")
            return []

    @staticmethod
    def _resolve_range(range_spec: str, total_layers: int) -> list[int]:
        """Resolve range specification (e.g., "5-10") to list of indices."""
        try:
            parts = range_spec.split("-")
            if len(parts) != 2:
                logger.warning(f"Invalid range format: {range_spec}")
                return []
            
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            
            # Clamp to valid range
            start = max(0, min(start, total_layers - 1))
            end = max(0, min(end, total_layers - 1))
            
            if start > end:
                logger.warning(
                    f"Invalid range {start}-{end} (start > end), using {end}-{start}"
                )
                start, end = end, start
            
            return list(range(start, end + 1))
        
        except ValueError as e:
            logger.warning(f"Failed to parse range '{range_spec}': {e}")
            return []

    @staticmethod
    def describe_layer_spec(
        layer_spec: str | list[str] | None,
        layer_groups_config: dict[str, Any],
    ) -> str:
        """Provide human-readable description of layer specification."""
        if layer_spec is None:
            return "all layers"
        
        if isinstance(layer_spec, list):
            return ", ".join(str(s) for s in layer_spec)
        
        # Enhance keyword descriptions
        specs = [s.strip() for s in layer_spec.split(",")]
        descriptions = []
        
        for spec in specs:
            if spec in layer_groups_config:
                desc = layer_groups_config[spec].get("description", spec)
                descriptions.append(f"{spec} ({desc})")
            else:
                descriptions.append(spec)
        
        return ", ".join(descriptions)
