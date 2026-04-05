"""Layer accessor for dynamic model layer access."""

import re

from torch import nn

from psyctl.core.logger import get_logger


class LayerAccessor:
    """
    Parse layer string and access model layers dynamically.

    Supports dot notation with bracket indexing for accessing nested model layers.
    Example: "model.layers[13].mlp.down_proj" → actual PyTorch module

    Attributes:
        logger: Logger instance for debugging
    """

    def __init__(self):
        """Initialize LayerAccessor with logger."""
        self.logger = get_logger("layer_accessor")

    def parse_layer_path(self, layer_str: str) -> list[str]:
        """
        Parse layer path string into components.

        Args:
            layer_str: Layer path string (e.g., "model.layers[13].mlp.down_proj")

        Returns:
            List of path components (e.g., ["model", "layers", "13", "mlp", "down_proj"])

        Example:
            >>> accessor = LayerAccessor()
            >>> accessor.parse_layer_path("model.layers[13].mlp.down_proj")
            ['model', 'layers', '13', 'mlp', 'down_proj']
        """
        # Replace brackets with dots: model.layers[13] → model.layers.13
        normalized = re.sub(r"\[(\d+)\]", r".\1", layer_str)

        # Split by dots and filter empty strings
        components = [c for c in normalized.split(".") if c]

        self.logger.debug(f"Parsed layer path '{layer_str}' → {components}")
        return components

    def get_layer(self, model: nn.Module, layer_str: str) -> nn.Module:
        """
        Get layer module from model using layer path string.

        Args:
            model: PyTorch model
            layer_str: Layer path string (e.g., "model.layers[13].mlp.down_proj")

        Returns:
            PyTorch module at the specified path

        Raises:
            AttributeError: If layer path is invalid or layer doesn't exist
            IndexError: If array index is out of bounds

        Example:
            >>> accessor = LayerAccessor()
            >>> layer = accessor.get_layer(model, "model.layers[13].mlp.down_proj")
        """
        components = self.parse_layer_path(layer_str)

        try:
            current = model
            path_so_far = []

            for component in components:
                path_so_far.append(component)

                # Check if component is a digit (array index)
                if component.isdigit():
                    index = int(component)
                    if not isinstance(current, nn.ModuleList | list | tuple):
                        raise AttributeError(
                            f"Cannot index into {type(current).__name__} "
                            f"at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    if index >= len(current):
                        raise IndexError(
                            f"Index {index} out of range for module list "
                            f"of length {len(current)} at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    current = current[index]
                else:
                    # Regular attribute access
                    if not hasattr(current, component):
                        raise AttributeError(
                            f"Module has no attribute '{component}' "
                            f"at path '{'.'.join(path_so_far[:-1])}'"
                        )
                    current = getattr(current, component)

            self.logger.debug(
                f"Successfully accessed layer '{layer_str}' → {type(current).__name__}"
            )
            return current

        except (AttributeError, IndexError) as e:
            self.logger.error(f"Failed to access layer '{layer_str}': {e}")
            raise

    def validate_layers(self, model: nn.Module, layer_strs: list[str]) -> bool:
        """
        Validate that all layer paths exist in the model.

        Args:
            model: PyTorch model
            layer_strs: List of layer path strings

        Returns:
            True if all layers are valid, False otherwise

        Example:
            >>> accessor = LayerAccessor()
            >>> layers = ["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"]
            >>> is_valid = accessor.validate_layers(model, layers)
        """
        self.logger.info(f"Validating {len(layer_strs)} layer paths...")

        all_valid = True
        for layer_str in layer_strs:
            try:
                self.get_layer(model, layer_str)
                self.logger.debug(f"Valid layer: {layer_str}")
            except (AttributeError, IndexError) as e:
                self.logger.error(f"Invalid layer '{layer_str}': {e}")
                all_valid = False

        if all_valid:
            self.logger.info("All layer paths are valid")
        else:
            self.logger.error("Some layer paths are invalid")

        return all_valid

    def get_layer_info(self, model: nn.Module, layer_str: str) -> dict:
        """
        Get information about a layer.

        Args:
            model: PyTorch model
            layer_str: Layer path string

        Returns:
            Dictionary with layer information (type, parameters, etc.)

        Example:
            >>> accessor = LayerAccessor()
            >>> info = accessor.get_layer_info(model, "model.layers[13].mlp.down_proj")
            >>> print(info['type'], info['num_parameters'])
        """
        layer = self.get_layer(model, layer_str)

        info = {
            "path": layer_str,
            "type": type(layer).__name__,
            "num_parameters": sum(p.numel() for p in layer.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            ),
        }

        # Add shape info for Linear layers
        if isinstance(layer, nn.Linear):
            info["in_features"] = layer.in_features
            info["out_features"] = layer.out_features
            info["bias"] = layer.bias is not None

        self.logger.debug(f"Layer info for '{layer_str}': {info}")
        return info

    def expand_layer_patterns(
        self, model: nn.Module, layer_patterns: list[str]
    ) -> list[str]:
        """
        Expand wildcard patterns to concrete layer paths.

        Supports:
        - [*] : all indices (e.g., "model.layers[*].mlp")
        - [start:end] : range (e.g., "model.layers[5:10].mlp")
        - [start:end:step] : range with step (e.g., "model.layers[0:20:2].mlp")
        - [start:] : from start to end (e.g., "model.layers[10:].mlp")
        - [:end] : from 0 to end (e.g., "model.layers[:5].mlp")

        Args:
            model: PyTorch model
            layer_patterns: List with possible wildcard patterns

        Returns:
            List of concrete layer paths

        Example:
            >>> accessor = LayerAccessor()
            >>> patterns = ["model.layers[*].mlp"]
            >>> expanded = accessor.expand_layer_patterns(model, patterns)
            >>> # Returns: ["model.layers[0].mlp", "model.layers[1].mlp", ...]
        """
        expanded = []
        for pattern in layer_patterns:
            if self._has_wildcard(pattern):
                self.logger.debug(f"Expanding pattern: {pattern}")
                expanded_paths = self._expand_single_pattern(model, pattern)
                self.logger.info(
                    f"Expanded pattern '{pattern}' to {len(expanded_paths)} layers"
                )
                expanded.extend(expanded_paths)
            else:
                # No wildcard, use as-is
                expanded.append(pattern)

        self.logger.info(
            f"Total: {len(layer_patterns)} patterns → {len(expanded)} layers"
        )
        return expanded

    def _has_wildcard(self, pattern: str) -> bool:
        """Check if pattern contains wildcard syntax."""
        return bool(re.search(r"\[\*\]|\[[\d\s:]*:[\d\s:]*\]", pattern))

    def _expand_single_pattern(self, model: nn.Module, pattern: str) -> list[str]:
        """
        Expand a single pattern to concrete paths.

        Args:
            model: PyTorch model
            pattern: Pattern string with wildcards

        Returns:
            List of concrete layer paths
        """
        # Parse the pattern into components
        # Find all bracket expressions
        bracket_pattern = r"\[([^\]]+)\]"
        bracket_info = []

        for match in re.finditer(bracket_pattern, pattern):
            content = match.group(1)
            bracket_info.append((match.start(), match.end(), content))

        # Build path components with wildcard info preserved
        if bracket_info:
            # Split pattern by bracket positions
            components = []
            prev_end = 0

            for start, end, content in bracket_info:
                # Get the part before the bracket
                before = pattern[prev_end:start]
                if before:
                    # Split by dots and add to components
                    components.extend(part for part in before.split(".") if part)

                # Add the bracket content as a component
                components.append(f"[{content}]")

                prev_end = end

            # Add any remaining parts after the last bracket
            temp_pattern = pattern[prev_end:]

            # Add any remaining parts after the last bracket
            if temp_pattern:
                components.extend(part for part in temp_pattern.split(".") if part)
        else:
            # No brackets, just split by dots
            components = [c for c in pattern.split(".") if c]

        self.logger.debug(f"Pattern components: {components}")

        # Recursively expand wildcards
        return self._recursive_expand(model, components, [])

    def _recursive_expand(
        self,
        current: nn.Module,
        remaining_components: list[str],
        path_so_far: list[str],
    ) -> list[str]:
        """
        Recursively expand wildcard components.

        Args:
            current: Current module in the traversal
            remaining_components: Components left to process
            path_so_far: Path components accumulated so far

        Returns:
            List of complete concrete paths
        """
        if not remaining_components:
            # Base case: no more components, return the formatted path
            return [self.format_layer_path(path_so_far)]

        component = remaining_components[0]
        rest = remaining_components[1:]

        # Check if component is a wildcard bracket
        if component.startswith("[") and component.endswith("]"):
            content = component[1:-1]

            if content == "*":
                # Wildcard: expand to all indices
                if not isinstance(current, nn.ModuleList | list | tuple):
                    self.logger.error(
                        f"Cannot expand wildcard on non-indexable type: {type(current).__name__}"
                    )
                    return []

                results = []
                for idx in range(len(current)):
                    new_path = [*path_so_far, str(idx)]
                    next_module = current[idx]
                    results.extend(self._recursive_expand(next_module, rest, new_path))
                return results

            elif ":" in content:
                # Slice notation
                if not isinstance(current, nn.ModuleList | list | tuple):
                    self.logger.error(
                        f"Cannot expand slice on non-indexable type: {type(current).__name__}"
                    )
                    return []

                # Parse slice
                slice_obj = self._parse_slice(content, len(current))
                indices = range(*slice_obj)

                results = []
                for idx in indices:
                    new_path = [*path_so_far, str(idx)]
                    next_module = current[idx]
                    results.extend(self._recursive_expand(next_module, rest, new_path))
                return results

            else:
                # Regular index
                idx = int(content)
                if not isinstance(current, nn.ModuleList | list | tuple):
                    self.logger.error(
                        f"Cannot index into non-indexable type: {type(current).__name__}"
                    )
                    return []

                new_path = [*path_so_far, str(idx)]
                next_module = current[idx]
                return self._recursive_expand(next_module, rest, new_path)

        else:
            # Regular attribute access
            if not hasattr(current, component):
                self.logger.error(
                    f"Module has no attribute '{component}' at path '{'.'.join(path_so_far)}'"
                )
                return []

            new_path = [*path_so_far, component]
            next_module = getattr(current, component)
            return self._recursive_expand(next_module, rest, new_path)

    def _parse_slice(self, slice_str: str, max_len: int) -> tuple[int, int, int]:
        """
        Parse slice notation like "5:10", ":5", "10:", "5:10:2".

        Args:
            slice_str: Slice string (e.g., "5:10", ":5", "10:", "5:10:2")
            max_len: Maximum length for the slice

        Returns:
            Tuple of (start, stop, step) for range()
        """
        parts = slice_str.split(":")

        if len(parts) == 2:
            start_str, stop_str = parts
            step = 1
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
            step = int(step_str) if step_str.strip() else 1
        else:
            raise ValueError(f"Invalid slice notation: {slice_str}")

        start = int(start_str) if start_str.strip() else 0
        stop = int(stop_str) if stop_str.strip() else max_len

        return (start, stop, step)

    def format_layer_path(self, components: list[str]) -> str:
        """
        Format list of components back to layer path string.

        Args:
            components: List of path components

        Returns:
            Formatted layer path string with bracket notation

        Example:
            >>> accessor = LayerAccessor()
            >>> accessor.format_layer_path(['model', 'layers', '13', 'mlp'])
            'model.layers[13].mlp'
        """
        result = []
        for comp in components:
            if comp.isdigit():
                # Add as bracket notation
                result[-1] += f"[{comp}]"
            else:
                result.append(comp)

        return ".".join(result)
