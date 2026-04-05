"""Vector storage and loading utilities."""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from psyctl.core.logger import get_logger


class VectorStore:
    """Store and load steering vectors with metadata."""

    def __init__(self):
        self.logger = get_logger("vector_store")

    def save_steering_vector(
        self, vector: torch.Tensor, metadata: dict[str, Any], filepath: Path
    ) -> None:
        """
        Save single steering vector with metadata (legacy method).

        Args:
            vector: Steering vector tensor
            metadata: Metadata dictionary
            filepath: Output file path

        Note:
            For new code, consider using save_multi_layer() for better compatibility
        """
        self.logger.info(f"Saving steering vector to: {filepath}")
        self.logger.debug(f"Vector shape: {vector.shape}")
        self.logger.debug(f"Metadata: {metadata}")

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {filepath.parent}")

            # Save vector and metadata
            save_file({"steering_vector": vector}, filepath, metadata=metadata)

            self.logger.success(f"Steering vector saved successfully to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save steering vector: {e}")
            raise

    def save_multi_layer(
        self,
        vectors: dict[str, torch.Tensor],
        output_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Save multiple steering vectors to single safetensors file.

        Args:
            vectors: Dictionary mapping layer names to steering vectors
            output_path: Output file path
            metadata: Metadata dictionary with extraction information

        Example:
            >>> store = VectorStore()
            >>> vectors = {
            ...     "model.layers[13].mlp.down_proj": tensor(...),
            ...     "model.layers[14].mlp.down_proj": tensor(...)
            ... }
            >>> metadata = {
            ...     "model": "meta-llama/Llama-3.2-3B-Instruct",
            ...     "method": "MeanDifferenceActivationVector",
            ...     "layers": list(vectors.keys())
            ... }
            >>> store.save_multi_layer(vectors, Path("out.safetensors"), metadata)
        """
        self.logger.info(f"Saving {len(vectors)} steering vectors to: {output_path}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare tensors with layer keys
            tensors_to_save = {}
            for idx, (layer_name, vector) in enumerate(vectors.items()):
                tensor_key = f"layer_{idx}"
                tensors_to_save[tensor_key] = vector
                self.logger.debug(
                    f"Prepared '{layer_name}' as '{tensor_key}': shape={vector.shape}"
                )

            # Add metadata with layer mapping
            enhanced_metadata = {
                **metadata,
                "num_layers": len(vectors),
                "layer_names": list(vectors.keys()),
                "created_at": datetime.now().isoformat(),
            }

            # Convert metadata values to strings (safetensors requirement)
            string_metadata = {
                k: str(v) if not isinstance(v, str) else v
                for k, v in enhanced_metadata.items()
            }

            # Save to file
            save_file(tensors_to_save, output_path, metadata=string_metadata)

            self.logger.success(
                f"Saved {len(vectors)} steering vectors to {output_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save multi-layer vectors: {e}")
            raise

    def load_steering_vector(
        self, filepath: Path
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Load steering vector and metadata (legacy method).

        Args:
            filepath: Path to safetensors file

        Returns:
            Tuple of (vector, metadata)
        """
        self.logger.info(f"Loading steering vector from: {filepath}")

        try:
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Steering vector file does not exist: {filepath}"
                )

            # Load vector and metadata
            tensors = load_file(filepath)
            vector = tensors["steering_vector"]

            from safetensors import safe_open

            with safe_open(filepath, framework="pt") as f:
                metadata = f.metadata() or {}

            self.logger.debug(f"Loaded vector shape: {vector.shape}")
            self.logger.debug(f"Loaded metadata: {metadata}")
            self.logger.success(f"Steering vector loaded successfully from {filepath}")

            return vector, metadata  # type: ignore[return-value]

        except Exception as e:
            self.logger.error(f"Failed to load steering vector: {e}")
            raise

    def load_multi_layer(
        self, filepath: Path
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Load multiple steering vectors from safetensors file.

        Args:
            filepath: Path to safetensors file

        Returns:
            Tuple of (vectors_dict, metadata)
            vectors_dict maps layer names to tensors

        Example:
            >>> store = VectorStore()
            >>> vectors, metadata = store.load_multi_layer(Path("out.safetensors"))
            >>> print(list(vectors.keys()))
            ['model.layers[13].mlp.down_proj', 'model.layers[14].mlp.down_proj']
        """
        self.logger.info(f"Loading multi-layer vectors from: {filepath}")

        try:
            if not filepath.exists():
                raise FileNotFoundError(f"File does not exist: {filepath}")

            # Load all tensors
            data = load_file(filepath)

            # Extract metadata from safetensors
            # Note: safetensors stores metadata separately, we need to load it differently

            from safetensors import safe_open

            metadata = {}
            with safe_open(filepath, framework="pt") as f:
                metadata_str = f.metadata()
                if metadata_str:
                    # Parse layer_names if it's a string representation of a list
                    if "layer_names" in metadata_str:
                        try:
                            import json as _json

                            metadata_str["layer_names"] = _json.loads(
                                metadata_str["layer_names"]
                            )
                        except (ValueError, TypeError):
                            import ast

                            try:
                                metadata_str["layer_names"] = ast.literal_eval(
                                    metadata_str["layer_names"]
                                )
                            except (ValueError, SyntaxError):
                                self.logger.warning(
                                    f"Failed to parse layer_names: {metadata_str['layer_names']}"
                                )
                    metadata = metadata_str

            # Reconstruct vectors dictionary
            vectors = {}
            layer_names = metadata.get("layer_names", [])

            if isinstance(layer_names, str):
                import json as _json

                try:
                    layer_names = _json.loads(layer_names)
                except (ValueError, TypeError):
                    import ast

                    try:
                        layer_names = ast.literal_eval(layer_names)
                    except (ValueError, SyntaxError):
                        self.logger.warning(
                            f"Failed to parse layer_names string: {layer_names}"
                        )
                        layer_names = []

            for idx, layer_name in enumerate(layer_names):
                tensor_key = f"layer_{idx}"
                if tensor_key in data:
                    vectors[layer_name] = data[tensor_key]
                    self.logger.debug(
                        f"Loaded '{layer_name}' from '{tensor_key}': "
                        f"shape={data[tensor_key].shape}"
                    )

            self.logger.info(f"Loaded {len(vectors)} steering vectors")
            self.logger.debug(f"Metadata: {metadata}")

            return vectors, metadata

        except Exception as e:
            self.logger.error(f"Failed to load multi-layer vectors: {e}")
            raise
