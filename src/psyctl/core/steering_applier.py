"""Steering vector applier for text generation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from psyctl.core.layer_accessor import LayerAccessor
from psyctl.core.logger import get_logger
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore


class SteeringApplier:
    """Apply steering vectors to models for text generation."""

    def __init__(self):
        self.logger = get_logger("steering_applier")
        self.llm_loader = LLMLoader()
        self.vector_store = VectorStore()
        self.layer_accessor = LayerAccessor()

    def apply_steering(
        self,
        steering_vector_path: Path,
        input_text: str,
        model_name: str | None = None,
        model: nn.Module | None = None,
        tokenizer: AutoTokenizer | None = None,
        strength: float | dict[str, float] = 1.0,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        orthogonal: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Apply steering vector and generate text.

        Args:
            steering_vector_path: Path to steering vector file
            input_text: Input text for generation
            model_name: Hugging Face model identifier (optional if model provided)
            model: Pre-loaded model (optional if model_name provided)
            tokenizer: Pre-loaded tokenizer (optional if model_name provided)
            strength: Steering strength multiplier. Can be:
                - float: Apply same strength to all layers (default: 1.0)
                - Dict[str, float]: Per-layer strength mapping. Missing layers default to 1.0
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            orthogonal: Use orthogonalized addition method
            verbose: Log full prompt after chat template application

        Returns:
            Generated text string

        Raises:
            ValueError: If neither model_name nor model is provided, or both are provided
            ValueError: If model is provided without tokenizer

        Examples:
            >>> # Example 1: Using model_name (original usage)
            >>> applier = SteeringApplier()
            >>> result = applier.apply_steering(
            ...     model_name="google/gemma-3-270m-it",
            ...     steering_vector_path=Path("./vector.safetensors"),
            ...     input_text="hello world",
            ...     strength=1.5
            ... )

            >>> # Example 2: Using pre-loaded model (efficient for multiple generations)
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
            >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
            >>> applier = SteeringApplier()
            >>> for strength in [0.5, 1.0, 1.5]:
            ...     result = applier.apply_steering(
            ...         model=model,
            ...         tokenizer=tokenizer,
            ...         steering_vector_path=Path("./vector.safetensors"),
            ...         input_text="hello world",
            ...         strength=strength
            ...     )
        """
        # Validate model parameters
        if model is not None and model_name is not None:
            raise ValueError(
                "Cannot provide both 'model' and 'model_name'. Choose one."
            )
        if model is None and model_name is None:
            raise ValueError("Must provide either 'model' or 'model_name'.")
        if model is not None and tokenizer is None:
            raise ValueError("Must provide 'tokenizer' when providing 'model'.")

        # Determine model identifier for logging
        if model_name:
            model_identifier = model_name
        else:
            assert model is not None
            try:
                # Access model name from config (private attribute, may not exist)
                model_identifier = getattr(model.config, "_name_or_path", "unknown")
            except AttributeError:
                model_identifier = "unknown"

        self.logger.info(f"Applying steering vector for model: {model_identifier}")
        self.logger.info(f"Steering vector path: {steering_vector_path}")
        self.logger.info(f"Input text: {input_text}")
        self.logger.info(f"Strength: {strength}, Temperature: {temperature}")

        try:
            # Validate inputs
            if not steering_vector_path.exists():
                raise FileNotFoundError(
                    f"Steering vector file does not exist: {steering_vector_path}"
                )

            # 1. Load model and tokenizer if not provided
            if model is None:
                assert model_name is not None
                self.logger.info("Loading model and tokenizer...")
                model, tokenizer = self.llm_loader.load_model(model_name)
            else:
                self.logger.info("Using pre-loaded model")

            # 2. Load steering vectors and metadata
            self.logger.info("Loading steering vectors...")
            vectors, metadata = self.vector_store.load_multi_layer(steering_vector_path)

            # 3. Prepare prompt with chat template
            prompt = self._prepare_prompt(input_text, tokenizer)
            if verbose:
                self.logger.info(f"Full prompt after chat template:\n{prompt}")
            else:
                self.logger.debug(f"Prepared prompt: {prompt[:100]}...")

            assert model is not None and tokenizer is not None
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore[call-arg]
            prompt_length = inputs.input_ids.shape[1]
            self.logger.debug(f"Prompt length: {prompt_length} tokens")

            # 4. Register hooks for each layer
            self.logger.info(f"Registering hooks for {len(vectors)} layers...")
            hooks = []
            try:
                for layer_name, steer_vec in vectors.items():
                    layer_module = self._get_layer_module(model, layer_name, metadata)
                    # Resolve strength for this layer
                    layer_strength = self._resolve_layer_strength(strength, layer_name)
                    hook = self._make_steering_hook(
                        prompt_length, steer_vec, layer_strength, orthogonal
                    )
                    handle = layer_module.register_forward_hook(hook)
                    hooks.append(handle)
                    self.logger.debug(
                        f"Registered hook on {layer_name} with strength={layer_strength}"
                    )

                # 5. Generate with steering
                self.logger.info("Generating text with steering...")
                with torch.inference_mode():
                    output_ids = model.generate(  # type: ignore[attr-defined]
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,  # type: ignore[attr-defined]
                        eos_token_id=tokenizer.eos_token_id,  # type: ignore[union-attr]
                        use_cache=False,  # Required when using hooks
                    )

                # 6. Decode and return
                result = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[call-arg]
                generated_text = result.replace(prompt, "").strip()

                self.logger.success("Text generation completed successfully")
                return generated_text

            finally:
                # Always remove hooks
                for handle in hooks:
                    handle.remove()
                self.logger.debug("Removed all hooks")

        except Exception as e:
            self.logger.error(f"Failed to apply steering vector: {e}")
            raise

    def _prepare_prompt(self, input_text: str, tokenizer) -> str:
        """
        Prepare prompt with chat template if available.

        Args:
            input_text: User input text
            tokenizer: HuggingFace tokenizer

        Returns:
            Formatted prompt string
        """
        # Try to use chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": input_text}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.debug("Applied chat template")
                return prompt
            except Exception as e:
                self.logger.debug(f"Chat template failed, using raw text: {e}")

        # Fallback to raw input
        return input_text

    def _get_layer_module(self, model, layer_name: str, metadata: dict):
        """
        Get layer module from model using layer name.

        Args:
            model: PyTorch model
            layer_name: Layer path string
            metadata: Metadata from vector file

        Returns:
            PyTorch module
        """
        try:
            return self.layer_accessor.get_layer(model, layer_name)
        except Exception as e:
            self.logger.error(f"Failed to access layer '{layer_name}': {e}")
            raise

    def _make_steering_hook(
        self,
        prompt_length: int,
        steer_vec: torch.Tensor,
        strength: float,
        orthogonal: bool,
    ):
        """
        Create steering hook function (CAA method).

        This implements the CAA method from the PoC code (CAA.ipynb cell-35).
        Setting prompt_length=0 applies steering to all tokens (BiPO-style).

        Args:
            prompt_length: Length of prompt in tokens (steering applied after this)
            steer_vec: Steering vector tensor
            strength: Multiplication strength
            orthogonal: Use orthogonalized addition method

        Returns:
            Hook function for register_forward_hook()
        """

        def hook(module, input, output):
            # Handle tuple output (some layers return (hidden_states, *extra))
            if isinstance(output, tuple):
                out = output[0]
                extra_outputs = output[1:]
            else:
                out = output
                extra_outputs = ()

            # Clone and ensure floating point
            if not torch.is_floating_point(out):
                out = out.float()
            out = out.clone()

            # Prepare steering vector
            steer = steer_vec.to(device=out.device, dtype=out.dtype)
            steer_reshaped = steer.view(1, 1, -1)  # [1, 1, H] for broadcasting

            # Apply steering to tokens after prompt
            if orthogonal:
                # Orthogonalized addition: remove existing component then add steering
                norm_steer = steer / (steer.norm(p=2) + 1e-8)
                proj_coeff = (out[:, prompt_length:, :] * norm_steer).sum(
                    dim=-1, keepdim=True
                )
                proj = proj_coeff * norm_steer
                out[:, prompt_length:, :] = (
                    out[:, prompt_length:, :] - proj
                ) + strength * steer_reshaped
            else:
                # Simple addition
                out[:, prompt_length:, :] = (
                    out[:, prompt_length:, :] + strength * steer_reshaped
                )

            # Return in original format
            if extra_outputs:
                return (out, *extra_outputs)
            else:
                return out

        return hook

    def _resolve_layer_strength(
        self,
        strength: float | dict[str, float],
        layer_name: str,
        default: float = 1.0,
    ) -> float:
        """
        Resolve strength value for a specific layer.

        Args:
            strength: Global strength or per-layer strength dict
            layer_name: Name of the layer
            default: Default strength if not found in dict

        Returns:
            Strength value for the layer
        """
        if isinstance(strength, dict):
            return strength.get(layer_name, default)
        return strength

    def get_steering_applied_model(
        self,
        steering_vector_path: Path,
        model_name: str | None = None,
        model: nn.Module | None = None,
        tokenizer: AutoTokenizer | None = None,
        strength: float | dict[str, float] = 1.0,
        prompt_length: int = 0,
        orthogonal: bool = False,
        layers: list[str] | None = None,
    ) -> tuple[nn.Module, AutoTokenizer]:
        """
        Apply steering vector hooks to model and return steered model.

        The returned model has a `remove_steering()` method attached for cleanup.
        This method is useful when you want to reuse the same model for multiple
        generations with steering applied.

        Args:
            steering_vector_path: Path to steering vector file
            model_name: HuggingFace model identifier (optional if model provided)
            model: Pre-loaded model (optional if model_name provided)
            tokenizer: Pre-loaded tokenizer (optional if model_name provided)
            strength: Steering strength multiplier. Can be:
                - float: Apply same strength to all layers (default: 1.0)
                - Dict[str, float]: Per-layer strength mapping. Missing layers default to 1.0
            prompt_length: Length of prompt in tokens (0 = apply to all tokens)
            orthogonal: Use orthogonalized addition method
            layers: List of layer names to apply steering. Uses all available layers if None.

        Returns:
            Tuple of (model, tokenizer) where model has remove_steering() method

        Raises:
            ValueError: If neither model_name nor model is provided, or both are provided
            ValueError: If model is provided without tokenizer
            FileNotFoundError: If steering vector file does not exist

        Examples:
            >>> applier = SteeringApplier()
            >>> model, tokenizer = applier.get_steering_applied_model(
            ...     model_name="google/gemma-3-270m-it",
            ...     steering_vector_path=Path("./vector.safetensors"),
            ...     strength=2.0,
            ...     orthogonal=True
            ... )
            >>>
            >>> # Use multiple times
            >>> for prompt in ["Hello", "How are you?"]:
            ...     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore[call-arg]
            ...     outputs = model.generate(**inputs, max_new_tokens=50, use_cache=False)
            ...     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>>
            >>> # Clean up when done
            >>> model.remove_steering()
        """
        # Validate model parameters
        if model is not None and model_name is not None:
            raise ValueError(
                "Cannot provide both 'model' and 'model_name'. Choose one."
            )
        if model is None and model_name is None:
            raise ValueError("Must provide either 'model' or 'model_name'.")
        if model is not None and tokenizer is None:
            raise ValueError("Must provide 'tokenizer' when providing 'model'.")

        # Determine model identifier for logging
        if model_name:
            model_identifier = model_name
        else:
            assert model is not None
            try:
                # Access model name from config (private attribute, may not exist)
                model_identifier = getattr(model.config, "_name_or_path", "unknown")
            except AttributeError:
                model_identifier = "unknown"

        self.logger.info(f"Applying steering hooks to model: {model_identifier}")
        self.logger.info(f"Steering vector path: {steering_vector_path}")
        self.logger.info(f"Strength: {strength}, Prompt length: {prompt_length}")

        try:
            # Validate inputs
            if not steering_vector_path.exists():
                raise FileNotFoundError(
                    f"Steering vector file does not exist: {steering_vector_path}"
                )

            # 1. Load model and tokenizer if not provided
            if model is None:
                assert model_name is not None
                self.logger.info("Loading model and tokenizer...")
                model, tokenizer = self.llm_loader.load_model(model_name)
            else:
                self.logger.info("Using pre-loaded model")

            # 2. Load steering vectors and metadata
            self.logger.info("Loading steering vectors...")
            vectors, metadata = self.vector_store.load_multi_layer(steering_vector_path)
            
            # 2.5. Filter layers if specified
            if layers is not None:
                original_count = len(vectors)
                available_layers = list(vectors.keys())
                vectors = {k: v for k, v in vectors.items() if k in layers}
                filtered_count = len(vectors)
                
                if filtered_count == 0:
                    raise ValueError(
                        f"None of the specified layers found in steering vector. "
                        f"Specified: {layers}, Available: {available_layers}"
                    )
                
                if filtered_count < len(layers):
                    missing = set(layers) - set(available_layers)
                    self.logger.warning(
                        f"Some specified layers not found: {missing}. "
                        f"Using {filtered_count}/{len(layers)} layers."
                    )
                
                self.logger.info(
                    f"Filtered to {filtered_count}/{original_count} layers"
                )

            # 3. Register hooks for each layer
            self.logger.info(f"Registering hooks for {len(vectors)} layers...")
            hooks = []
            for layer_name, steer_vec in vectors.items():
                layer_module = self._get_layer_module(model, layer_name, metadata)
                # Resolve strength for this layer
                layer_strength = self._resolve_layer_strength(strength, layer_name)
                hook = self._make_steering_hook(
                    prompt_length, steer_vec, layer_strength, orthogonal
                )
                handle = layer_module.register_forward_hook(hook)
                hooks.append(handle)
                self.logger.debug(
                    f"Registered hook on {layer_name} with strength={layer_strength}"
                )

            # 4. Store handles in model and add cleanup method
            model._steering_handles = hooks  # type: ignore[attr-defined]

            def remove_steering():
                """Remove all steering hooks from this model."""
                if hasattr(model, "_steering_handles"):
                    for handle in model._steering_handles:  # type: ignore[attr-defined]
                        handle.remove()
                    del model._steering_handles  # type: ignore[attr-defined]
                    self.logger.info("Removed all steering hooks")
                else:
                    self.logger.warning("No steering hooks found on model")

            model.remove_steering = remove_steering  # type: ignore[attr-defined]

            self.logger.success(
                f"Successfully applied steering hooks to {len(vectors)} layers"
            )
            assert model is not None and tokenizer is not None
            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to apply steering hooks: {e}")
            raise
