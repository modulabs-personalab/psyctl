"""
Dataset Builder for Personality Steering Vector Extraction

This module implements the DatasetBuilder class which creates steering datasets
for personality steering vector extraction. These datasets are compatible with multiple
extraction methods including mean_diff and BiPO.

Key Concepts:
- Steering Dataset: Raw data (situation, character, positive/neutral responses) for training
- Extraction Methods: mean_diff (Mean Difference from CAA paper) and BiPO (preference optimization)
- Application Method: CAA (Contrastive Activation Addition) - adds extracted vectors to activations
- Personality Steering: Modifying LLM behavior to exhibit specific personality characteristics

Workflow:
1. Load a base model and tokenizer
2. Load conversation dataset (allenai/soda)
3. Generate personality-specific prompts using P2 class
4. Create contrastive pairs (positive vs neutral personality)
5. Save raw components in JSONL format

Example Usage:
    builder = DatasetBuilder()
    builder.build_steer_dataset(
        model="meta-llama/Llama-3.2-3B-Instruct",
        personality="Extroversion",
        output_dir=Path("./dataset"),
        limit_samples=1000
    )

References:
- CAA Paper: https://arxiv.org/abs/2312.06681
- BiPO Paper: https://arxiv.org/abs/2406.00045
- SoDA Dataset: https://huggingface.co/datasets/allenai/soda
"""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import jinja2
import torch
from datasets import load_dataset  # type: ignore[import-not-found]
from tqdm import tqdm

from psyctl.config import CHECKPOINT_INTERVAL, INFERENCE_BATCH_SIZE
from psyctl.core.logger import get_logger
from psyctl.core.prompt import P2
from psyctl.core.prompt_openrouter import P2OpenRouter
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.openrouter_client import OpenRouterClient

# PSYCTL branding
PSYCTL_LOGO_URL = "https://cdn.caveduck.io/cdn-cgi/image/anim=false,dpr=1.5,f=auto,w=400/charim/5eaf363a-94b4-4b6c-bd79-e8bf4008af70"


class DatasetBuilder:
    """
    Build steering datasets for personality steering vector extraction.

    This class implements steering dataset generation for multiple extraction methods
    (CAA, BiPO, etc.). It creates training data by comparing model responses with different
    personality traits, enabling the extraction of steering vectors that can modify
    model behavior to exhibit specific personality characteristics.

    Attributes:
        llm_loader (LLMLoader): Loader for Hugging Face models
        p2 (P2): Personality prompt generator
        logger: Logger instance for debugging and monitoring
        dataset: Loaded conversation dataset (allenai/soda)
        model: Loaded language model
        tokenizer: Model tokenizer
        personality (str): Target personality trait for steering

    Methods:
        build_steer_dataset: Main method to build steering dataset
        _load_model: Load model and tokenizer
        _load_dataset: Load conversation dataset
        _generate_sample_context: Generate conversation contexts
        _get_answer: Generate personality-specific responses
        _gen_caa_data: Create contrastive data pairs
        _save_sample_to_jsonl: Save data to JSONL file
        _build_caa_dataset: Core dataset building logic
    """

    def __init__(
        self,
        use_openrouter: bool = False,
        openrouter_api_key: str | None = None,
        openrouter_max_workers: int = 1,
        roleplay_prompt_template: str | None = None,
    ):
        """
        Initialize DatasetBuilder with required components.

        Initializes the LLM loader, logger, and placeholder attributes for
        model, tokenizer, dataset, and P2 personality generator.

        Args:
            use_openrouter (bool): Whether to use OpenRouter API instead of local model
            openrouter_api_key (str): OpenRouter API key (required if use_openrouter=True)
            openrouter_max_workers (int): Number of parallel workers for OpenRouter (1 = sequential)
            roleplay_prompt_template (str): Path to custom Jinja2 template for roleplay prompts (optional)
        """
        self.use_openrouter = use_openrouter
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_max_workers = openrouter_max_workers
        self.openrouter_client = None
        self.active_model = None  # Store the active model being used

        self.llm_loader = LLMLoader()
        self.p2 = None
        self.logger = get_logger("dataset_builder")
        self.dataset = None
        self.model = None
        self.tokenizer = None
        self.write_lock = threading.Lock()
        self.checkpoint_data = []

        # Generation parameters (will be set in build_caa_dataset)
        self.temperature = 0.01
        self.top_k = None
        self.top_p = None
        self.max_tokens = 100

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.PackageLoader("psyctl", "templates"),
            autoescape=jinja2.select_autoescape(),
        )

        # Store custom template path
        self.roleplay_prompt_template_path = roleplay_prompt_template

        # Validate OpenRouter configuration
        if self.use_openrouter and not self.openrouter_api_key:
            raise ValueError("OpenRouter API key is required when use_openrouter=True")

    def build_steer_dataset(
        self,
        model: str,
        personality: str,
        output_dir: Path,
        limit_samples: int,
        dataset_name: str = "allenai/soda",
        temperature: float = 0.01,
        top_k: int | None = None,
        top_p: float | None = None,
        max_tokens: int = 100,
        dtype: str = None,
    ) -> Path:
        """
        Build steering dataset for given personality traits.

        This is the main entry point for steering dataset generation. It orchestrates
        the entire process from model loading to dataset creation.

        Args:
            model (str): Hugging Face model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
            personality (str): Target personality trait (e.g., "Extroversion", "Introversion")
            output_dir (Path): Directory to save the generated dataset
            limit_samples (int): Maximum number of samples to generate (0 for unlimited)
            dataset_name (str): Hugging Face dataset identifier (default: "allenai/soda")
            temperature (float): Sampling temperature for generation (default: 0)
            top_k (int): Top-k sampling parameter (default: None)
            top_p (float): Top-p (nucleus) sampling parameter (default: None)
            max_tokens (int): Maximum tokens to generate per response (default: 100)

        Returns:
            Path: Path to the generated JSONL file

        Raises:
            Exception: If any step in the dataset building process fails

        Example:
            >>> builder = DatasetBuilder()
            >>> output_file = builder.build_steer_dataset(
            ...     model="meta-llama/Llama-3.2-3B-Instruct",
            ...     personality="Extroversion",
            ...     output_dir=Path("./dataset"),
            ...     limit_samples=1000,
            ...     temperature=0.5
            ... )
            >>> print(f"Generated dataset: {output_file}")
        """
        # Store generation parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.logger.info(f"Building steering dataset for model: {model}")
        self.logger.info(f"Personality traits: {personality}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Dataset: {dataset_name}")
        self.logger.info(
            f"Generation params: temperature={temperature}, top_k={top_k}, top_p={top_p}, max_tokens={max_tokens}"
        )

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_dir}")
            self.personality = personality
            self.dataset_name = dataset_name

            # 1. Load model or initialize OpenRouter client
            if self.use_openrouter:
                self.active_model = model
                self.logger.info(
                    f"Using OpenRouter API with model: {self.active_model}"
                )
                self.logger.debug(
                    f"OpenRouter max workers: {self.openrouter_max_workers}"
                )
                assert self.openrouter_api_key is not None, (
                    "API key must be set for OpenRouter"
                )
                self.openrouter_client = OpenRouterClient(
                    api_key=self.openrouter_api_key
                )
                self.logger.debug(
                    f"Initializing P2OpenRouter with model: {self.active_model}"
                )
                self.p2 = P2OpenRouter(
                    client=self.openrouter_client, model=self.active_model
                )
            else:
                self.active_model = model
                self._load_model(model, dtype=dtype)
                assert self.model is not None and self.tokenizer is not None
                self.p2 = P2(self.model, self.tokenizer)

            # 2. Load dataset
            self._load_dataset(dataset_name)

            # 3. Build steering dataset
            output_file = self._build_caa_dataset(output_dir, limit_samples)

            self.logger.info("Finished building steering dataset")

            # Log OpenRouter usage if applicable
            if self.use_openrouter:
                assert self.openrouter_client is not None
                self.logger.info(
                    f"Total OpenRouter requests: {self.openrouter_client.get_total_requests()}"
                )
                self.logger.info(
                    f"Total OpenRouter cost: ${self.openrouter_client.get_total_cost():.6f}"
                )

            return output_file

        except Exception as e:
            self.logger.error(f"Failed to build steering dataset: {e}")
            raise

    # Alias for backward compatibility
    def build_caa_dataset(self, *args, **kwargs) -> Path:
        """Deprecated: Use build_steer_dataset() instead."""
        self.logger.warning(
            "build_caa_dataset() is deprecated. Use build_steer_dataset() instead."
        )
        return self.build_steer_dataset(*args, **kwargs)

    def _load_model(self, model_name: str, dtype: str = None) -> None:
        """
        Load model and tokenizer from Hugging Face.

        Args:
            model_name (str): Hugging Face model identifier

        Raises:
            Exception: If model loading fails
        """
        self.model, self.tokenizer = self.llm_loader.load_model(model_name, None, dtype)
        self.logger.info(f"Loaded model: {model_name}")
        self.logger.info(f"Loaded tokenizer: {model_name}")

    def _load_dataset(self, dataset_name: str = "allenai/soda") -> None:
        """
        Load a conversation dataset from Hugging Face.

        Loads the specified dataset which should contain conversational data
        with speakers, dialogue, and narrative context fields.

        Args:
            dataset_name (str): Hugging Face dataset identifier (default: "allenai/soda")

        Raises:
            Exception: If dataset loading fails
        """
        # Log HF_TOKEN status for debugging
        import os

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # Mask the token for security (show first 4 and last 4 characters)
            masked_token = (
                f"{hf_token[:4]}...{hf_token[-4:]}" if len(hf_token) > 8 else "***"
            )
            self.logger.info(f"HF_TOKEN found: {masked_token}")
        else:
            self.logger.warning("HF_TOKEN not found in environment variables")

        try:
            dataset = load_dataset(dataset_name, split="train")
            self.dataset = dataset
            self.logger.info(f"Loaded dataset: {dataset_name}")

            # Validate required fields
            if len(dataset) > 0:  # type: ignore[arg-type]
                sample = dataset[0]  # type: ignore[index]
                required_fields = ["speakers", "dialogue", "narrative"]
                missing_fields = [
                    field for field in required_fields if field not in sample
                ]
                if missing_fields:
                    self.logger.warning(
                        f"Dataset may be missing required fields: {missing_fields}. "
                        f"Expected fields: {required_fields}"
                    )
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _generate_sample_context(
        self, limit_samples: int = 0
    ) -> Generator[dict[str, str], None, None]:
        """
        Generate conversation contexts from the SoDA dataset.

        Iterates through the dataset to extract valid conversation contexts
        with at least 2 speakers and narrative content. Creates structured
        context data for personality-specific response generation.

        Args:
            limit_samples (int): Maximum number of contexts to generate (0 for unlimited)

        Yields:
            Dict[str, str]: Context dictionary with keys:
                - char_name: Character who will respond
                - user_name: Character who asked the question
                - situation: Combined narrative and dialogue context

        Note:
            Filters out entries with insufficient speakers or missing narrative
        """
        num_generated = 0

        # Calculate total iterations for tqdm
        assert self.dataset is not None
        total = len(self.dataset) if limit_samples == 0 else limit_samples  # type: ignore[arg-type]
        pbar = tqdm(range(len(self.dataset)), desc="Generating samples", total=total)  # type: ignore[arg-type]
        for idx in pbar:
            data = self.dataset[idx]  # type: ignore[index]
            # 화자 최소 2명 보장
            if len(data["speakers"]) < 2 or len(data["dialogue"]) < 1:  # type: ignore[arg-type]
                continue
            asker = data["speakers"][0]  # type: ignore[index]
            answerer = data["speakers"][1]  # type: ignore[index] # 두 번째 인물
            narrative = data["narrative"] or ""
            if narrative == "":
                continue
            query = data["dialogue"][0]  # type: ignore[index]
            situation = f"{narrative}\n{asker}: {query}\n"
            yield {"char_name": answerer, "user_name": asker, "situation": situation}  # type: ignore[misc]

            num_generated += 1
            # Update progress bar description with actual count
            pbar.set_description(f"Generating samples ({num_generated})")

            if limit_samples > 0 and num_generated >= limit_samples:
                break
        pbar.close()

        self.logger.info("Finished generating samples.")

    def _load_template(
        self, template_name: str, custom_template_path: str | None = None
    ) -> jinja2.Template:
        """
        Load a Jinja2 template.

        Loads a custom template from file if provided, otherwise loads the default
        template from the templates directory. Also checks for in-memory custom templates
        set via set_*_template() methods.

        Args:
            template_name (str): Name of the default template file (e.g., 'md_question.j2', 'roleplay_prompt.j2')
            custom_template_path (str): Path to custom template file (optional)

        Returns:
            jinja2.Template: Loaded template object

        Raises:
            FileNotFoundError: If custom template path is provided but file doesn't exist
        """
        # Check for in-memory custom templates first
        if template_name == "roleplay_prompt.j2" and hasattr(
            self, "_custom_roleplay_template"
        ):
            return self.jinja_env.from_string(self._custom_roleplay_template)

        # Load from file if path provided
        if custom_template_path:
            custom_path = Path(custom_template_path)
            if not custom_path.exists():
                raise FileNotFoundError(
                    f"Custom template not found: {custom_template_path}"
                )
            with Path(custom_path).open(encoding="utf-8") as f:
                template_content = f.read()
            return self.jinja_env.from_string(template_content)

        # Load default template
        return self.jinja_env.get_template(template_name)

    def get_roleplay_prompt_template(self) -> str:
        """
        Get the current roleplay prompt template content as string.

        Returns:
            str: Template content as string

        Example:
            >>> builder = DatasetBuilder()
            >>> template_str = builder.get_roleplay_prompt_template()
            >>> print(template_str)
        """
        # Check for in-memory custom template first
        if hasattr(self, "_custom_roleplay_template"):
            return self._custom_roleplay_template

        # Load from file if custom path provided
        if self.roleplay_prompt_template_path:
            with Path(self.roleplay_prompt_template_path).open(encoding="utf-8") as f:
                return f.read()

        # Load default template from package by reading the source file
        import importlib.resources

        try:
            # Python 3.9+
            template_content = (
                importlib.resources.files("psyctl.templates")
                .joinpath("roleplay_prompt.j2")
                .read_text(encoding="utf-8")
            )
        except AttributeError:
            # Fallback for older Python versions
            with (
                importlib.resources.path(
                    "psyctl.templates", "roleplay_prompt.j2"
                ) as template_path,
                Path(template_path).open(encoding="utf-8") as f,
            ):
                template_content = f.read()
        return template_content

    def set_roleplay_prompt_template(self, template_content: str) -> None:
        """
        Set a custom roleplay prompt template from string content.

        This method allows setting a template directly from a string without
        needing to save it to a file first.

        Args:
            template_content (str): Jinja2 template content as string

        Example:
            >>> builder = DatasetBuilder()
            >>> custom_template = '''
            ... # Overview
            ... You are {{ char_name }}.
            ... User is {{ user_name }}.
            ...
            ... # About {{ char_name }}
            ... {{ p2 }}
            ...
            ... # Situation
            ... {{ situation }}
            ... '''
            >>> builder.set_roleplay_prompt_template(custom_template)
        """
        # Store template content in a temporary attribute
        self._custom_roleplay_template = template_content
        # Clear file path since we're using string template
        self.roleplay_prompt_template_path = None

    def _get_answer(
        self,
        user_name: str,
        char_name: str,
        p2: str,
        situation: str,
        verbose: bool = False,
    ) -> str:
        """
        Generate personality-specific response for a given situation.

        Creates a role-playing prompt that instructs the model to respond as a character
        with specific personality traits in a given conversational context.

        Args:
            user_name (str): Name of the user/asker in the conversation
            char_name (str): Name of the character who will respond
            p2 (str): Personality description generated by P2 class
            situation (str): Conversational context and situation
            verbose (bool): Whether to print the generated prompt for debugging

        Returns:
            str: Generated response from the model

        Note:
            The prompt structure follows a role-playing format with clear instructions
            for the model to adopt the character's personality and respond appropriately.
        """
        # Load and render template
        template = self._load_template(
            "roleplay_prompt.j2", self.roleplay_prompt_template_path
        )
        prompt = template.render(
            user_name=user_name, char_name=char_name, p2=p2, situation=situation
        )
        if verbose:
            print(prompt)

        # OpenRouter mode
        if self.use_openrouter:
            try:
                assert self.openrouter_client is not None
                assert self.active_model is not None
                _, output_text = self.openrouter_client.generate(
                    prompt=prompt,
                    model=self.active_model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )
                return output_text
            except Exception as e:
                self.logger.error(f"OpenRouter generation failed: {e}")
                return ""

        # Local model mode
        # Use the same approach as P2._get_result
        messages = [{"role": "user", "content": prompt}]

        # 1. Convert user message to chat template
        try:
            assert self.tokenizer is not None
            tokenized_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors=None,
            )
            tokenized = self.tokenizer(
                tokenized_input, return_tensors="pt", add_special_tokens=False
            )
        except Exception:
            assert self.tokenizer is not None
            tokenized = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )

        # Move tensors to the same device as the model
        assert self.model is not None and self.tokenizer is not None
        device = next(self.model.parameters()).device
        tokenized["input_ids"] = tokenized["input_ids"].to(device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

        # 2. Generate response
        generate_kwargs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "max_new_tokens": self.max_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.top_k is not None:
            generate_kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            generate_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(**generate_kwargs)

        # 3. Decode the generated text
        len_input = tokenized["input_ids"][0].shape[0]
        output_text = self.tokenizer.decode(
            outputs[0, len_input:], skip_special_tokens=True
        )
        return output_text

    def _get_batch_answers(
        self,
        batch_contexts: list[tuple[str, str, str, str]],
        batch_size: int | None = None,
    ) -> list[str]:
        """
        Generate personality-specific responses for multiple contexts in batches.

        This method processes multiple contexts simultaneously to improve efficiency
        by batching model inference operations. It handles tokenization, padding,
        and generation for multiple prompts at once.

        Args:
            batch_contexts (List[Tuple[str, str, str, str]]): List of context tuples
                Each tuple contains: (user_name, char_name, p2, situation)
            batch_size (int, optional): Batch size for inference. Uses config default if None.

        Returns:
            List[str]: Generated responses for each context in the batch

        Note:
            Uses dynamic padding and attention masks to handle variable length inputs
            efficiently. Falls back to individual processing if batch inference fails.
        """
        if batch_size is None:
            batch_size = INFERENCE_BATCH_SIZE

        if not batch_contexts:
            return []

        # Prepare prompts for all contexts using template
        template = self._load_template(
            "roleplay_prompt.j2", self.roleplay_prompt_template_path
        )
        prompts = []
        for user_name, char_name, p2, situation in batch_contexts:
            prompt = template.render(
                user_name=user_name, char_name=char_name, p2=p2, situation=situation
            )
            prompts.append(prompt)

        # OpenRouter mode
        if self.use_openrouter:
            assert self.openrouter_client is not None
            assert self.active_model is not None
            results = self.openrouter_client.generate_batch(
                prompts=prompts,
                model=self.active_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_workers=self.openrouter_max_workers,
            )
            return [text for _, text in results]

        # Local model mode - Process in batches
        all_responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            try:
                # Tokenize batch
                messages_batch = [
                    [{"role": "user", "content": prompt}] for prompt in batch_prompts
                ]

                tokenized_inputs = []
                for messages in messages_batch:
                    try:
                        assert self.tokenizer is not None
                        tokenized_input = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            return_tensors=None,
                        )
                        tokenized_inputs.append(tokenized_input)
                    except Exception:
                        assert self.tokenizer is not None
                        tokenized_inputs.append(batch_prompts[len(tokenized_inputs)])

                # Batch tokenization with padding
                assert self.tokenizer is not None
                tokenized = self.tokenizer(
                    tokenized_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    add_special_tokens=False,
                )

                # Move to device
                assert self.model is not None and self.tokenizer is not None
                device = next(self.model.parameters()).device
                tokenized["input_ids"] = tokenized["input_ids"].to(device)
                tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

                # Generate responses
                with torch.no_grad():
                    generate_kwargs = {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "max_new_tokens": self.max_tokens,
                        "do_sample": True,
                        "temperature": self.temperature,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "num_return_sequences": 1,
                    }
                    if self.top_k is not None:
                        generate_kwargs["top_k"] = self.top_k
                    if self.top_p is not None:
                        generate_kwargs["top_p"] = self.top_p

                    outputs = self.model.generate(**generate_kwargs)

                # Decode responses
                batch_responses = []
                for j, output in enumerate(outputs):
                    len_input = tokenized["input_ids"][j].shape[0]
                    output_text = self.tokenizer.decode(
                        output[len_input:], skip_special_tokens=True
                    )
                    batch_responses.append(output_text)

                all_responses.extend(batch_responses)

            except Exception as e:
                self.logger.warning(
                    f"Batch inference failed, falling back to individual: {e}"
                )
                # Fallback to individual processing
                for prompt in batch_prompts:
                    try:
                        # Extract context from original batch_contexts
                        ctx_idx = prompts.index(prompt)
                        user_name, char_name, p2, situation = batch_contexts[ctx_idx]
                        response = self._get_answer(user_name, char_name, p2, situation)
                        all_responses.append(response)
                    except Exception as fallback_e:
                        self.logger.error(
                            f"Individual fallback also failed: {fallback_e}"
                        )
                        all_responses.append("")

        return all_responses

    def _save_sample_to_jsonl(self, sample: dict[str, str], output_file: Path) -> None:
        """
        Save CAA data samples to JSONL file.

        Appends the generated contrastive data pairs to a JSONL file
        for later use in training steering vector extraction models.

        Args:
            sample (Dict[str, str]): Dictionary containing question, positive, and neutral keys
            output_file (Path): Path to the output JSONL file

        Note:
            Uses UTF-8 encoding and ensures proper JSON formatting with
            non-ASCII character support.
        """
        with self.write_lock, Path(output_file).open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_batch_to_jsonl(
        self, samples: list[dict[str, str]], output_file: Path
    ) -> None:
        """
        Save multiple CAA data samples to JSONL file.

        Args:
            samples (List[Dict[str, str]]): List of samples to save
            output_file (Path): Path to the output JSONL file
        """
        with self.write_lock, Path(output_file).open("a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, output_file: Path, num_generated: int) -> None:
        """
        Save checkpoint data for resuming dataset generation.

        Args:
            output_file (Path): Path to the output JSONL file
            num_generated (int): Number of samples generated so far
        """
        checkpoint_file = output_file.with_suffix(".checkpoint.json")
        checkpoint_data = {
            "num_generated": num_generated,
            "output_file": str(output_file),
            "timestamp": datetime.now().isoformat(),
        }

        with Path(checkpoint_file).open("w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Checkpoint saved: {num_generated} samples generated")

    def _load_checkpoint(self, output_file: Path) -> dict | None:
        """
        Load checkpoint data if available.

        Args:
            output_file (Path): Path to the output JSONL file

        Returns:
            Dict | None: Checkpoint data if available, None otherwise
        """
        checkpoint_file = output_file.with_suffix(".checkpoint.json")
        if checkpoint_file.exists():
            try:
                with Path(checkpoint_file).open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return None

    def _build_caa_dataset(self, output_dir: Path, limit_samples: int) -> Path:
        """
        Core steering dataset building logic with batch processing.

        This is the main implementation of the steering dataset generation process.
        It processes multiple contexts in batches, generates personality-specific
        responses using P2 prompts, creates contrastive pairs, and saves them
        to a timestamped JSONL file with checkpoint support.

        Args:
            output_dir (Path): Directory to save the generated dataset
            limit_samples (int): Maximum number of samples to generate

        Returns:
            Path: Path to the generated JSONL file

        Note:
            Uses batch processing for improved efficiency, checkpoint support
            for resuming interrupted runs.
        """

        self.logger.info("Building steering dataset with batch processing...")
        self.logger.info(f"Limit samples: {limit_samples}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Batch size: {INFERENCE_BATCH_SIZE}")

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"caa_dataset_{datetime_str}.jsonl"
        self.logger.info(f"Output file: {output_file}")

        # Check for existing checkpoint
        checkpoint = self._load_checkpoint(output_file)
        num_generated = checkpoint["num_generated"] if checkpoint else 0

        if checkpoint:
            self.logger.info(
                f"Resuming from checkpoint: {num_generated} samples already generated"
            )

        # Generate personality-specific character descriptions using P2
        self.logger.debug(f"Generating positive P2 for personality: {self.personality}")
        assert self.p2 is not None
        positive_p2 = self.p2.build("Xylo", self.personality)
        self.logger.debug(f"Positive P2 generated: {positive_p2[:200]}...")

        self.logger.debug("Generating neutral P2")
        neutral_p2 = self.p2.build("Xylo", "Normal")
        self.logger.debug(f"Neutral P2 generated: {neutral_p2[:200]}...")

        # Create templates with placeholder for character name
        positive_template = positive_p2.replace("Xylo", "{{char}}").replace(
            "Xylo", "{{char}}"
        )
        neutral_template = neutral_p2.replace("Xylo", "{{char}}").replace(
            "Xylo", "{{char}}"
        )

        # Collect contexts in batches
        batch_size = (
            INFERENCE_BATCH_SIZE // 2
        )  # Each context generates 2 inference calls

        self._build_caa_dataset_sync(
            output_file,
            limit_samples,
            num_generated,
            positive_template,
            neutral_template,
            batch_size,
        )

        return output_file

    def _build_caa_dataset_sync(
        self,
        output_file: Path,
        limit_samples: int,
        num_generated: int,
        positive_template: str,
        neutral_template: str,
        batch_size: int,
    ) -> int:
        """Synchronous batch processing implementation."""

        context_batch = []

        for context in self._generate_sample_context(limit_samples):
            if num_generated >= limit_samples > 0:
                break

            context_batch.append(context)

            if len(context_batch) >= batch_size:
                num_generated += self._process_context_batch_sync(
                    context_batch,
                    output_file,
                    positive_template,
                    neutral_template,
                    num_generated,
                )
                context_batch = []

                # Save checkpoint
                if num_generated % CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(output_file, num_generated)

        # Process remaining contexts
        if context_batch:
            num_generated += self._process_context_batch_sync(
                context_batch,
                output_file,
                positive_template,
                neutral_template,
                num_generated,
            )

        self.logger.info(
            f"Finished building steering dataset. Total samples: {num_generated}"
        )
        return num_generated

    def _process_context_batch_sync(
        self,
        context_batch: list[dict],
        output_file: Path,
        positive_template: str,
        neutral_template: str,
        start_idx: int,
    ) -> int:
        """Process a batch of contexts synchronously."""

        # Prepare batch contexts for inference
        batch_contexts = []
        for context in context_batch:
            user_name = context["user_name"]
            char_name = context["char_name"]
            situation = context["situation"]

            positive = positive_template.replace("{{char}}", char_name)
            neutral = neutral_template.replace("{{char}}", char_name)

            # Add both positive and neutral contexts
            batch_contexts.append((user_name, char_name, positive, situation))
            batch_contexts.append((user_name, char_name, neutral, situation))

        # Get batch responses
        responses = self._get_batch_answers(batch_contexts)

        # Process responses and create samples
        samples = []
        for i, context in enumerate(context_batch):
            char_name = context["char_name"]
            situation = context["situation"]

            answer_positive = responses[i * 2]
            answer_neutral = responses[i * 2 + 1]

            sample = {}

            # Store raw components only
            sample["situation"] = situation
            sample["char_name"] = char_name

            # Always store answers consistently (no swapping)
            # BiPO relies on field names, not answer order
            sample["positive"] = answer_positive
            sample["neutral"] = answer_neutral

            samples.append(sample)

        # Save all samples
        self._save_batch_to_jsonl(samples, output_file)

        return len(samples)

    def _generate_dataset_card(
        self,
        personality: str,
        model: str,
        num_samples: int,
        timestamp: str,
        dataset_source: str = "allenai/soda",
        license: str | None = None,
        repo_id: str | None = None,
    ) -> str:
        """
        Generate HuggingFace dataset card with PSYCTL branding.

        Args:
            personality: Target personality trait
            model: Model used for generation
            num_samples: Number of samples in dataset
            timestamp: Generation timestamp (ISO format)
            dataset_source: Source dataset used
            license: License identifier (e.g., 'mit', 'apache-2.0', 'cc-by-4.0')
            repo_id: HuggingFace repository ID for usage examples

        Returns:
            str: Markdown content for README.md
        """
        # Build YAML front matter
        yaml_parts = ["---"]
        if license:
            yaml_parts.append(f"license: {license}")

        # Detect language from dataset source
        language = (
            "ko"
            if "korean" in dataset_source.lower() or "_kr" in dataset_source.lower()
            else "en"
        )

        # Determine size category
        if num_samples < 100 or num_samples < 1000:
            size_category = "n<1K"
        elif num_samples < 10000:
            size_category = "1K<n<10K"
        elif num_samples < 100000:
            size_category = "10K<n<100K"
        else:
            size_category = "100K<n<1M"

        yaml_parts.extend(["language:", f"- {language}", "tags:"])
        yaml_header = "\n".join(yaml_parts)

        return f"""{yaml_header}
- psyctl
- caa
- personality-steering
- contrastive-activation-addition
- {personality.lower().replace(" ", "-")}
task_categories:
- text-generation
size_categories:
- {size_category}
---


## 📊 Dataset Overview

This dataset contains **{
            num_samples
        } samples** designed for extracting personality steering vectors using the **Contrastive Activation Addition (CAA)** method. Each sample presents a scenario with two response options: one exhibiting the target personality trait and one neutral.

### Dataset Details

| Property | Value |
|----------|-------|
| **Personality Trait** | {personality} |
| **Generation Model** | `{model}` |
| **Source Dataset** | `{dataset_source}` |
| **Sample Count** | {num_samples} |
| **Generated** | {timestamp} |
| **Format** | JSONL |

---

## 🎯 Intended Use

### Primary Use Case
Extract steering vectors to modify LLM behavior to exhibit **{personality}** traits.

### Workflow
1. **Dataset Generation** (this dataset) [DONE]
2. **Vector Extraction**: Use PSYCTL `extract.steering` command
3. **Personality Application**: Apply vectors with `steering` command
4. **Evaluation**: Test with psychological inventories

---

## 📝 Dataset Structure

### Fields
- **situation**: Scenario description and dialogue context
- **char_name**: Character name in the scenario
- **positive**: Response exhibiting the target personality trait
- **neutral**: Response with neutral/baseline personality

### Example
```json
{{{{
  "situation": "Alice is at a party and someone asks her to join the dance floor.\\nFriend: Hey Alice, want to come dance with us?\\n",
  "char_name": "Alice",
  "positive": "Absolutely! I'd love to—let's get everyone together and make it a group thing!",
  "neutral": "Sure, I'll join you."
}}}}
```

---

## 🚀 Usage with PSYCTL

### Installation
```bash
pip install psyctl
```

### Extract Steering Vector
```bash
psyctl extract.steering \\
  --model "meta-llama/Llama-3.2-3B-Instruct" \\
  --layer "model.layers[13].mlp.down_proj" \\
  --dataset "{repo_id or "YOUR_USERNAME/repo-name"}" \\
  --output "./vectors/steering_vector.safetensors"
```

### Apply Personality Steering
```bash
psyctl steering \\
  --model "meta-llama/Llama-3.2-3B-Instruct" \\
  --steering-vector "./vectors/steering_vector.safetensors" \\
  --input-text "How should I approach this situation?"
```

---

## 📚 References

- **PSYCTL**: [GitHub Repository](https://github.com/modulabs-personalab/psyctl)
- **CAA Paper**: [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- **P2 Paper**: [Evaluating and Inducing Personality](https://arxiv.org/abs/2206.07550)
- **Source Dataset**: [{dataset_source}](https://huggingface.co/datasets/{
            dataset_source
        })

---
{
            ""
            if not license
            else f'''
## 📄 License

{license.upper()} License - See [LICENSE](LICENSE) for details.

---
'''
        }
<div align="center">
  <sub>
    Generated with ❤️ by <a href="https://github.com/modulabs-personalab/psyctl">PSYCTL</a>
  </sub>
</div>
"""

    def upload_to_hub(
        self,
        jsonl_file: Path,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Upload steering dataset via PSYCTL",
        token: str | None = None,
        license: str | None = None,
        personality: str | None = None,
        model: str | None = None,
        dataset_source: str | None = None,
    ) -> str:
        """
        Upload steering dataset to HuggingFace Hub with PSYCTL branding.

        Args:
            jsonl_file: Path to JSONL dataset file
            repo_id: HuggingFace repository ID (username/repo-name)
            private: Make repository private (default: False)
            commit_message: Commit message for upload
            token: HuggingFace token (uses HF_TOKEN env if None)
            license: License identifier (e.g., 'mit', 'apache-2.0', 'cc-by-4.0')
            personality: Personality trait for dataset card (optional)
            model: Model name used to generate dataset (optional)
            dataset_source: Source dataset used (optional)

        Returns:
            str: Repository URL

        Raises:
            ValueError: If repo_id format is invalid
            FileNotFoundError: If jsonl_file doesn't exist

        Example:
            >>> builder = DatasetBuilder()
            >>> url = builder.upload_to_hub(
            ...     jsonl_file=Path("./dataset/caa_dataset_20250107.jsonl"),
            ...     repo_id="username/extroversion-caa",
            ...     private=False,
            ...     license="mit",
            ...     personality="Extroversion",
            ...     model="meta-llama/Llama-3.2-3B-Instruct"
            ... )
            >>> print(f"Uploaded to: {url}")
        """
        from datasets import Dataset  # type: ignore[import-not-found]
        from huggingface_hub import HfApi

        # Validate inputs
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")

        if "/" not in repo_id:
            raise ValueError(
                f"Invalid repo_id format: '{repo_id}'. "
                "Expected format: 'username/repo-name'"
            )

        # Use provided token or environment variable
        if token is None:
            from psyctl.core.utils import validate_hf_token

            token = validate_hf_token()

        self.logger.info(f"Loading dataset from: {jsonl_file}")

        # Load JSONL to Dataset
        data = []
        with Path(jsonl_file).open(encoding="utf-8") as f:
            data.extend(json.loads(line) for line in f)

        dataset = Dataset.from_list(data)
        self.logger.info(f"Loaded {len(dataset)} samples")

        # Extract metadata from filename or use defaults
        # Format: caa_dataset_20250107_143022.jsonl
        timestamp = datetime.now().isoformat()

        # Generate dataset card
        card_content = self._generate_dataset_card(
            personality=personality or self.personality or "Unknown",
            model=model or self.active_model or "Unknown",
            num_samples=len(dataset),
            timestamp=timestamp,
            dataset_source=dataset_source or self.dataset_name or "allenai/soda",
            license=license,
            repo_id=repo_id,
        )

        # Create README.md
        readme_path = jsonl_file.parent / "README.md"
        with Path(readme_path).open("w", encoding="utf-8") as f:
            f.write(card_content)

        self.logger.info(f"Generated dataset card: {readme_path}")

        # Upload to Hub
        self.logger.info(f"Uploading to HuggingFace Hub: {repo_id}")
        self.logger.info(f"Privacy: {'Private' if private else 'Public'}")

        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                private=private,
                token=token,
                commit_message=commit_message,
            )

            # Upload README separately
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add PSYCTL dataset card",
            )

            repo_url = f"https://huggingface.co/datasets/{repo_id}"
            self.logger.info(f"Successfully uploaded to: {repo_url}")

            return repo_url

        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of DatasetBuilder for steering dataset generation.

    This demonstrates how to use the DatasetBuilder class to create
    personality steering datasets for different personality traits.
    """

    from pathlib import Path

    # Initialize builder
    builder = DatasetBuilder()

    # Example: Build dataset for extroversion personality
    try:
        num_samples = builder.build_caa_dataset(
            model="meta-llama/Llama-3.2-3B-Instruct",
            personality="Extroversion",
            output_dir=Path("./dataset/extroversion"),
            limit_samples=100,
        )
        print(f"Successfully generated {num_samples} samples for Extroversion")

    except Exception as e:
        print(f"Failed to build dataset: {e}")

    # Example: Build dataset for introversion personality
    try:
        num_samples = builder.build_caa_dataset(
            model="meta-llama/Llama-3.2-3B-Instruct",
            personality="Introversion",
            output_dir=Path("./dataset/introversion"),
            limit_samples=100,
        )
        print(f"Successfully generated {num_samples} samples for Introversion")

    except Exception as e:
        print(f"Failed to build dataset: {e}")
