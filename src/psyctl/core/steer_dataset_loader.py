"""Steering dataset loader for steering vector extraction."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-not-found]
from jinja2 import Environment, FileSystemLoader
from transformers import AutoTokenizer

from psyctl.core.logger import get_logger


class SteerDatasetLoader:
    """
    Load and process steering dataset for steering vector extraction.

    Dataset format:
    {
        "situation": "Conversation context...",
        "char_name": "Character name",
        "positive": "Full positive personality answer",
        "neutral": "Full neutral personality answer"
    }

    Attributes:
        logger: Logger instance for debugging
        jinja_env: Jinja2 environment for template loading
    """

    def __init__(self):
        """Initialize SteerDatasetLoader with logger and Jinja2 environment."""
        self.logger = get_logger("steer_dataset_loader")

        # Setup Jinja2 environment for template loading
        template_dir = Path(__file__).parent.parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.logger.debug(f"Template directory: {template_dir}")

    def load(self, dataset_path: Path | str) -> list[dict]:
        """
        Load steering dataset from JSONL file or HuggingFace dataset.

        Args:
            dataset_path: Path to dataset directory/JSONL file or HuggingFace dataset name

        Returns:
            List of dataset entries

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid

        Example:
            >>> loader = SteerDatasetLoader()
            >>> dataset = loader.load(Path("./dataset/steering"))
            >>> dataset = loader.load("CaveduckAI/steer-personality-rudeness-ko")
        """
        # Check if it's a HuggingFace dataset name
        if isinstance(dataset_path, str) and "/" in dataset_path:
            self.logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            try:
                hf_dataset = load_dataset(dataset_path, split="train")
                dataset = []
                for item in hf_dataset:  # type: ignore[attr-defined]
                    # Convert HF dataset format to steering dataset format
                    # Support both old (question) and new (situation) formats
                    entry = {
                        "situation": item.get("situation", item.get("question", "")),  # type: ignore[union-attr]
                        "char_name": item.get("char_name", "Assistant"),  # type: ignore[union-attr]
                        "positive": item.get("positive", ""),  # type: ignore[union-attr]
                        "neutral": item.get("neutral", ""),  # type: ignore[union-attr]
                    }
                    dataset.append(entry)
                self.logger.info(
                    f"Loaded {len(dataset)} entries from HuggingFace dataset"
                )
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset: {e}") from e

        # Convert string to Path for local files
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        # If path is directory, find JSONL file
        if dataset_path.is_dir():
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    f"No JSONL files found in directory: {dataset_path}"
                )
            if len(jsonl_files) > 1:
                self.logger.warning(
                    f"Multiple JSONL files found, using first: {jsonl_files[0].name}"
                )
            dataset_file = jsonl_files[0]
        else:
            dataset_file = dataset_path

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        self.logger.info(f"Loading dataset from: {dataset_file}")

        dataset = []
        with Path(dataset_file).open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())

                    # Validate required fields
                    required_fields = ["situation", "char_name", "positive", "neutral"]
                    missing_fields = [
                        field for field in required_fields if field not in entry
                    ]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")

                    dataset.append(entry)

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON at line {line_num}: {e}")
                    raise ValueError(f"Invalid JSON format at line {line_num}") from e
                except ValueError as e:
                    self.logger.error(f"Invalid entry at line {line_num}: {e}")
                    raise

        self.logger.info(f"Loaded {len(dataset)} entries from dataset")
        return dataset

    def create_prompts(
        self,
        dataset: list[dict],
        tokenizer: AutoTokenizer,
        format_type: str = "index",
        use_chat_template: bool = True,
        return_questions: bool = False,
    ) -> tuple[list[str], list[str]] | tuple[list[str], list[str], list[str]]:
        """
        Create positive and neutral prompt pairs from dataset.

        Args:
            dataset: List of dataset entries
            tokenizer: Tokenizer for applying chat template
            format_type: Prompt format type
                - "index": Multiple choice with indices (for CAA)
                - "direct": Direct answer (for BiPO full text)
            use_chat_template: Whether to use chat template
            return_questions: If True, also return question prompts (for BiPO to calculate answer token positions)

        Returns:
            If return_questions=False: Tuple of (positive_prompts, neutral_prompts)
            If return_questions=True: Tuple of (positive_prompts, neutral_prompts, question_prompts)

        Example:
            >>> loader = SteerDatasetLoader()
            >>> dataset = loader.load(Path("./dataset/steering"))
            >>> pos_prompts, neu_prompts = loader.create_prompts(dataset, tokenizer)
            >>> # For BiPO (needs question lengths)
            >>> pos_prompts, neu_prompts, questions = loader.create_prompts(dataset, tokenizer, format_type="direct", return_questions=True)
        """
        self.logger.info(
            f"Creating prompts from {len(dataset)} dataset entries (format: {format_type})"
        )

        positive_prompts = []
        neutral_prompts = []
        question_prompts = [] if return_questions else None

        for idx, entry in enumerate(dataset):
            situation = entry["situation"]
            char_name = entry["char_name"]
            positive_answer = entry["positive"]
            neutral_answer = entry["neutral"]

            # Create prompts based on format type
            if format_type == "index":
                # CAA format: show both answers with indices, append index
                # Alternate answer order to prevent order bias
                if idx % 2 == 0:
                    # Even: positive=(1, neutral=(2
                    positive_prompt = self._build_prompt_with_choices(
                        situation,
                        char_name,
                        positive_answer,
                        neutral_answer,
                        "(1",
                        tokenizer,
                        use_chat_template,
                    )
                    neutral_prompt = self._build_prompt_with_choices(
                        situation,
                        char_name,
                        positive_answer,
                        neutral_answer,
                        "(2",
                        tokenizer,
                        use_chat_template,
                    )
                else:
                    # Odd: positive=(2, neutral=(1 (swapped order)
                    positive_prompt = self._build_prompt_with_choices(
                        situation,
                        char_name,
                        neutral_answer,
                        positive_answer,
                        "(2",
                        tokenizer,
                        use_chat_template,
                    )
                    neutral_prompt = self._build_prompt_with_choices(
                        situation,
                        char_name,
                        neutral_answer,
                        positive_answer,
                        "(1",
                        tokenizer,
                        use_chat_template,
                    )
            elif format_type == "direct":
                # BiPO format: direct answer without choices
                if return_questions:
                    # Build question prompt (without answer)
                    question_prompt = self._build_question_only(
                        situation, char_name, tokenizer, use_chat_template
                    )
                    question_prompts.append(question_prompt)  # type: ignore[union-attr]

                positive_prompt = self._build_prompt_direct(
                    situation, char_name, positive_answer, tokenizer, use_chat_template
                )
                neutral_prompt = self._build_prompt_direct(
                    situation, char_name, neutral_answer, tokenizer, use_chat_template
                )
            else:
                raise ValueError(f"Unknown format_type: {format_type}")

            positive_prompts.append(positive_prompt)
            neutral_prompts.append(neutral_prompt)

        self.logger.info(
            f"Created {len(positive_prompts)} positive and {len(neutral_prompts)} neutral prompts"
        )

        if return_questions:
            return positive_prompts, neutral_prompts, question_prompts  # type: ignore[return-value]
        return positive_prompts, neutral_prompts

    def _build_prompt_with_choices(
        self,
        situation: str,
        char_name: str,
        answer_1: str,
        answer_2: str,
        selected: str,
        tokenizer: AutoTokenizer,
        use_chat_template: bool = True,
    ) -> str:
        """
        Build prompt with multiple choices for CAA extraction method.

        This method constructs prompts to extract activation from the answer token,
        not from special tokens like <end_of_turn>. It applies chat template to the
        question part only, then appends the answer selection outside the template.

        Args:
            situation: Situation description
            char_name: Character name
            answer_1: First answer option
            answer_2: Second answer option
            selected: Which answer is selected ("(1" or "(2")
            tokenizer: Tokenizer for chat template
            use_chat_template: Whether to use chat template
        Returns:
            Complete prompt with choices and selection

        Note:
            The answer is appended AFTER chat template application to ensure
            the last token is from answer content, matching PoC behavior.
        """
        # Load template
        template = self.jinja_env.get_template("md_question.j2")
        question = template.render(
            char_name=char_name,
            situation=situation.strip(),
            answer_1=answer_1.strip().replace("\n", ""),
            answer_2=answer_2.strip().replace("\n", ""),
        )

        # Apply chat template WITHOUT answer (add_generation_prompt=True)
        try:
            if use_chat_template:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Add <start_of_turn>model section
                )
                return prompt + selected
            else:
                return question + selected
        except Exception as e:
            self.logger.debug(f"Chat template failed, using raw text: {e}")
            return question + selected

    def _build_question_only(
        self,
        situation: str,
        char_name: str,
        tokenizer: AutoTokenizer,
        use_chat_template: bool = True,
    ) -> str:
        """
        Build question prompt without answer (for BiPO token length calculation).

        Args:
            situation: Situation description
            char_name: Character name
            tokenizer: Tokenizer for chat template
            use_chat_template: Whether to use chat template

        Returns:
            Question prompt without answer

        Note:
            This is used by BiPO to calculate the question length,
            so it can identify which tokens are from the answer.
        """
        # Build question without answer
        template = self.jinja_env.get_template("bipo_question.j2")
        question = template.render(
            char_name=char_name,
            situation=situation.strip(),
        )

        # Apply chat template WITHOUT answer (add_generation_prompt=True)
        try:
            if use_chat_template:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Add <start_of_turn>model section
                )
                return prompt
            else:
                return question
        except Exception as e:
            self.logger.debug(f"Chat template failed, using raw text: {e}")
            return question

    def _build_prompt_direct(
        self,
        situation: str,
        char_name: str,
        answer: str,
        tokenizer: AutoTokenizer,
        use_chat_template: bool = True,
    ) -> str:
        """
        Build BiPO-style prompt with direct answer (no choices shown).

        This method constructs prompts to extract activation from the answer token,
        not from special tokens. It applies chat template to the question part only,
        then appends the answer outside the template.

        Args:
            situation: Situation description
            char_name: Character name
            answer: The answer text
            tokenizer: Tokenizer for chat template
            use_chat_template: Whether to use chat template
        Returns:
            Complete prompt with direct answer

        Note:
            The answer is appended AFTER chat template application to ensure
            the last token is from answer content, matching PoC behavior.
        """
        # Build question without answer
        template = self.jinja_env.get_template("bipo_question.j2")
        question = template.render(
            char_name=char_name,
            situation=situation.strip(),
        )

        # Apply chat template WITHOUT answer (add_generation_prompt=True)
        try:
            if use_chat_template:
                messages = [{"role": "user", "content": question}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[call-arg]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Add <start_of_turn>model section
                )
                prompt = prompt + answer
            else:
                prompt = question
            return prompt + answer
        except Exception as e:
            self.logger.debug(f"Chat template failed, using raw text: {e}")
            return question + answer

    def get_batch_iterator(
        self, prompts: list[str], batch_size: int
    ) -> Iterator[list[str]]:
        """
        Create batch iterator for prompts.

        Args:
            prompts: List of prompt strings
            batch_size: Number of prompts per batch

        Yields:
            Batches of prompts

        Example:
            >>> loader = SteerDatasetLoader()
            >>> for batch in loader.get_batch_iterator(prompts, batch_size=16):
            ...     # Process batch
            ...     pass
        """
        for i in range(0, len(prompts), batch_size):
            yield prompts[i : i + batch_size]

    def get_dataset_info(self, dataset_path: Path) -> dict:
        """
        Get information about the dataset without loading all data.

        Args:
            dataset_path: Path to dataset directory or JSONL file

        Returns:
            Dictionary with dataset information

        Example:
            >>> loader = SteerDatasetLoader()
            >>> info = loader.get_dataset_info(Path("./dataset/steering"))
            >>> print(info['num_samples'], info['file_size_mb'])
        """
        # Find dataset file
        if dataset_path.is_dir():
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                raise FileNotFoundError(
                    f"No JSONL files found in directory: {dataset_path}"
                )
            dataset_file = jsonl_files[0]
        else:
            dataset_file = dataset_path

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        # Count lines for num_samples
        with Path(dataset_file).open("r", encoding="utf-8") as f:
            num_samples = sum(1 for _ in f)

        # Get file size
        file_size_bytes = dataset_file.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        info = {
            "file_path": str(dataset_file),
            "num_samples": num_samples,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_mb, 2),
        }

        self.logger.debug(f"Dataset info: {info}")
        return info

    def validate_dataset(self, dataset_path: Path) -> bool:
        """
        Validate dataset format and structure.

        Args:
            dataset_path: Path to dataset directory or JSONL file

        Returns:
            True if dataset is valid, False otherwise

        Example:
            >>> loader = SteerDatasetLoader()
            >>> is_valid = loader.validate_dataset(Path("./dataset/steering"))
        """
        try:
            dataset = self.load(dataset_path)

            if len(dataset) == 0:
                self.logger.error("Dataset is empty")
                return False

            # Check first entry structure
            first_entry = dataset[0]
            required_fields = ["situation", "char_name", "positive", "neutral"]
            for field in required_fields:
                if field not in first_entry:
                    self.logger.error(f"Missing required field: {field}")
                    return False

            self.logger.info("Dataset validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False
