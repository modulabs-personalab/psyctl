"""Tests for SteerDatasetLoader."""

import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from psyctl.core.logger import get_logger, setup_logging
from psyctl.core.steer_dataset_loader import SteerDatasetLoader

setup_logging()
logger = get_logger("test_steer_dataset_loader")


@pytest.fixture
def tokenizer():
    """Load a test tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dataset_file(tmp_path):
    """Create a dataset file."""
    dataset_path = tmp_path / "test_dataset.jsonl"

    samples = [
        {
            "situation": "Alice meets Bob at a party.\nBob: Hi, how are you?",
            "char_name": "Alice",
            "positive": "I'm so excited to be here! Want to dance?",
            "neutral": "I'm fine, thanks. Just looking around.",
        },
        {
            "situation": "Charlie is at work.\nManager: Can you help with this project?",
            "char_name": "Charlie",
            "positive": "Absolutely! I'd love to lead the team!",
            "neutral": "Sure, I can assist if needed.",
        },
    ]

    with Path(dataset_path).open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return dataset_path


def test_load_dataset(dataset_file):
    """Test loading dataset."""
    logger.info("Testing SteerDatasetLoader.load()")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    assert len(dataset) == 2

    # Check first sample
    sample1 = dataset[0]
    assert sample1["situation"] == "Alice meets Bob at a party.\nBob: Hi, how are you?"
    assert sample1["char_name"] == "Alice"
    assert "excited" in sample1["positive"]
    assert "fine" in sample1["neutral"]

    logger.success("Dataset loading test passed")


def test_create_prompts_index_format(dataset_file, tokenizer):
    """Test creating prompts with index format (CAA style)."""
    logger.info("Testing create_prompts() with index format")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    pos_prompts, neu_prompts = loader.create_prompts(
        dataset, tokenizer, format_type="index", use_chat_template=True
    )

    assert len(pos_prompts) == 2
    assert len(neu_prompts) == 2

    # Check structure - should include both answers with indices
    assert "[Situation]" in pos_prompts[0] or "Alice meets Bob" in pos_prompts[0]
    assert "1." in pos_prompts[0]  # Should have choice 1
    assert "2." in pos_prompts[0]  # Should have choice 2
    assert "(1" in pos_prompts[0]  # Should select index 1
    assert "<bos>" in pos_prompts[0]  # Should have special token <bos>
    assert "<start_of_turn>model" in pos_prompts[0]  # Should have special token <start_of_turn>model

    logger.success("Index format prompt creation test passed")

def test_create_prompts_index_format_without_chat_template(dataset_file, tokenizer):
    """Test creating prompts with index format (CAA style) without chat template."""
    logger.info("Testing create_prompts() with index format without chat template")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    pos_prompts, neu_prompts = loader.create_prompts(
        dataset, tokenizer, format_type="index",
        use_chat_template=False
    )

    assert len(pos_prompts) == 2
    assert len(neu_prompts) == 2

    # Check structure - should include both answers with indices
    assert "[Situation]" in pos_prompts[0] or "Alice meets Bob" in pos_prompts[0]
    assert "1." in pos_prompts[0]  # Should have choice 1
    assert "2." in pos_prompts[0]  # Should have choice 2
    assert "(1" in pos_prompts[0]  # Should select index 1
    assert "<bos>" not in pos_prompts[0]  # Should have special token <bos>
    assert "<start_of_turn>model" not in pos_prompts[0]  # Should have special token <start_of_turn>model
    
    logger.success("Index format prompt creation test passed")

def test_create_prompts_direct_format(dataset_file, tokenizer):
    """Test creating prompts with direct format (BiPO style)."""
    logger.info("Testing create_prompts() with direct format")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    pos_prompts, neu_prompts = loader.create_prompts(
        dataset, tokenizer, format_type="direct", 
        use_chat_template=True
    )

    assert len(pos_prompts) == 2
    assert len(neu_prompts) == 2

    # Check structure - should NOT include choices
    assert "1." not in pos_prompts[0]  # No choice numbering
    assert "2." not in pos_prompts[0]
    assert "excited" in pos_prompts[0]  # Should have positive answer
    assert "fine" in neu_prompts[0]  # Should have neutral answer

    # Positive and neutral should be different
    assert pos_prompts[0] != neu_prompts[0]

    logger.success("Direct format prompt creation test passed")

def test_create_prompts_direct_format_without_chat_template(dataset_file, tokenizer):
    """Test creating prompts with direct format (BiPO style) without chat template."""
    logger.info("Testing create_prompts() with direct format without chat template")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    pos_prompts, neu_prompts = loader.create_prompts(
        dataset, tokenizer, format_type="direct", 
        use_chat_template=False
    )

    assert len(pos_prompts) == 2
    assert len(neu_prompts) == 2

    assert "<bos>" not in pos_prompts[0]  # Should have special token <bos>
    assert "<start_of_turn>model" not in pos_prompts[0]  # Should have special token <start_of_turn>model
    
    logger.success("Direct format prompt creation test without chat template passed")


def test_prompt_format_comparison(dataset_file, tokenizer):
    """Test that index and direct formats produce different prompts."""
    logger.info("Testing format comparison")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    pos_index, _ = loader.create_prompts(dataset, tokenizer, format_type="index")
    pos_direct, _ = loader.create_prompts(dataset, tokenizer, format_type="direct")

    # Formats should be different
    assert pos_index[0] != pos_direct[0]

    # Index format should be longer (includes both choices)
    assert len(pos_index[0]) > len(pos_direct[0])

    logger.success("Format comparison test passed")


def test_dataset_structure_validation(dataset_file):
    """Test that dataset has correct structure."""
    logger.info("Testing dataset structure validation")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    for entry in dataset:
        # Required fields
        assert "situation" in entry
        assert "char_name" in entry
        assert "positive" in entry
        assert "neutral" in entry

        # Should NOT have old fields
        assert "question" not in entry

        # Validate types
        assert isinstance(entry["situation"], str)
        assert isinstance(entry["char_name"], str)
        assert isinstance(entry["positive"], str)
        assert isinstance(entry["neutral"], str)

    logger.success("Structure validation test passed")


def test_build_prompt_with_choices(dataset_file, tokenizer):
    """Test _build_prompt_with_choices method."""
    logger.info("Testing _build_prompt_with_choices()")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    sample = dataset[0]
    prompt = loader._build_prompt_with_choices(
        situation=sample["situation"],
        char_name=sample["char_name"],
        answer_1=sample["positive"],
        answer_2=sample["neutral"],
        selected="(1",
        tokenizer=tokenizer,
    )

    # Should include all components
    assert "Alice" in prompt or "party" in prompt  # Situation/char_name
    assert "1." in prompt  # First choice
    assert "2." in prompt  # Second choice
    assert "(1" in prompt  # Selected index

    logger.success("_build_prompt_with_choices test passed")


def test_build_prompt_direct(dataset_file, tokenizer):
    """Test _build_prompt_direct method."""
    logger.info("Testing _build_prompt_direct()")

    loader = SteerDatasetLoader()
    dataset = loader.load(dataset_file)

    sample = dataset[0]
    prompt = loader._build_prompt_direct(
        situation=sample["situation"],
        char_name=sample["char_name"],
        answer=sample["positive"],
        tokenizer=tokenizer,
    )

    # Should include components
    assert "Alice" in prompt or "party" in prompt
    assert "excited" in prompt  # The answer

    # Should NOT include choice formatting
    assert "1." not in prompt
    assert "2." not in prompt

    logger.success("_build_prompt_direct test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
