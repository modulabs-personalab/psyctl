"""
Tests for activation extraction fix in SteerDatasetLoader.

These tests verify that prompts are generated correctly to extract
activations from answer content tokens, not special tokens.

Related:
    - ACTIVATION_EXTRACTION_FIX_PLAN.md
    - PoC: Target_Layer_Search.ipynb
"""

import pytest
import torch
from transformers import AutoTokenizer

from psyctl.core.steer_dataset_loader import SteerDatasetLoader


@pytest.fixture
def dataset():
    """Basic test dataset."""
    return [{
        'situation': 'You are at a party.',
        'char_name': 'Alice',
        'positive': 'I love parties!',
        'neutral': 'Parties are okay.'
    }]


@pytest.fixture
def tokenizer_gemma():
    """Gemma tokenizer with chat template."""
    return AutoTokenizer.from_pretrained("google/gemma-3-270m-it")


@pytest.fixture
def tokenizer_gpt2():
    """GPT-2 tokenizer without chat template."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TestTokenPosition:
    """Tests for token position correctness."""

    def test_last_token_is_answer_content(self, dataset, tokenizer_gemma):
        """
        TC-1.1: Verify last token is answer content, not special token.

        Purpose:
            Ensure activation extraction happens at answer token position.

        Related:
            - PoC: cell-10 collects activation from last token
            - Issue: Previously extracted from <end_of_turn> or newline
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        # Tokenize
        tokens = tokenizer_gemma(pos_prompts[0], return_tensors='pt', add_special_tokens=False)
        last_token_id = tokens['input_ids'][0][-1].item()
        last_token_text = tokenizer_gemma.decode([last_token_id])

        # Last token should NOT be special tokens
        assert last_token_text not in ['\n', '<end_of_turn>', '<pad>'], \
            f"Last token should be answer content, got: {repr(last_token_text)}"

        # Last token should be from answer selection (1 or 2)
        assert last_token_text.strip() in ['1', '2'], \
            f"Last token should be '1' or '2', got: {repr(last_token_text)}"

    def test_second_to_last_token(self, dataset, tokenizer_gemma):
        """
        TC-1.2: Verify second-to-last token is part of answer structure.

        Purpose:
            Ensure the token sequence leading to answer is correct.
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        tokens = tokenizer_gemma(pos_prompts[0], return_tensors='pt', add_special_tokens=False)
        second_last_token_id = tokens['input_ids'][0][-2].item()
        second_last_token_text = tokenizer_gemma.decode([second_last_token_id])

        # Should be '(' before '1' or '2'
        assert second_last_token_text in ['(', '\n'], \
            f"Second to last token unexpected: {repr(second_last_token_text)}"


class TestChatTemplateStructure:
    """Tests for chat template structure."""

    def test_contains_model_turn(self, dataset, tokenizer_gemma):
        """
        TC-2.1: Verify prompt contains <start_of_turn>model section.

        Purpose:
            Ensure add_generation_prompt=True is working correctly.

        Related:
            - PoC uses manual template with model turn
            - layer_analyzer.py uses add_generation_prompt=True
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        # Check for model turn
        assert "<start_of_turn>model" in pos_prompts[0], \
            "Prompt must contain '<start_of_turn>model' section"

        # Check it appears AFTER user turn
        user_pos = pos_prompts[0].find("<start_of_turn>user")
        model_pos = pos_prompts[0].find("<start_of_turn>model")

        assert user_pos < model_pos, \
            "Model turn should appear after user turn"

    def test_no_trailing_end_of_turn(self, dataset, tokenizer_gemma):
        """
        TC-2.2: Verify prompt does NOT end with <end_of_turn>.

        Purpose:
            Confirm answer is appended AFTER chat template, not inside it.

        Expected:
            Prompt ends with answer like "(1" or "(2", not special tokens.
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        # The prompt should NOT end with <end_of_turn>
        assert not pos_prompts[0].rstrip().endswith("<end_of_turn>"), \
            "Prompt should not end with <end_of_turn>"

        # Should end with answer content
        assert pos_prompts[0].rstrip().endswith("(1") or \
               pos_prompts[0].rstrip().endswith("(2"), \
            f"Prompt should end with answer index, got: {pos_prompts[0][-20:]}"

    def test_proper_sequence(self, dataset, tokenizer_gemma):
        """
        TC-2.3: Verify proper sequence: User -> Model -> Answer.

        Purpose:
            Ensure complete chat template structure is correct.
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        prompt = pos_prompts[0]

        # Find positions
        bos_pos = prompt.find("<bos>")
        user_start = prompt.find("<start_of_turn>user")
        user_end = prompt.find("<end_of_turn>", user_start)
        model_start = prompt.find("<start_of_turn>model")
        answer_pos = max(prompt.rfind("(1"), prompt.rfind("(2"))

        # Verify order
        assert bos_pos < user_start < user_end < model_start < answer_pos, \
            "Sequence must be: <bos> -> user -> <end_of_turn> -> model -> answer"


class TestPoCCompatibility:
    """Tests for PoC compatibility."""

    def test_poc_compatibility(self, dataset, tokenizer_gemma):
        """
        TC-3.1: Compare with PoC manual construction.

        Purpose:
            Verify we generate same token sequence as PoC.

        Related:
            - PoC: Target_Layer_Search.ipynb cell-3 build_prompt()
        """
        loader = SteerDatasetLoader()

        # Project's method
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")
        tokens_project = tokenizer_gemma(pos_prompts[0], return_tensors='pt', add_special_tokens=False)

        # PoC's manual method (simplified for testing)
        situation = "You are at a party."
        question_text = f"""[Situation]
{situation}
[Question]
You are Alice. What would your response be in this situation?
1. I love parties!
2. Parties are okay.
[Answer]
"""
        poc_prompt = tokenizer_gemma.apply_chat_template(
            [{"role": "user", "content": question_text}],
            tokenize=False,
            add_generation_prompt=True
        ) + "(1"

        tokens_poc = tokenizer_gemma(poc_prompt, return_tensors='pt', add_special_tokens=False)

        # Compare last token (most critical)
        project_last = tokens_project['input_ids'][0][-1].item()
        poc_last = tokens_poc['input_ids'][0][-1].item()

        assert project_last == poc_last, \
            f"Last token mismatch: project={project_last}, poc={poc_last}"

    def test_layer_analyzer_consistency(self, dataset, tokenizer_gemma):
        """
        TC-3.2: Verify consistency with layer_analyzer.py approach.

        Purpose:
            Ensure same method as layer_analyzer (which works correctly).

        Related:
            - layer_analyzer.py:271-275
        """
        loader = SteerDatasetLoader()

        # SteerDatasetLoader method
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")
        tokens_loader = tokenizer_gemma(pos_prompts[0], return_tensors='pt', add_special_tokens=False)

        # layer_analyzer.py method (simulation)
        situation = "You are at a party."
        prompt_analyzer = tokenizer_gemma.apply_chat_template(
            [{"role": "user", "content": situation}],
            tokenize=False,
            add_generation_prompt=True
        ) + "(1"
        tokens_analyzer = tokenizer_gemma(prompt_analyzer, return_tensors='pt', add_special_tokens=False)

        # Compare structure (should have model turn)
        assert "<start_of_turn>model" in pos_prompts[0]
        assert "<start_of_turn>model" in prompt_analyzer

        # Compare last tokens
        loader_last = tokenizer_gemma.decode([tokens_loader['input_ids'][0][-1].item()])
        analyzer_last = tokenizer_gemma.decode([tokens_analyzer['input_ids'][0][-1].item()])

        assert loader_last == analyzer_last, \
            f"Last token mismatch: loader={repr(loader_last)}, analyzer={repr(analyzer_last)}"


class TestMultiModel:
    """Tests across different models."""

    def test_gemma_model(self, dataset):
        """
        TC-4.1: Test with Gemma model.

        Purpose:
            Verify works with Gemma's chat template.
        """
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        loader = SteerDatasetLoader()

        pos_prompts, _ = loader.create_prompts(dataset, tokenizer, format_type="index")

        # Common checks
        assert "<start_of_turn>model" in pos_prompts[0]
        tokens = tokenizer(pos_prompts[0], return_tensors='pt', add_special_tokens=False)
        last_token = tokenizer.decode([tokens['input_ids'][0][-1].item()])
        assert last_token not in ['\n', '<end_of_turn>']
        assert last_token.strip() in ['1', '2']

    def test_no_chat_template_fallback(self, dataset, tokenizer_gpt2):
        """
        TC-4.3: Test fallback when chat template is not available.

        Purpose:
            Ensure graceful fallback for models without chat template.
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gpt2, format_type="index")

        # Should fallback to raw text
        # Still should end with answer
        assert pos_prompts[0].rstrip().endswith("(1") or \
               pos_prompts[0].rstrip().endswith("(2"), \
            "Fallback should still end with answer"

        # Check that prompt is not empty
        assert len(pos_prompts[0]) > 0
        assert "[Situation]" in pos_prompts[0]


class TestFormatTypes:
    """Tests for different format types."""

    def test_index_format(self, dataset, tokenizer_gemma):
        """
        TC-5.1: Test index format (CAA).

        Purpose:
            Verify index format generates correct structure with both choices.
        """
        loader = SteerDatasetLoader()
        pos_prompts, neu_prompts = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        # Should contain both choices
        assert "1." in pos_prompts[0]
        assert "2." in pos_prompts[0]

        # Should end with selection
        assert pos_prompts[0].rstrip().endswith("(1") or pos_prompts[0].rstrip().endswith("(2")

        # Should have model turn
        assert "<start_of_turn>model" in pos_prompts[0]

    def test_direct_format(self, dataset, tokenizer_gemma):
        """
        TC-5.2: Test direct format (BiPO).

        Purpose:
            Verify direct format generates correct structure with answer only.
        """
        loader = SteerDatasetLoader()
        pos_prompts, neu_prompts = loader.create_prompts(dataset, tokenizer_gemma, format_type="direct")

        # Should NOT contain choices
        assert "1." not in pos_prompts[0]
        assert "2." not in pos_prompts[0]

        # Should contain answer directly
        assert "love parties" in pos_prompts[0].lower() or "parties" in pos_prompts[0].lower()
        assert "okay" in neu_prompts[0].lower() or "parties" in neu_prompts[0].lower()

        # Should still have model turn
        assert "<start_of_turn>model" in pos_prompts[0]

        # Last token should be from answer
        tokens = tokenizer_gemma(pos_prompts[0], return_tensors='pt', add_special_tokens=False)
        last_token = tokenizer_gemma.decode([tokens['input_ids'][0][-1].item()])
        assert last_token.strip() != ''
        assert last_token not in ['\n', '<end_of_turn>']


class TestActivationExtraction:
    """Tests simulating actual activation extraction."""

    def test_attention_mask_position(self, dataset, tokenizer_gemma):
        """
        TC-6.1: Test attention mask based position calculation.

        Purpose:
            Simulate how hook_manager extracts last real token position.

        Related:
            - hook_manager.py:_get_last_real_token_position()
        """
        loader = SteerDatasetLoader()
        pos_prompts, _ = loader.create_prompts(dataset, tokenizer_gemma, format_type="index")

        # Tokenize with padding (simulate batch)
        prompts_batch = [pos_prompts[0], pos_prompts[0]]
        tokens = tokenizer_gemma(prompts_batch, return_tensors='pt', padding=True, add_special_tokens=False)

        # Find last real token using attention mask (like hook_manager does)
        for i in range(len(prompts_batch)):
            mask = tokens['attention_mask'][i]
            real_positions = torch.where(mask == 1)[0]
            last_pos = real_positions[-1].item()

            last_token_id = tokens['input_ids'][i][last_pos].item()
            last_token_text = tokenizer_gemma.decode([last_token_id])

            # Should be answer content, not special token
            assert last_token_text not in ['\n', '<end_of_turn>', '<pad>'], \
                f"Sample {i}: Last real token is special token: {repr(last_token_text)}"

            assert last_token_text.strip() in ['1', '2'], \
                f"Sample {i}: Last real token should be answer: {repr(last_token_text)}"


class TestRegression:
    """Regression tests to ensure existing functionality."""

    def test_position_bias_prevention(self, tokenizer_gemma):
        """
        TC-7.1: Verify position bias prevention still works.

        Purpose:
            Ensure alternating order (even/odd) is preserved after fix.
        """
        loader = SteerDatasetLoader()

        # Create 4 samples
        dataset_multi = [
            {'situation': f'Test {i}', 'char_name': 'A', 'positive': 'Pos', 'neutral': 'Neu'}
            for i in range(4)
        ]

        pos_prompts, neu_prompts = loader.create_prompts(dataset_multi, tokenizer_gemma, format_type="index")

        # Check alternating pattern
        for idx in range(4):
            if idx % 2 == 0:
                # Even: positive=(1, neutral=(2
                assert pos_prompts[idx].rstrip().endswith("(1"), f"Sample {idx} should end with (1"
                assert neu_prompts[idx].rstrip().endswith("(2"), f"Sample {idx} should end with (2"
            else:
                # Odd: positive=(2, neutral=(1
                assert pos_prompts[idx].rstrip().endswith("(2"), f"Sample {idx} should end with (2"
                assert neu_prompts[idx].rstrip().endswith("(1"), f"Sample {idx} should end with (1"


class TestEdgeCases:
    """Edge case tests."""

    def test_special_characters(self, tokenizer_gemma):
        """
        TC-8.3: Test special characters in answer.

        Purpose:
            Ensure special characters don't break prompt generation.
        """
        loader = SteerDatasetLoader()

        dataset_special = [{
            'situation': 'Test',
            'char_name': 'A',
            'positive': '(1 <test> [special]',
            'neutral': '(2 \n\t normal'
        }]

        pos_prompts, neu_prompts = loader.create_prompts(dataset_special, tokenizer_gemma, format_type="direct")

        # Should handle special chars
        assert len(pos_prompts) > 0
        assert '<test>' in pos_prompts[0] or 'test' in pos_prompts[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
