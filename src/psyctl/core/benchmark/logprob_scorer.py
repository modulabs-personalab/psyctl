"""Shared log probability scoring utilities for evaluation."""

from __future__ import annotations

import torch
from typing import Any

from psyctl.core.logger import get_logger

logger = get_logger("logprob_scorer")


class LogProbScorer:
    """
    Shared log probability scoring for personality evaluations.
    
    Used by both inventory-based tests and LLM judge evaluations
    to calculate weighted scores from model logits.
    """

    @staticmethod
    def calculate_weighted_score(
        logits: torch.Tensor,
        tokenizer: Any,
        score_range: tuple[int, int] = (1, 5),
    ) -> tuple[float, float]:
        """
        Calculate weighted score from logits using softmax.

        This method extracts logits for numeric tokens in the specified range,
        applies softmax to get probabilities, and computes a weighted average.
        Only numeric tokens within the score range are considered.

        Args:
            logits: Model output logits (vocab_size,)
            tokenizer: Tokenizer to map scores to token IDs
            score_range: Tuple of (min_score, max_score) inclusive

        Returns:
            Tuple of (weighted_score, confidence) where:
            - weighted_score: Weighted average of scores using softmax probabilities
            - confidence: Probability of the most likely score token

        Example:
            >>> logits = model(**inputs).logits[0, -1, :]
            >>> score, conf = LogProbScorer.calculate_weighted_score(
            ...     logits, tokenizer, score_range=(1, 5)
            ... )
            >>> print(f"Score: {score:.2f}, Confidence: {conf:.2%}")
        """
        min_score, max_score = score_range

        # Find token IDs for all scores in range by searching vocab directly
        score_tokens = {}
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab
        
        for score in range(min_score, max_score + 1):
            score_str = str(score)
            # Search for the numeric token in vocab
            # Try different representations: "1", "▁1", " 1", etc.
            found = False
            for token, token_id in vocab.items():
                # Check if token matches the score (with or without whitespace/special chars)
                if token == score_str or token.strip() == score_str or token.strip("▁ ") == score_str:
                    score_tokens[score] = token_id
                    found = True
                    break
            
            # Fallback: try encoding and use the last token (numeric part)
            if not found:
                token_ids = tokenizer.encode(score_str, add_special_tokens=False)
                if len(token_ids) >= 1:
                    # Use the last token which is typically the actual number
                    score_tokens[score] = token_ids[-1]

        # Check if we found all expected tokens
        expected_count = max_score - min_score + 1
        if len(score_tokens) < expected_count:
            missing = set(range(min_score, max_score + 1)) - set(score_tokens.keys())
            logger.warning(
                f"Could not find all score tokens. Missing: {missing}. "
                f"Found: {list(score_tokens.keys())}"
            )
            # Return neutral score if tokens are missing
            neutral = (min_score + max_score) / 2
            return neutral, 0.0

        # Extract logits for score tokens only
        score_logits = {score: logits[token_id].item() for score, token_id in score_tokens.items()}

        # Convert to tensors for softmax calculation
        scores_list = sorted(score_logits.keys())
        logits_list = [score_logits[s] for s in scores_list]
        logits_tensor = torch.tensor(logits_list)

        # Apply softmax to get probabilities
        probs = torch.softmax(logits_tensor, dim=0)

        # Calculate weighted average
        scores_tensor = torch.tensor([float(s) for s in scores_list])
        weighted_score = (probs * scores_tensor).sum().item()

        # Confidence is the maximum probability
        confidence = probs.max().item()

        logger.debug(
            f"Score calculation: weighted={weighted_score:.3f}, "
            f"confidence={confidence:.3f}, "
            f"probs={dict(zip(scores_list, probs.tolist()))}"
        )

        return weighted_score, confidence

    @staticmethod
    def calculate_weighted_score_from_logprobs(
        top_logprobs: dict[str, float], score_range: tuple[int, int] = (0, 5)
    ) -> tuple[float, float]:
        """
        Calculate weighted score from API-returned log probabilities.

        Used when evaluating with API models that return top_logprobs directly
        (e.g., OpenAI API, vLLM API).

        Args:
            top_logprobs: Dictionary mapping tokens to their log probabilities
            score_range: Tuple of (min_score, max_score) inclusive

        Returns:
            Tuple of (weighted_score, confidence)

        Example:
            >>> # From OpenAI API response
            >>> top_logprobs = {"0": -2.5, "1": -1.2, "2": -0.5, ...}
            >>> score, conf = LogProbScorer.calculate_weighted_score_from_logprobs(
            ...     top_logprobs, score_range=(0, 5)
            ... )
        """
        min_score, max_score = score_range

        if not top_logprobs:
            neutral = (min_score + max_score) / 2
            return neutral, 0.0

        # Filter numeric tokens within range
        numeric_tokens = {}
        for token, log_prob in top_logprobs.items():
            token_clean = token.strip()
            if token_clean.isdigit():
                score_val = int(token_clean)
                if min_score <= score_val <= max_score:
                    numeric_tokens[score_val] = log_prob

        if not numeric_tokens:
            logger.warning(
                f"No numeric tokens found in range {score_range}. "
                f"Available: {list(top_logprobs.keys())}"
            )
            neutral = (min_score + max_score) / 2
            return neutral, 0.0

        # Apply softmax to log probabilities
        scores_list = sorted(numeric_tokens.keys())
        log_probs_list = [numeric_tokens[s] for s in scores_list]

        # Softmax calculation (numerically stable)
        max_log_prob = max(log_probs_list)
        exp_probs = [torch.exp(torch.tensor(lp - max_log_prob)).item() for lp in log_probs_list]
        sum_exp = sum(exp_probs)
        softmax_probs = [ep / sum_exp for ep in exp_probs]

        # Weighted average
        weighted_score = sum(score * prob for score, prob in zip(scores_list, softmax_probs))

        # Confidence is max probability
        confidence = max(softmax_probs)

        logger.debug(
            f"Score from logprobs: weighted={weighted_score:.3f}, "
            f"confidence={confidence:.3f}"
        )

        return weighted_score, confidence

