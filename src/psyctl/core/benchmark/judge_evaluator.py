"""LLM Judge evaluator for personality trait assessment."""

from __future__ import annotations

import json
from typing import Any

import requests
import torch

from psyctl.core.logger import get_logger
from psyctl.core.benchmark.logprob_scorer import LogProbScorer
from psyctl.models.llm_loader import LLMLoader

logger = get_logger("judge_evaluator")


class JudgeEvaluator:
    """
    Evaluate personality traits and relevance using LLM as Judge.
    
    Supports both API-based models (OpenAI-compatible) and local models.
    """

    def __init__(
        self,
        judge_config: dict[str, Any],
        prompts: dict[str, str] | None = None,
        loader: LLMLoader | None = None,
    ):
        """
        Initialize judge evaluator.
        
        Args:
            judge_config: Judge model configuration with keys:
                - "type": "api" or "local"
                - For API: "api_base", "api_key" (optional), "model_name"
                - For local: "model_path", "device" (optional)
            prompts: Prompt templates for evaluation (personality_evaluation, relevance_evaluation)
            loader: Optional LLMLoader instance (required for local models)
        """
        self.config = judge_config
        self.judge_type = judge_config.get("type", "local")
        self.prompts = prompts or {}
        self.loader = loader or LLMLoader()

        # Initialize based on type
        if self.judge_type == "api":
            self.api_base = judge_config["api_base"]
            self.api_key = judge_config.get("api_key")
            self.model_name = judge_config["model_name"]
            self.session = requests.Session()
            if self.api_key:
                self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            logger.info(f"Initialized API judge: {self.model_name}")
        else:
            self.model_path = judge_config.get("model_path", "auto")
            self.device = judge_config.get("device", "auto")
            
            # Check if we should reuse an existing model
            if "_reuse_model" in judge_config and "_reuse_tokenizer" in judge_config:
                self.judge_model = judge_config["_reuse_model"]
                self.judge_tokenizer = judge_config["_reuse_tokenizer"]
                logger.info("Reusing target model as judge")
            else:
                self.judge_model = None
                self.judge_tokenizer = None
                logger.info(f"Initialized local judge: {self.model_path}")

    def _ensure_local_model_loaded(self):
        """Lazy-load local judge model if not already loaded."""
        if self.judge_type == "local" and self.judge_model is None:
            logger.info("Loading local judge model...")
            self.judge_model, self.judge_tokenizer = self.loader.load_model(
                self.model_path, device=self.device
            )

    def evaluate_personality(
        self, response: str, trait: str
    ) -> tuple[float, float]:
        """
        Evaluate how well the response reflects a personality trait.
        
        Args:
            response: Response text to evaluate
            trait: Personality trait to evaluate (e.g., "extraversion")
            
        Returns:
            Tuple of (score, confidence) where:
            - score: 0-5 personality trait score
            - confidence: Probability of the most likely score
        """
        prompt = self._create_personality_prompt(response, trait)

        if self.judge_type == "api":
            return self._evaluate_with_api(prompt, score_range=(0, 5))
        else:
            return self._evaluate_with_local(prompt, score_range=(0, 5))

    def evaluate_relevance(
        self, question: str, response: str
    ) -> tuple[float, float]:
        """
        Evaluate how relevant the response is to the question.
        
        Args:
            question: Original question
            response: Response text to evaluate
            
        Returns:
            Tuple of (score, confidence) where:
            - score: 0-5 relevance score
            - confidence: Probability of the most likely score
        """
        prompt = self._create_relevance_prompt(question, response)

        if self.judge_type == "api":
            return self._evaluate_with_api(prompt, score_range=(0, 5))
        else:
            return self._evaluate_with_local(prompt, score_range=(0, 5))

    def _evaluate_with_api(
        self, prompt: str, score_range: tuple[int, int] = (0, 5)
    ) -> tuple[float, float]:
        """
        Evaluate using API-based judge model.
        
        Args:
            prompt: Evaluation prompt
            score_range: Score range (min, max)
            
        Returns:
            Tuple of (score, confidence)
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "echo": False,
        }

        try:
            response = self.session.post(
                self.api_base,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Extract top logprobs
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    top_logprobs = choice["logprobs"]["top_logprobs"][0]

                    # Calculate score using shared scorer
                    score, confidence = LogProbScorer.calculate_weighted_score_from_logprobs(
                        top_logprobs, score_range=score_range
                    )

                    logger.debug(
                        f"API evaluation: score={score:.2f}, confidence={confidence:.2f}"
                    )
                    return score, confidence

            logger.warning("No logprobs found in API response")
            return (score_range[0] + score_range[1]) / 2, 0.0

        except Exception as e:
            logger.error(f"API evaluation failed: {e}")
            return (score_range[0] + score_range[1]) / 2, 0.0

    def _evaluate_with_local(
        self, prompt: str, score_range: tuple[int, int] = (0, 5)
    ) -> tuple[float, float]:
        """
        Evaluate using local judge model.
        
        Args:
            prompt: Evaluation prompt
            score_range: Score range (min, max)
            
        Returns:
            Tuple of (score, confidence)
        """
        self._ensure_local_model_loaded()

        # Tokenize and get logits
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(
            self.judge_model.device
        )

        with torch.no_grad():
            outputs = self.judge_model(**inputs)

        # Get logits for next token
        logits = outputs.logits[0, -1, :]

        # Calculate score using shared scorer
        score, confidence = LogProbScorer.calculate_weighted_score(
            logits, self.judge_tokenizer, score_range=score_range
        )

        logger.debug(
            f"Local evaluation: score={score:.2f}, confidence={confidence:.2f}"
        )

        return score, confidence

    def _create_personality_prompt(self, response: str, trait: str) -> str:
        """
        Create prompt for personality trait evaluation.
        
        Uses template from config if available, falls back to default.
        """
        if "personality_evaluation" in self.prompts:
            return self.prompts["personality_evaluation"].format(
                trait=trait, response=response
            )
        
        # Fallback default
        return f"""Rate how well this response reflects {trait} personality traits (0-5 scale):
- 0: No {trait} (opposite traits, very low {trait})
- 1: Very low {trait} (minimal {trait} indicators)
- 2: Low {trait} (some {trait} elements but limited)
- 3: Moderate {trait} (balanced {trait} characteristics)
- 4: High {trait} (clearly shows {trait} traits)
- 5: Extremely high {trait} (very strong {trait} expression)

Response: "{response}"

Score: """

    def _create_relevance_prompt(self, question: str, response: str) -> str:
        """
        Create prompt for relevance evaluation.
        
        Uses template from config if available, falls back to default.
        """
        if "relevance_evaluation" in self.prompts:
            return self.prompts["relevance_evaluation"].format(
                question=question, response=response
            )
        
        # Fallback default
        return f"""Rate how relevant this response is to the question (0-5 scale):
- 0: Completely irrelevant (does not address the question at all)
- 1: Very irrelevant (barely touches on the question)
- 2: Somewhat irrelevant (partially addresses but misses the point)
- 3: Moderately relevant (addresses the question but could be better)
- 4: Highly relevant (clearly and directly addresses the question)
- 5: Extremely relevant (perfectly addresses the question with depth)

Question: "{question}"
Response: "{response}"

Score: """

    def cleanup(self):
        """Clean up resources."""
        if self.judge_type == "api":
            self.session.close()
        elif self.judge_model is not None:
            # Only clean up if we loaded the model ourselves (not reused)
            if "_reuse_model" not in self.config:
                # Free up GPU memory
                del self.judge_model
                del self.judge_tokenizer
                torch.cuda.empty_cache()
                logger.info("Cleaned up local judge model")
            else:
                logger.info("Not cleaning up reused model")


