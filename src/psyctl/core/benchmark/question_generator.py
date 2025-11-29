"""Question generation for personality testing using LLMs."""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from psyctl.core.logger import get_logger
from psyctl.models.llm_loader import LLMLoader

logger = get_logger("question_generator")


class QuestionGenerator:
    """
    Generate personality test questions using LLM.
    
    Used in LLM-as-Judge benchmarks when users don't provide
    a predefined question list.
    """

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        default_questions: dict[str, list[str]] | None = None,
        loader: LLMLoader | None = None,
    ):
        """
        Initialize question generator.
        
        Args:
            prompts: Prompt templates (question_generation)
            default_questions: Default questions by trait
            loader: Optional LLMLoader instance (creates new one if not provided)
        """
        self.prompts = prompts or {}
        self.default_questions = default_questions or {}
        self.loader = loader or LLMLoader()

    def generate_questions(
        self,
        trait: str,
        num_questions: int = 8,
        generator_model: Any | None = None,
        generator_tokenizer: Any | None = None,
        device: str = "auto",
    ) -> list[str]:
        """
        Generate personality test questions for a given trait.
        
        Args:
            trait: Personality trait to test (e.g., "extraversion")
            num_questions: Number of questions to generate
            generator_model: Optional pre-loaded model (loads if not provided)
            generator_tokenizer: Optional pre-loaded tokenizer
            device: Device to use for generation
            
        Returns:
            List of generated questions
            
        Example:
            >>> generator = QuestionGenerator()
            >>> questions = generator.generate_questions("extraversion", num_questions=5)
            >>> print(questions)
            ["/no_think 파티에서 ...", "/no_think 새로운 사람을 ...", ...]
        """
        # Load model if not provided
        if generator_model is None or generator_tokenizer is None:
            logger.info("Loading generator model for question generation...")
            # Use default model from config or loader
            generator_model, generator_tokenizer = self.loader.load_model(
                "auto", device=device
            )

        # Generate prompt
        prompt = self._create_question_generation_prompt(trait, num_questions)

        logger.info(f"Generating {num_questions} questions for trait: {trait}")
        logger.debug(f"Prompt: {prompt[:200]}...")

        # Generate response
        logger.info("Generating questions (this may take a moment)...")
        inputs = generator_tokenizer(prompt, return_tensors="pt").to(generator_model.device)

        with torch.no_grad():
            outputs = generator_model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=generator_tokenizer.eos_token_id,
            )

        logger.info("Question generation complete, parsing response...")
        response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log full response for debugging (first 1000 chars)
        logger.info(f"Full generation response (first 1000 chars):\n{response[:1000]}...")

        # Extract questions from response
        questions = self._extract_questions_from_response(response, num_questions)
        
        # Log extracted questions
        logger.info(f"Extracted {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            logger.info(f"  {i}. {q}")

        if len(questions) < num_questions:
            logger.warning(
                f"Generated only {len(questions)}/{num_questions} questions. "
                f"Using fallback questions for the rest."
            )
            # Add fallback questions if needed
            questions.extend(
                self._get_fallback_questions(trait, num_questions - len(questions))
            )

        logger.info(f"Successfully generated {len(questions)} questions")

        return questions[:num_questions]

    def _create_question_generation_prompt(
        self, trait: str, num_questions: int
    ) -> str:
        """
        Create prompt for question generation.
        
        Uses template from config if available, falls back to default.
        """
        if "question_generation" in self.prompts:
            return self.prompts["question_generation"].format(
                trait=trait, num_questions=num_questions
            )
        
        # Fallback default (English)
        return f"""Generate {num_questions} diverse questions to test {trait} personality traits.

Each question should:
1. Assess {trait} in everyday situations
2. Be related to {trait} characteristics
3. Be written in English
4. Include "/no_think " prefix

Generate {num_questions} questions as a JSON array:
[
  "/no_think Question 1",
  "/no_think Question 2",
  ...
]

JSON array:"""

    def _extract_questions_from_response(
        self, response: str, expected_count: int
    ) -> list[str]:
        """
        Extract questions from LLM response.
        
        Tries multiple parsing strategies:
        1. JSON array parsing
        2. Line-by-line extraction with /no_think prefix
        3. Numbered list extraction
        """
        questions = []

        # Strategy 1: Try JSON parsing
        try:
            # Remove markdown code blocks if present
            cleaned_response = response
            if '```' in cleaned_response:
                # Extract content between ```json and ``` or just between ```
                code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', cleaned_response)
                if code_block_match:
                    cleaned_response = code_block_match.group(1)
            
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*?\]', cleaned_response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    # Handle both string arrays and object arrays
                    for item in parsed:
                        if isinstance(item, str):
                            questions.append(item.strip())
                        elif isinstance(item, dict):
                            # Extract question from object (e.g., {"question": "...", "type": "..."})
                            if "question" in item:
                                questions.append(str(item["question"]).strip())
                            elif "text" in item:
                                questions.append(str(item["text"]).strip())
                    logger.info(f"Extracted {len(questions)} questions via JSON parsing")
                    if questions:
                        return questions
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {e}, trying alternative methods")

        # Strategy 2: Extract lines with /no_think prefix
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('/no_think '):
                # Filter out very short lines (likely incomplete or just prefix)
                if len(line) > 20:
                    questions.append(line)
                    if len(questions) >= expected_count:
                        break

        if questions:
            logger.debug(f"Extracted {len(questions)} questions via /no_think prefix")
            return questions

        # Strategy 3: Extract numbered questions
        for line in lines:
            line = line.strip()
            # Match patterns like "1. 질문", "1) 질문", "- 질문"
            match = re.match(r'^[\d\-\*\.]+[\)\.\s]+(.+)$', line)
            if match:
                question_text = match.group(1).strip()
                # Add /no_think prefix if not present
                if not question_text.startswith('/no_think '):
                    question_text = '/no_think ' + question_text
                questions.append(question_text)
                if len(questions) >= expected_count:
                    break

        logger.debug(f"Extracted {len(questions)} questions via numbered list")
        return questions

    def _get_fallback_questions(self, trait: str, count: int) -> list[str]:
        """
        Get fallback questions when generation fails.
        
        Uses default questions from config if available.
        """
        # Normalize trait name (lowercase, first word only)
        trait_key = trait.lower().split()[0]

        # Try to get from config first
        if trait_key in self.default_questions:
            questions = self.default_questions[trait_key][:count]
            logger.warning(
                f"Using {len(questions)} default questions from config for {trait}"
            )
            return questions

        # Fallback to hardcoded if not in config
        fallback_by_trait = {
            "extraversion": [
                "/no_think 새로운 사람을 만났을 때 먼저 대화를 시작하나요?",
                "/no_think 사교 모임에서 적극적으로 참여하나요?",
                "/no_think 혼자 있는 것보다 친구들과 있는 것을 선호하나요?",
                "/no_think 파티나 행사에 참석하는 것을 즐기나요?",
                "/no_think 발표나 공개 연설을 편하게 하나요?",
            ],
            "neuroticism": [
                "/no_think 작은 일에도 쉽게 스트레스를 받나요?",
                "/no_think 불안감을 자주 느끼나요?",
                "/no_think 걱정거리가 많은 편인가요?",
                "/no_think 감정 기복이 큰 편인가요?",
                "/no_think 예상치 못한 일이 생기면 당황하나요?",
            ],
            "openness": [
                "/no_think 새로운 아이디어나 경험에 열려 있나요?",
                "/no_think 예술이나 문화 활동을 즐기나요?",
                "/no_think 추상적이고 철학적인 주제에 관심이 있나요?",
                "/no_think 전통적인 방식보다 혁신적인 방식을 선호하나요?",
                "/no_think 상상력이 풍부한 편인가요?",
            ],
            "agreeableness": [
                "/no_think 다른 사람의 감정을 잘 배려하나요?",
                "/no_think 갈등 상황을 피하려고 노력하나요?",
                "/no_think 타인을 쉽게 신뢰하나요?",
                "/no_think 도움을 요청받으면 기꺼이 돕나요?",
                "/no_think 협력하는 것을 중요하게 생각하나요?",
            ],
            "conscientiousness": [
                "/no_think 계획을 세우고 그대로 실행하나요?",
                "/no_think 세부사항에 주의를 기울이나요?",
                "/no_think 책임감이 강한 편인가요?",
                "/no_think 정리정돈을 잘 하나요?",
                "/no_think 목표를 달성하기 위해 끈기 있게 노력하나요?",
            ],
        }

        if trait_key in fallback_by_trait:
            questions = fallback_by_trait[trait_key][:count]
        else:
            # Generic fallback
            questions = [
                f"/no_think {trait} 특성과 관련된 질문 {i+1}"
                for i in range(count)
            ]

        logger.warning(f"Using {len(questions)} fallback questions for {trait}")
        return questions


