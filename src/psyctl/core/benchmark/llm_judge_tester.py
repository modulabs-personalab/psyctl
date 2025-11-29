"""LLM Judge-based personality evaluation."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from psyctl.core.benchmark.judge_evaluator import JudgeEvaluator
from psyctl.core.benchmark.layer_resolver import LayerResolver
from psyctl.core.logger import get_logger
from psyctl.core.benchmark.question_generator import QuestionGenerator
from psyctl.core.steering_applier import SteeringApplier
from psyctl.models.llm_loader import LLMLoader
from psyctl.models.vector_store import VectorStore

logger = get_logger("llm_judge_tester")


class LLMJudgeTester:
    """
    Test personality steering using LLM as Judge evaluation.
    
    Complements inventory-based testing with flexible, LLM-generated
    or user-provided questions evaluated by a judge model.
    """

    def __init__(
        self,
        prompts: dict[str, str] | None = None,
        default_questions: dict[str, list[str]] | None = None,
        layer_groups_config: dict[str, Any] | None = None,
    ):
        """
        Initialize LLM Judge Tester with shared components.
        
        Args:
            prompts: Prompt templates from config
            default_questions: Default questions by trait from config
            layer_groups_config: Layer group configurations from config
        """
        self.loader = LLMLoader()
        self.applier = SteeringApplier()
        self.vector_store = VectorStore()
        self.prompts = prompts or {}
        self.default_questions = default_questions or {}
        self.layer_groups_config = layer_groups_config or {}
        self.generator = QuestionGenerator(
            prompts=prompts,
            default_questions=default_questions,
            loader=self.loader,
        )

    def test_with_judge(
        self,
        model: str,
        trait: str,
        questions: list[str] | None = None,
        num_questions: int = 8,
        steering_vector_path: Path | None = None,
        judge_config: dict[str, Any] | None = None,
        steering_strengths: list[float] | None = None,
        layer_spec: str | list[str] | None = None,
        device: str = "auto",
        output_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """
        Test personality steering with LLM Judge evaluation.
        
        Args:
            model: Target model path or name
            trait: Personality trait to test (e.g., "extraversion")
            questions: Optional list of questions (generates if None)
            num_questions: Number of questions to generate if questions is None
            steering_vector_path: Path to steering vector file
            judge_config: Judge model configuration
            steering_strengths: List of steering strengths to test
            layer_spec: Layer specification (numbers, ranges, or keywords like "early", "middle", "late")
            device: Device to use for model
            output_dir: Directory to save results
            
        Returns:
            List of result dictionaries (one per strength)
        """
        logger.info("=" * 80)
        logger.info("LLM JUDGE PERSONALITY TEST")
        logger.info("=" * 80)
        logger.info(f"Target Model: {model}")
        logger.info(f"Personality Trait: {trait}")
        logger.info(f"Steering Strengths: {steering_strengths}")
        logger.info(
            f"Layer Spec: {LayerResolver.describe_layer_spec(layer_spec, self.layer_groups_config) if layer_spec else 'all'}"
        )
        logger.info("=" * 80)

        # Default values
        if steering_strengths is None:
            steering_strengths = [1.0]
        if judge_config is None:
            judge_config = {"type": "local", "model_path": "auto"}

        # 1. Load target model
        logger.info("Loading target model...")
        model_obj, tokenizer = self.loader.load_model(model, device=device)
        logger.info("Model loaded successfully")

        # 2. Get or generate questions
        if questions is None:
            logger.info(f"Generating {num_questions} questions for {trait}...")
            questions = self.generator.generate_questions(
                trait=trait,
                num_questions=num_questions,
                generator_model=model_obj,
                generator_tokenizer=tokenizer,
                device=device,
            )
            logger.info(f"Generated {len(questions)} questions")
        else:
            logger.info(f"Using {len(questions)} provided questions")

        # 3. Initialize judge with prompts
        logger.info("Initializing judge evaluator...")
        
        # If judge model is "auto", reuse the target model
        judge_cfg_copy = judge_config.copy()
        if judge_cfg_copy.get("type") == "local" and judge_cfg_copy.get("model_path") == "auto":
            logger.info("Using target model as judge (model_path='auto')")
            judge_cfg_copy["_reuse_model"] = model_obj
            judge_cfg_copy["_reuse_tokenizer"] = tokenizer
        
        judge = JudgeEvaluator(
            judge_cfg_copy, prompts=self.prompts, loader=self.loader
        )
        logger.info("Judge initialized")

        # 4. Evaluate baseline (no steering)
        logger.info("\nEvaluating baseline responses...")
        baseline_responses = self._generate_responses(
            model_obj, tokenizer, questions
        )
        baseline_scores = self._evaluate_with_judge(
            judge, questions, baseline_responses, trait
        )
        logger.info(
            f"Baseline - Personality: {baseline_scores['personality_avg']:.2f}, "
            f"Relevance: {baseline_scores['relevance_avg']:.2f}"
        )

        # 5. Load steering vector if provided
        all_results = []
        
        if steering_vector_path:
            logger.info(f"\nLoading steering vector: {steering_vector_path}")
            steering_vectors, metadata = self.vector_store.load_multi_layer(
                steering_vector_path
            )
            logger.info(f"Loaded {len(steering_vectors)} layer vectors")

            # Resolve layer specification
            available_layers = list(steering_vectors.keys())
            resolved_layers = LayerResolver.resolve_layer_spec(
                layer_spec, available_layers, self.layer_groups_config
            )
            
            if resolved_layers:
                logger.info(
                    f"Resolved to {len(resolved_layers)} layers: "
                    f"{[LayerResolver._extract_layer_number(l) for l in resolved_layers]}"
                )
            else:
                logger.info("Using all available layers")

            # Test each strength
            for strength in steering_strengths:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Testing strength: {strength}")
                logger.info(f"{'=' * 80}")

                # Apply steering with resolved layers
                steered_model, _ = self.applier.get_steering_applied_model(
                    steering_vector_path=steering_vector_path,
                    model=model_obj,
                    tokenizer=tokenizer,
                    strength=strength,
                    prompt_length=0,
                    orthogonal=False,
                    layers=resolved_layers,
                )

                # Generate and evaluate steered responses
                logger.info("Generating steered responses...")
                steered_responses = self._generate_responses(
                    steered_model, tokenizer, questions
                )

                logger.info("Evaluating steered responses...")
                steered_scores = self._evaluate_with_judge(
                    judge, questions, steered_responses, trait
                )
                logger.info(
                    f"Steered - Personality: {steered_scores['personality_avg']:.2f}, "
                    f"Relevance: {steered_scores['relevance_avg']:.2f}"
                )

                # Remove steering hooks
                if hasattr(steered_model, "remove_steering"):
                    steered_model.remove_steering()

                # Compile results
                result = {
                    "model": model,
                    "trait": trait,
                    "steering_vector": str(steering_vector_path) if steering_vector_path else None,
                    "steering_strength": strength,
                    "layer_spec": layer_spec,
                    "resolved_layers": [LayerResolver._extract_layer_number(l) for l in resolved_layers] if resolved_layers else "all",
                    "num_questions": len(questions),
                    "judge_config": judge_config,
                    "timestamp": datetime.now().isoformat(),
                    "baseline": {
                        "personality_score": baseline_scores["personality_avg"],
                        "relevance_score": baseline_scores["relevance_avg"],
                        "personality_confidence": baseline_scores["personality_conf"],
                        "relevance_confidence": baseline_scores["relevance_conf"],
                        "per_question": baseline_scores["per_question"],
                    },
                    "steered": {
                        "personality_score": steered_scores["personality_avg"],
                        "relevance_score": steered_scores["relevance_avg"],
                        "personality_confidence": steered_scores["personality_conf"],
                        "relevance_confidence": steered_scores["relevance_conf"],
                        "per_question": steered_scores["per_question"],
                    },
                    "comparison": {
                        "personality_change": steered_scores["personality_avg"] - baseline_scores["personality_avg"],
                        "relevance_change": steered_scores["relevance_avg"] - baseline_scores["relevance_avg"],
                        "personality_percent_change": (
                            (steered_scores["personality_avg"] - baseline_scores["personality_avg"])
                            / baseline_scores["personality_avg"]
                            * 100
                            if baseline_scores["personality_avg"] > 0
                            else 0.0
                        ),
                    },
                }

                all_results.append(result)

                # Save individual result
                if output_dir:
                    self._save_result(result, output_dir, strength, resolved_layers)

        else:
            # No steering - just baseline
            result = {
                "model": model,
                "trait": trait,
                "steering_vector": None,
                "steering_strength": 0.0,
                "layer_spec": None,
                "resolved_layers": None,
                "num_questions": len(questions),
                "judge_config": judge_config,
                "timestamp": datetime.now().isoformat(),
                "baseline": {
                    "personality_score": baseline_scores["personality_avg"],
                    "relevance_score": baseline_scores["relevance_avg"],
                    "personality_confidence": baseline_scores["personality_conf"],
                    "relevance_confidence": baseline_scores["relevance_conf"],
                    "per_question": baseline_scores["per_question"],
                },
                "steered": None,
                "comparison": None,
            }
            all_results.append(result)

            if output_dir:
                self._save_result(result, output_dir, strength=0.0, layers=None)

        # Cleanup
        judge.cleanup()
        logger.info("\n" + "=" * 80)
        logger.info("LLM JUDGE TEST COMPLETE")
        logger.info("=" * 80)

        return all_results

    def _generate_responses(
        self, model: Any, tokenizer: Any, questions: list[str]
    ) -> list[str]:
        """
        Generate responses to questions.
        
        Args:
            model: Model to use
            tokenizer: Tokenizer
            questions: List of questions
            
        Returns:
            List of generated responses
        """
        responses = []

        for i, question in enumerate(questions, 1):
            # Remove /no_think prefix if present for cleaner prompt
            clean_question = question.replace("/no_think ", "").strip()
            logger.info(f"Generating response {i}/{len(questions)} for: {clean_question[:60]}...")
            
            # Simple, direct prompt - model should not repeat the question
            prompt = f"{clean_question}\n\nAnswer in 2-3 sentences:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Get prompt length for proper response extraction
            prompt_length = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Extract only the generated part (skip the prompt)
            generated_ids = outputs[0][prompt_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Simple cleanup: remove question if it appears at the start (common issue)
            if response.lower().startswith(clean_question.lower()):
                response = response[len(clean_question):].strip()
            
            # Limit response length (take first 2-3 sentences)
            sentences = response.split('. ')
            if len(sentences) > 3:
                response = '. '.join(sentences[:3]) + '.'
            if len(response) > 250:
                response = response[:250].rsplit('.', 1)[0] + '.'
            
            # Log the response
            logger.info(f"Generated response {i}: {response[:150]}...")

            responses.append(response)

        return responses

    def _evaluate_with_judge(
        self,
        judge: JudgeEvaluator,
        questions: list[str],
        responses: list[str],
        trait: str,
    ) -> dict[str, Any]:
        """
        Evaluate responses using judge model.
        
        Args:
            judge: Judge evaluator
            questions: List of questions
            responses: List of responses
            trait: Personality trait
            
        Returns:
            Dictionary with evaluation scores
        """
        personality_scores = []
        relevance_scores = []
        personality_confs = []
        relevance_confs = []
        per_question = []

        for i, (question, response) in enumerate(zip(questions, responses), 1):
            logger.info(f"Evaluating response {i}/{len(questions)}...")

            # Evaluate personality
            p_score, p_conf = judge.evaluate_personality(response, trait)
            personality_scores.append(p_score)
            personality_confs.append(p_conf)

            # Evaluate relevance
            r_score, r_conf = judge.evaluate_relevance(question, response)
            relevance_scores.append(r_score)
            relevance_confs.append(r_conf)
            
            logger.debug(f"Response {i} - Personality: {p_score:.2f}, Relevance: {r_score:.2f}")

            per_question.append({
                "question": question,
                "response": response,
                "personality_score": p_score,
                "personality_confidence": p_conf,
                "relevance_score": r_score,
                "relevance_confidence": r_conf,
            })

        return {
            "personality_avg": sum(personality_scores) / len(personality_scores),
            "relevance_avg": sum(relevance_scores) / len(relevance_scores),
            "personality_conf": sum(personality_confs) / len(personality_confs),
            "relevance_conf": sum(relevance_confs) / len(relevance_confs),
            "per_question": per_question,
        }

    def _save_result(
        self,
        result: dict[str, Any],
        output_dir: Path,
        strength: float,
        resolved_layers: list[str] | None,
    ):
        """Save result to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        trait = result["trait"]
        if resolved_layers:
            layer_numbers = [LayerResolver._extract_layer_number(l) for l in resolved_layers]
            layer_str = f"layers_{'_'.join(map(str, layer_numbers[:5]))}"  # First 5 layers
            if len(layer_numbers) > 5:
                layer_str += f"_etc{len(layer_numbers)}"
        else:
            layer_str = "all_layers"
        
        filename = f"judge_{trait}_strength_{strength}_{layer_str}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result to: {output_path}")

