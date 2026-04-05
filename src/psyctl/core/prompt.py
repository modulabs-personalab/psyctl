from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from psyctl.core.logger import get_logger


class P2:
    """
    P2 is a prompt that is used to generate a personality-specific prompt.
    https://arxiv.org/abs/2206.07550
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.keywords = None
        self.personality = None
        self.keywords_build_prompt = None
        self.personality_build_prompt = None
        self.char_name = None
        self.logger = get_logger("p2")

    def build(self, char_name: str, personality_trait: str):
        self.logger.info(f"Building P2 for {char_name} with trait: {personality_trait}")

        keywords_prompt = (
            f"Words related to {personality_trait}? (format: Comma sperated words)"
        )
        self.logger.debug(f"Keywords prompt: {keywords_prompt}")
        keywords_build_prompt, keywords = self._get_result(keywords_prompt)
        self.logger.debug(f"Generated keywords: {keywords}")

        personality_prompt = (
            f"{keywords} are traits of {char_name}.\n\nDesribe about {char_name}"
        )
        prefill = f"Here's a description of {char_name}, built from the traits suggested by the list:"
        self.logger.debug(f"Personality prompt: {personality_prompt}")
        self.logger.debug(f"Prefill: {prefill}")
        personality_build_prompt, personality = self._get_result(
            personality_prompt, prefill=prefill
        )
        self.logger.debug(f"Generated personality: {personality}")

        self.char_name = char_name
        self.keywords = keywords
        self.personality = personality
        self.keywords_build_prompt = keywords_build_prompt
        self.personality_build_prompt = personality_build_prompt
        return self.personality

    def _get_result(self, prompt, prefill=None):
        messages = [{"role": "user", "content": prompt}]

        self.logger.debug(f"Input prompt: {prompt}")
        if prefill:
            self.logger.debug(f"Using prefill: {prefill}")

        # 1. 유저 메시지를 chat template로 변환 (<|assistant|> 까지 포함)
        try:
            # Try with return_dict=True to get dictionary format
            tokenized_input = self.tokenizer.apply_chat_template(  # type: ignore[call-arg]
                messages,
                tokenize=False,  # Get string first
                add_generation_prompt=True,
                return_tensors=None,
            )
            self.logger.debug(f"Chat template applied: {tokenized_input[:200]}...")

            # Now tokenize the string
            tokenized = self.tokenizer(  # type: ignore[call-arg]
                tokenized_input,
                return_tensors="pt",
                add_special_tokens=False,  # Chat template already adds special tokens
            )

        except Exception as e:
            # Fallback: some models don't support chat templates
            self.logger.debug(f"Chat template failed, using fallback: {e}")
            tokenized = self.tokenizer(  # type: ignore[call-arg]
                prompt, return_tensors="pt", add_special_tokens=True
            )

        # 2. prefill이 있다면 assistant 답변의 시작 부분으로 추가
        if prefill:
            prefill_ids = self.tokenizer.encode(  # type: ignore[attr-defined]
                prefill, add_special_tokens=False, return_tensors="pt"
            )  # End encode

            # tokenized["input_ids"]와 prefill_ids 이어붙이기
            tokenized["input_ids"] = torch.cat(
                [tokenized["input_ids"], prefill_ids], dim=1
            )

            # attention_mask도 확장
            prefill_attention = torch.ones_like(prefill_ids)
            tokenized["attention_mask"] = torch.cat(
                [tokenized["attention_mask"], prefill_attention], dim=1
            )
            self.logger.debug(
                f"Added prefill tokens, new length: {tokenized['input_ids'].shape[1]}"
            )

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device  # type: ignore[attr-defined]
        self.logger.debug(f"Using device: {device}")
        tokenized["input_ids"] = tokenized["input_ids"].to(device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

        # 3. 모델 생성
        self.logger.debug("Starting model generation...")
        outputs = self.model.generate(  # type: ignore[attr-defined]
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.01,
            pad_token_id=self.tokenizer.pad_token_id,  # type: ignore[attr-defined]
        )
        len_input = tokenized["input_ids"][0].shape[0]
        input_text = self.tokenizer.decode(  # type: ignore[attr-defined]
            tokenized["input_ids"][0], skip_special_tokens=True
        )
        output_text = self.tokenizer.decode(  # type: ignore[attr-defined]
            outputs[0, len_input:], skip_special_tokens=True
        )
        self.logger.debug(f"Generated output: {output_text}")
        return input_text, output_text
