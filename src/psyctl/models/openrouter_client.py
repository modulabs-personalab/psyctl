"""
OpenRouter API Client for PSYCTL

This module provides an interface to OpenRouter API for generating personality-based
responses using cloud-based LLMs. It supports structured output generation and batch
processing for efficient steering dataset creation.

Key Features:
- Structured JSON output using OpenRouter's response_format
- Batch processing for multiple prompts
- Cost tracking and reporting
- Error handling and retry logic
- Rate limiting support

Example Usage:
    client = OpenRouterClient(api_key="sk-or-xxxx")
    response = client.generate(
        prompt="Your prompt here",
        model="qwen/qwen3-next-80b-a3b-instruct"
    )
"""

from __future__ import annotations

import html
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from psyctl.core.logger import get_logger


class OpenRouterAPIError(Exception):
    """Raised when the OpenRouter API returns an error response."""


class OpenRouterTimeoutError(OpenRouterAPIError):
    """Raised when an OpenRouter API request times out."""


class OpenRouterClient:
    """
    Client for OpenRouter API with support for structured output generation.

    This class handles communication with OpenRouter API for generating text
    responses from various LLMs. It provides cost tracking, error handling,
    and batch processing capabilities.

    Attributes:
        api_key (str): OpenRouter API key
        base_url (str): OpenRouter API base URL
        logger: Logger instance for debugging
        total_requests (int): Total number of API requests made
        total_cost (float): Cumulative cost of all requests
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        Initialize OpenRouter client.

        Args:
            api_key (str): OpenRouter API key (format: sk-or-xxxx)
            base_url (str): OpenRouter API base URL
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self.api_key = api_key
        self.base_url = base_url
        self.logger = get_logger("openrouter_client")
        self.total_requests = 0
        self.total_cost = 0.0
        self._lock = threading.Lock()

        self.logger.info("OpenRouter client initialized")

    def generate(
        self,
        prompt: str,
        model: str = "qwen/qwen3-next-80b-a3b-instruct",
        temperature: float = 0,
        max_tokens: int = 100,
        system_prompt: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> tuple[str, str]:
        """
        Generate text response from OpenRouter API.

        Args:
            prompt (str): User prompt for generation
            model (str): Model identifier on OpenRouter
            temperature (float): Sampling temperature (0.0-2.0, default: 0)
            max_tokens (int): Maximum tokens to generate
            system_prompt (str | None): System prompt for context
            top_k (int | None): Top-k sampling parameter
            top_p (float | None): Top-p (nucleus) sampling parameter

        Returns:
            Tuple[str, str]: (generation_id, generated_text)

        Raises:
            Exception: If API request fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Build request body
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_k is not None:
            request_body["top_k"] = top_k
        if top_p is not None:
            request_body["top_p"] = top_p

        self.logger.debug(f"OpenRouter request - Model: {model}")
        self.logger.debug(
            f"OpenRouter request - Messages: {json.dumps(messages, ensure_ascii=False)[:500]}..."
        )
        self.logger.debug(
            f"OpenRouter request - Temperature: {temperature}, Max tokens: {max_tokens}"
        )
        if top_k is not None:
            self.logger.debug(f"OpenRouter request - Top K: {top_k}")
        if top_p is not None:
            self.logger.debug(f"OpenRouter request - Top P: {top_p}")

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=60,
            )

            self.logger.debug(f"OpenRouter response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                raise OpenRouterAPIError(error_msg)

            result = response.json()
            self.logger.debug(
                f"OpenRouter response JSON: {json.dumps(result, ensure_ascii=False)[:1000]}..."
            )

            generation_id = result["id"]
            generated_text = result["choices"][0]["message"]["content"]

            self.logger.debug(f"Generated text (raw): {generated_text}")

            # Log original response for debugging
            if "&#" in generated_text:
                self.logger.debug(
                    f"HTML entities detected in response: {generated_text[:100]}..."
                )

            # Decode HTML entities if present
            generated_text = html.unescape(generated_text)

            # Log after unescape
            if "&#" in generated_text:
                self.logger.warning(
                    f"HTML entities still present after unescape: {generated_text[:100]}..."
                )

            with self._lock:
                self.total_requests += 1
            self.logger.debug(
                f"Generated response (ID: {generation_id}): {generated_text}"
            )

            return generation_id, generated_text

        except requests.exceptions.Timeout:
            self.logger.error("Request timeout")
            raise OpenRouterTimeoutError("OpenRouter API request timeout") from None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise OpenRouterAPIError(f"OpenRouter API request failed: {e}") from e
        except KeyError as e:
            self.logger.error(f"Unexpected response format: {e}")
            raise OpenRouterAPIError(
                f"Unexpected OpenRouter API response: {e}"
            ) from e

    def generate_batch(
        self,
        prompts: list[str],
        model: str = "qwen/qwen3-next-80b-a3b-instruct",
        temperature: float = 0,
        max_tokens: int = 100,
        system_prompt: str | None = None,
        max_workers: int = 1,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[tuple[str, str]]:
        """
        Generate responses for multiple prompts with optional parallel processing.

        Args:
            prompts (List[str]): List of user prompts
            model (str): Model identifier on OpenRouter
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens per generation
            system_prompt (str | None): System prompt for all requests
            max_workers (int): Number of parallel workers (1 = sequential)
            top_k (int | None): Top-k sampling parameter
            top_p (float | None): Top-p (nucleus) sampling parameter

        Returns:
            List[Tuple[str, str]]: List of (generation_id, generated_text) tuples
        """
        if max_workers <= 1:
            # Sequential processing
            return self._generate_batch_sequential(
                prompts, model, temperature, max_tokens, system_prompt, top_k, top_p
            )
        else:
            # Parallel processing
            return self._generate_batch_parallel(
                prompts,
                model,
                temperature,
                max_tokens,
                system_prompt,
                max_workers,
                top_k,
                top_p,
            )

    def _generate_batch_sequential(
        self,
        prompts: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None,
        top_k: int | None,
        top_p: float | None,
    ) -> list[tuple[str, str]]:
        """Sequential batch generation."""
        results = []

        for i, prompt in enumerate(prompts):
            try:
                gen_id, text = self.generate(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    top_k=top_k,
                    top_p=top_p,
                )
                results.append((gen_id, text))

                # Small delay to avoid rate limiting
                if i < len(prompts) - 1:
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Failed to generate for prompt {i}: {e}")
                results.append(("", ""))

        return results

    def _generate_batch_parallel(
        self,
        prompts: list[str],
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str | None,
        max_workers: int,
        top_k: int | None,
        top_p: float | None,
    ) -> list[tuple[str, str]]:
        """Parallel batch generation using ThreadPoolExecutor."""
        results: list[tuple[str, str] | None] = [None] * len(
            prompts
        )  # Pre-allocate results list

        def generate_single(index: int, prompt: str):
            try:
                gen_id, text = self.generate(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    top_k=top_k,
                    top_p=top_p,
                )
                return index, (gen_id, text)
            except Exception as e:
                self.logger.error(f"Failed to generate for prompt {index}: {e}")
                return index, ("", "")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_single, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

        return results  # type: ignore[return-value]

    def get_generation_cost(self, generation_id: str) -> float | None:
        """
        Get the cost of a specific generation.

        Args:
            generation_id (str): Generation ID from API response

        Returns:
            float | None: Cost in USD, or None if retrieval fails
        """
        try:
            time.sleep(1.0)  # Rate limiting

            response = requests.get(
                f"{self.base_url}/generation",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"id": generation_id},
                timeout=10,
            )

            if response.status_code == 200:
                cost = response.json()["data"]["total_cost"]
                with self._lock:
                    self.total_cost += cost
                return cost
            else:
                self.logger.warning(f"Failed to get cost for {generation_id}")
                return None

        except Exception as e:
            self.logger.warning(f"Cost retrieval failed: {e}")
            return None

    def get_total_cost(self) -> float:
        """
        Get cumulative cost of all API calls.

        Returns:
            float: Total cost in USD
        """
        with self._lock:
            return self.total_cost

    def get_total_requests(self) -> int:
        """
        Get total number of API requests made.

        Returns:
            int: Number of requests
        """
        with self._lock:
            return self.total_requests


# Example usage
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv(override=True)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY environment variable not set")
        exit(1)

    client = OpenRouterClient(api_key=api_key)

    # Single generation test
    gen_id, response = client.generate(
        prompt="What is the capital of France?",
        model="qwen/qwen3-next-80b-a3b-instruct",
        max_tokens=50,
    )
    print(f"Response: {response}")
    print(f"Generation ID: {gen_id}")

    # Batch generation test
    prompts = [
        "What is 2+2?",
        "Name a color.",
        "What day comes after Monday?",
    ]
    results = client.generate_batch(prompts)
    for i, (_gen_id, text) in enumerate(results):
        print(f"Prompt {i}: {text}")

    print(f"\nTotal requests: {client.get_total_requests()}")
    print(f"Total cost: ${client.get_total_cost():.6f}")
