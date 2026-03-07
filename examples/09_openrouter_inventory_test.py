"""OpenRouter-based inventory test for multiple API models.

Runs REI-40 inventory test against OpenRouter models by requesting
direct 1-5 Likert scale responses via chat completions API.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psyctl.data.inventories import create_inventory
from psyctl.models.openrouter_client import OpenRouterClient

load_dotenv(override=True)

# Models to test
MODELS = [
    ("openai/o3", "OpenAI o3"),
    ("anthropic/claude-opus-4.5", "Claude Opus 4.5"),
    ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("x-ai/grok-3", "Grok 3"),
    ("z-ai/glm-4.7", "GLM 4.7"),
]

SYSTEM_PROMPT = """You are taking a personality assessment. For each statement, respond with ONLY a single number from 1 to 5 indicating how accurately the statement describes you.

Scale:
1 = Definitely not true of myself
2 = Somewhat not true of myself
3 = Neither true nor untrue of myself
4 = Somewhat true of myself
5 = Definitely true of myself

Respond with ONLY the number (1, 2, 3, 4, or 5). No explanation, no other text."""


def parse_score(response: str) -> float:
    """Parse a 1-5 score from model response."""
    response = response.strip()
    # Try to find a number 1-5 in the response
    match = re.search(r"\b([1-5])\b", response)
    if match:
        return float(match.group(1))
    # Fallback: neutral
    return 3.0


def run_inventory_test(
    client: OpenRouterClient,
    model_id: str,
    model_name: str,
    inventory_name: str = "rei_40",
) -> dict:
    """Run inventory test for a single model."""
    inventory = create_inventory(inventory_name)
    questions = inventory.get_questions()

    print(f"\n  Testing {model_name} ({model_id})...")
    print(f"  Questions: {len(questions)}")

    domain_responses: dict[str, list[float]] = {}
    errors = 0

    for i, question in enumerate(questions):
        prompt = f'Statement: "{question["text"]}"\n\nYour rating (1-5):'

        try:
            _, response_text = client.generate(
                prompt=prompt,
                model=model_id,
                temperature=0,
                max_tokens=16,
                system_prompt=SYSTEM_PROMPT,
            )

            score = parse_score(response_text)

            # Reverse scoring for minus-keyed items
            if question["keyed"] == "minus":
                score = 6.0 - score

            domain = question["domain"]
            if domain not in domain_responses:
                domain_responses[domain] = []
            domain_responses[domain].append(score)

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{len(questions)}")

        except Exception as e:
            errors += 1
            print(f"    Error on Q{i + 1}: {e}")
            # Use neutral score on error
            domain = question["domain"]
            if domain not in domain_responses:
                domain_responses[domain] = []
            domain_responses[domain].append(3.0)

        # Rate limiting
        time.sleep(0.3)

    # Calculate scores
    scores = inventory.calculate_scores(domain_responses)

    print(f"  Done! (errors: {errors})")
    return {
        "model_id": model_id,
        "model_name": model_name,
        "inventory": inventory_name,
        "scores": scores,
        "errors": errors,
        "total_questions": len(questions),
    }


def print_results_table(all_results: list[dict]):
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print("REI-40 INVENTORY TEST RESULTS")
    print("=" * 90)

    # Header
    domains = ["RA", "RE", "EA", "EE", "R", "E"]
    header = f"{'Model':<20}"
    for d in domains:
        header += f"  {d:>6}"
    print(header)
    print("-" * 90)

    # Each model's scores (raw scores)
    print("\n[Raw Scores]")
    print(f"{'Model':<20}", end="")
    for d in domains:
        print(f"  {d:>6}", end="")
    print()
    print("-" * 70)

    for result in all_results:
        row = f"{result['model_name']:<20}"
        for d in domains:
            if d in result["scores"]:
                raw = result["scores"][d]["raw_score"]
                row += f"  {raw:>6.1f}"
            else:
                row += f"  {'N/A':>6}"
        print(row)

    # Z-scores
    print("\n[Z-Scores]")
    print(f"{'Model':<20}", end="")
    for d in domains:
        print(f"  {d:>6}", end="")
    print()
    print("-" * 70)

    for result in all_results:
        row = f"{result['model_name']:<20}"
        for d in domains:
            if d in result["scores"]:
                z = result["scores"][d]["z_score"]
                row += f"  {z:>+6.2f}"
            else:
                row += f"  {'N/A':>6}"
        print(row)

    # Percentiles
    print("\n[Percentiles]")
    print(f"{'Model':<20}", end="")
    for d in domains:
        print(f"  {d:>6}", end="")
    print()
    print("-" * 70)

    for result in all_results:
        row = f"{result['model_name']:<20}"
        for d in domains:
            if d in result["scores"]:
                pct = result["scores"][d]["percentile"]
                row += f"  {pct:>5.1f}%"
            else:
                row += f"  {'N/A':>6}"
        print(row)

    print("\n" + "=" * 90)
    print("Domains: RA=Rational Ability, RE=Rational Engagement,")
    print("         EA=Experiential Ability, EE=Experiential Engagement,")
    print("         R=Rationality (RA+RE), E=Experientiality (EA+EE)")
    print("=" * 90)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    client = OpenRouterClient(api_key=api_key)

    print("=" * 60)
    print("OpenRouter REI-40 Inventory Test")
    print(f"Models: {len(MODELS)}")
    print(f"Questions per model: 40")
    print("=" * 60)

    all_results = []

    for model_id, model_name in MODELS:
        try:
            result = run_inventory_test(client, model_id, model_name)
            all_results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {model_name} - {e}")
            continue

    if all_results:
        print_results_table(all_results)

        # Save raw results to JSON
        output_path = Path("results/rei_40_openrouter_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nRaw results saved to: {output_path}")

    print(f"\nTotal API requests: {client.get_total_requests()}")


if __name__ == "__main__":
    main()
