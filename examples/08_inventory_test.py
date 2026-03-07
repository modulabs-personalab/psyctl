"""
PSYCTL Example: Local Model Inventory Test (Logprob-based)

Runs a psychological inventory (IPIP-NEO-120) against a local model to measure
Big Five personality traits using logprob-based scoring. Optionally compares
baseline vs steered personality profiles.

This is the local-model counterpart of 09_openrouter_inventory_test.py.
Instead of chat API responses, it uses token logprobs for more precise scoring.

Usage::

    # Baseline only (no steering vector)
    python 08_inventory_test.py --model google/gemma-3-270m-it

    # With steering vector
    python 08_inventory_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector ./steering_vector/extroversion.safetensors \
        --strength 2.0

    # Test specific trait with a specific inventory
    python 08_inventory_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector ./steering_vector/extroversion.safetensors \
        --inventory ipip_neo_120 \
        --trait Extraversion

    # Sweep multiple strengths
    python 08_inventory_test.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector ./steering_vector/extroversion.safetensors \
        --strengths 0.5,1.0,2.0,3.0

Requirements:
    - HF_TOKEN environment variable (or .env file)
    - GPU recommended for larger models
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from psyctl.core.benchmark.inventory_tester import InventoryTester

load_dotenv()
console = Console()


def print_scores_table(
    scores: dict, title: str = "Inventory Results"
) -> None:
    """Print domain scores as a Rich table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Domain", style="bold")
    table.add_column("Raw Score", justify="right")
    table.add_column("Population Norm", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("Percentile", justify="right", style="green")

    for domain_code in sorted(scores.keys()):
        s = scores[domain_code]
        table.add_row(
            s["domain_name"],
            f"{s['raw_score']:.2f}",
            f"{s['population_mean']:.1f} +/- {s['population_std']:.1f}",
            f"{s['z_score']:+.2f}",
            f"{s['percentile']:.1f}%",
        )

    console.print(table)


def print_comparison_table(
    baseline: dict, steered: dict, strength: float
) -> None:
    """Print baseline vs steered comparison."""
    table = Table(
        title=f"Baseline vs Steered (strength={strength})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Domain", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Steered", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Z-Score Delta", justify="right", style="magenta")

    for domain_code in sorted(baseline.keys()):
        if domain_code not in steered:
            continue
        b = baseline[domain_code]
        s = steered[domain_code]
        change = s["raw_score"] - b["raw_score"]
        z_delta = s["z_score"] - b["z_score"]
        change_style = "green" if change > 0 else "red"
        table.add_row(
            b["domain_name"],
            f"{b['raw_score']:.2f}",
            f"{s['raw_score']:.2f}",
            f"[{change_style}]{change:+.2f}[/{change_style}]",
            f"{z_delta:+.2f}",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSYCTL Local Inventory Test (logprob-based)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-270m-it",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--vector",
        type=str,
        default=None,
        help="Path to steering vector (.safetensors)",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="ipip_neo_120",
        help="Inventory name (e.g. ipip_neo_120, rei_40, vgq_14)",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Specific trait to test (e.g. Extraversion, N, E, O, A, C)",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Single steering strength (default: 1.0)",
    )
    parser.add_argument(
        "--strengths",
        type=str,
        default=None,
        help="Comma-separated strengths for sweep (e.g. 0.5,1.0,2.0,3.0)",
    )
    parser.add_argument(
        "--layer-spec",
        type=str,
        default=None,
        help="Layer specification (e.g. '15', '0-5', 'middle')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    # Parse strengths
    if args.strengths:
        strengths = [float(s) for s in args.strengths.split(",")]
    elif args.vector:
        strengths = [args.strength]
    else:
        strengths = []

    vector_path = Path(args.vector) if args.vector else None

    console.rule("[bold]PSYCTL Inventory Test[/bold]")
    console.print(f"  Model:     {args.model}")
    console.print(f"  Inventory: {args.inventory}")
    console.print(f"  Trait:     {args.trait or 'All'}")
    if vector_path:
        console.print(f"  Vector:    {vector_path}")
        console.print(f"  Strengths: {strengths}")
    else:
        console.print("  Mode:      Baseline only (no steering vector)")
    console.print()

    tester = InventoryTester()
    all_results = []

    # --- Baseline ---
    console.rule("[bold]Baseline (no steering)[/bold]")
    baseline_result = tester.test_inventory(
        model=args.model,
        steering_vector_path=None,
        inventory_name=args.inventory,
        device=args.device,
        target_trait=args.trait,
    )
    baseline_scores = baseline_result["baseline_scores"]
    print_scores_table(baseline_scores, title="Baseline Scores")
    all_results.append({"strength": 0.0, "scores": baseline_scores})

    # --- Steered (one or more strengths) ---
    if vector_path:
        for strength in strengths:
            console.rule(f"[bold]Steered (strength={strength})[/bold]")
            steered_result = tester.test_inventory(
                model=args.model,
                steering_vector_path=vector_path,
                inventory_name=args.inventory,
                steering_strength=strength,
                device=args.device,
                target_trait=args.trait,
                layer_spec=args.layer_spec,
            )
            steered_scores = steered_result.get("steered_scores", {})
            if steered_scores:
                print_scores_table(
                    steered_scores, title=f"Steered Scores (strength={strength})"
                )
                print_comparison_table(baseline_scores, steered_scores, strength)
                all_results.append(
                    {"strength": strength, "scores": steered_scores}
                )

    # --- Save results ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "model": args.model,
            "inventory": args.inventory,
            "trait": args.trait,
            "vector": str(vector_path) if vector_path else None,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        console.print(f"\nResults saved to: {output_path}")

    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    main()
