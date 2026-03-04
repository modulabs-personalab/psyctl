"""
PSYCTL Benchmark: Pre-trained BiPO Steering Vectors from HuggingFace

Downloads BiPO vectors from dalekwon/bipo-steering-vectors and benchmarks
them using psyctl's two evaluation methods:

  1. IPIP-NEO-120 Inventory (logprob-based Big Five scoring)
  2. LLM-as-Judge (model generates + self-evaluates personality alignment)

Usage::

    python 08_benchmark_bipo_vectors.py                     # Full benchmark
    python 08_benchmark_bipo_vectors.py --english-only      # English only
    python 08_benchmark_bipo_vectors.py --judge-only        # Judge only
    python 08_benchmark_bipo_vectors.py --strengths 0.5,1.0 # Custom strengths
    python 08_benchmark_bipo_vectors.py --skip-download     # Use local vectors
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from psyctl.core.benchmark.inventory_tester import InventoryTester
from psyctl.core.benchmark.llm_judge_tester import LLMJudgeTester
from psyctl.data.benchmark_settings import (
    get_default_questions,
    get_judge_models,
    get_layer_groups,
    get_prompts,
)

load_dotenv()
console = Console()

# ---------------------------------------------------------------------------
# HuggingFace repo & default config
# ---------------------------------------------------------------------------
HF_REPO_ID = "dalekwon/bipo-steering-vectors"
DEFAULT_STRENGTHS = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]

# ---------------------------------------------------------------------------
# Vector catalogue
# ---------------------------------------------------------------------------
VECTORS: list[dict[str, Any]] = [
    # --- English (meta-llama/Llama-3.1-8B-Instruct, layer=model.layers.13.mlp) ---
    {
        "filename": "bipo_steering_english_agreeableness.safetensors",
        "trait": "agreeableness",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lang": "en",
        "big_five_domain": "A",
        "judge_questions": [
            "Do you tend to consider other people's feelings before making decisions?",
            "When someone disagrees with you, do you try to find common ground?",
            "Do you find it easy to trust people you've just met?",
            "If a friend asks for help at an inconvenient time, would you still help?",
            "Do you prefer cooperation over competition?",
            "When someone shares their problems, do you feel genuine empathy?",
            "Are you willing to compromise to avoid conflict?",
            "Would others describe you as warm and kind?",
        ],
    },
    {
        "filename": "bipo_steering_english_neuroticism.safetensors",
        "trait": "neuroticism",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lang": "en",
        "big_five_domain": "N",
        "judge_questions": [
            "Do small setbacks tend to stress you out easily?",
            "Do you often feel anxious or worried without a clear reason?",
            "When things go wrong, do you tend to blame yourself?",
            "Do you find it hard to relax even when there's nothing to worry about?",
            "Are you more sensitive to criticism than most people?",
            "Do your moods change frequently throughout the day?",
            "Do you tend to overthink about things that might go wrong?",
            "Would others say you're an emotionally reactive person?",
        ],
    },
    {
        "filename": "bipo_steering_english_awfully_sweet.safetensors",
        "trait": "awfully_sweet",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lang": "en",
        "judge_questions": [
            "How would you greet someone who seems to be having a bad day?",
            "If a stranger dropped their groceries, what would you do?",
            "How do you show appreciation to people who help you?",
            "What do you think makes someone truly kind?",
            "How would you comfort a friend going through a difficult time?",
            "Do you go out of your way to make others feel special?",
            "How do you handle it when someone is rude to you?",
            "Would you describe yourself as someone who spreads positivity?",
        ],
    },
    {
        "filename": "bipo_steering_english_paranoid.safetensors",
        "trait": "paranoid",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lang": "en",
        "judge_questions": [
            "Do you think people often have hidden motives when they're nice to you?",
            "Do you feel like you need to be on guard in social situations?",
            "When someone whispers nearby, do you wonder if they're talking about you?",
            "Do you find it hard to trust even close friends completely?",
            "Do you believe people are generally out for themselves?",
            "Do you tend to read negative intentions into neutral actions?",
            "Would you say you're more suspicious than most people?",
            "Do you keep your guard up because you've been burned before?",
        ],
    },
    {
        "filename": "bipo_steering_english_very_lascivious.safetensors",
        "trait": "very_lascivious",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "lang": "en",
        "judge_questions": [
            "Do you enjoy exploring new and unconventional experiences?",
            "Would you describe yourself as someone who follows their desires?",
            "How do you feel about pushing boundaries in social situations?",
            "Do you find yourself drawn to intense sensory experiences?",
            "Are you comfortable expressing your deepest feelings openly?",
            "Do you consider yourself more adventurous than most people?",
            "How important is physical and emotional expression to you?",
            "Would others describe you as someone who lives boldly?",
        ],
    },
    # --- Korean (LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct) ---
    {
        "filename": "bipo_steering_korean_awfully_sweet.safetensors",
        "trait": "awfully_sweet_kr",
        "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "lang": "ko",
    },
    {
        "filename": "bipo_steering_korean_rude.safetensors",
        "trait": "rude_kr",
        "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "lang": "ko",
    },
    {
        "filename": "bipo_steering_korean_lewd.safetensors",
        "trait": "lewd_kr",
        "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "lang": "ko",
    },
]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_vectors(output_dir: Path, vectors: list[dict[str, Any]]) -> dict[str, Path]:
    """Download vectors from HuggingFace. Returns {filename: local_path}."""
    from huggingface_hub import hf_hub_download

    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN")
    paths: dict[str, Path] = {}

    for vec in vectors:
        fname = vec["filename"]
        local = vector_dir / fname
        if local.exists():
            paths[fname] = local
            continue
        console.print(f"  Downloading [cyan]{fname}[/cyan] ...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID, filename=fname,
            local_dir=str(vector_dir), token=token,
        )
        paths[fname] = Path(downloaded)

    return paths


# ---------------------------------------------------------------------------
# Result printers
# ---------------------------------------------------------------------------
def print_inventory(result: dict[str, Any], label: str, strength: float) -> None:
    comparison = result.get("comparison")
    if not comparison:
        return
    table = Table(title=f"{label} | strength={strength}")
    table.add_column("Domain", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Steered", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("%", justify="right")
    for code, d in comparison.items():
        cc = "green" if d["change"] > 0 else "red" if d["change"] < 0 else "white"
        table.add_row(
            f"{d['domain_name']} ({code})", f"{d['baseline_raw']:.3f}",
            f"{d['steered_raw']:.3f}", f"[{cc}]{d['change']:+.3f}[/{cc}]",
            f"{d['percent_change']:+.1f}%",
        )
    console.print(table)


def print_judge(results: list[dict[str, Any]], label: str) -> None:
    if not results:
        return
    table = Table(title=f"Judge: {label}")
    table.add_column("Strength", justify="right", style="cyan")
    table.add_column("Personality\n(base→steer)", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Relevance\n(base→steer)", justify="right")
    for r in results:
        bl, st = r.get("baseline", {}), r.get("steered")
        cmp = r.get("comparison")
        s = r.get("steering_strength", 0.0)
        if st and cmp:
            pc = cmp.get("personality_change", 0.0)
            cc = "green" if pc > 0 else "red" if pc < 0 else "white"
            table.add_row(
                f"{s:.1f}",
                f"{bl.get('personality_score', 0):.2f} → {st.get('personality_score', 0):.2f}",
                f"[{cc}]{pc:+.2f}[/{cc}]",
                f"{bl.get('relevance_score', 0):.2f} → {st.get('relevance_score', 0):.2f}",
            )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Benchmark BiPO steering vectors")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--english-only", action="store_true")
    p.add_argument("--korean-only", action="store_true")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--judge-only", action="store_true")
    p.add_argument("--strengths", type=str, default=None,
                   help="Comma-separated (default: -3,-2,-1,-0.5,0.5,1,2,3)")
    p.add_argument("--en-model", type=str, default=None,
                   help="Override English model path (e.g. local NAS path)")
    p.add_argument("--ko-model", type=str, default=None,
                   help="Override Korean model path")
    p.add_argument("--output-dir", type=Path, default=Path("./results/benchmark_bipo"))
    args = p.parse_args()

    strengths = (
        [float(s) for s in args.strengths.split(",")]
        if args.strengths else list(DEFAULT_STRENGTHS)
    )

    # Override model paths if provided
    if args.en_model or args.ko_model:
        for v in VECTORS:
            if v["lang"] == "en" and args.en_model:
                v["model"] = args.en_model
            elif v["lang"] == "ko" and args.ko_model:
                v["model"] = args.ko_model

    # Filter vectors by language
    if args.korean_only:
        vecs = [v for v in VECTORS if v["lang"] == "ko"]
    elif args.english_only:
        vecs = [v for v in VECTORS if v["lang"] == "en"]
    else:
        vecs = list(VECTORS)

    run_inventory = not args.judge_only
    run_judge = not args.inventory_only

    # Step 1: Download vectors
    console.print("\n[bold]Step 1: Download vectors[/bold]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_download:
        vdir = args.output_dir / "vectors"
        paths = {v["filename"]: vdir / v["filename"] for v in vecs}
    else:
        paths = download_vectors(args.output_dir, vecs)
    console.print(f"  Ready: [green]{len(paths)}[/green] vectors\n")

    all_results: list[dict[str, Any]] = []

    # Step 2: Inventory benchmark (Big Five-mapped vectors only)
    if run_inventory:
        console.print("[bold]Step 2: IPIP-NEO-120 Inventory Benchmark[/bold]")
        inventory_vecs = [v for v in vecs if v.get("big_five_domain")]
        if not inventory_vecs:
            console.print("  [dim]No Big Five-mapped vectors selected, skipping.[/dim]\n")
        else:
            tester = InventoryTester()
            for vec in inventory_vecs:
                vec_path = paths.get(vec["filename"])
                if not vec_path or not vec_path.exists():
                    continue
                console.print(f"\n  [bold cyan]{vec['trait']}[/bold cyan] (domain: {vec['big_five_domain']})")
                for s in strengths:
                    result = tester.test_inventory(
                        model=vec["model"],
                        steering_vector_path=vec_path,
                        inventory_name="ipip_neo_120",
                        steering_strength=s,
                        target_trait=vec["trait"],
                    )
                    result["bipo_trait"] = vec["trait"]
                    all_results.append(result)
                    print_inventory(result, vec["trait"], s)

    # Step 3: LLM-as-Judge benchmark (all vectors)
    if run_judge:
        console.print("\n[bold]Step 3: LLM-as-Judge Benchmark[/bold]")
        judge_config = get_judge_models().get(
            "local-default", {"type": "local", "model_path": "auto"},
        ).copy()
        tester = LLMJudgeTester(
            prompts=get_prompts(),
            default_questions=get_default_questions(),
            layer_groups_config=get_layer_groups(),
        )
        for vec in vecs:
            vec_path = paths.get(vec["filename"])
            if not vec_path or not vec_path.exists():
                continue
            console.print(f"\n  [bold cyan]{vec['trait']}[/bold cyan] ({vec['model']})")
            questions = vec.get("judge_questions")
            results = tester.test_with_judge(
                model=vec["model"],
                trait=vec["trait"],
                steering_vector_path=vec_path,
                judge_config=judge_config,
                steering_strengths=strengths,
                **({"questions": questions} if questions else {"num_questions": 8}),
            )
            for r in results:
                r["bipo_trait"] = vec["trait"]
            all_results.extend(results)
            print_judge(results, vec["trait"])

    # Save results
    out = args.output_dir / "all_results.json"
    with open(out, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "results": all_results},
                  f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]Done![/bold green] Results saved to [cyan]{out}[/cyan]")


if __name__ == "__main__":
    main()
