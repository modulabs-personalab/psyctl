"""CLI command for LLM Judge-based personality evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from psyctl.core.benchmark.llm_judge_tester import LLMJudgeTester
from psyctl.data import benchmark_settings

console = Console()


@click.command(name="llm-as-judge")
@click.option(
    "--model",
    type=str,
    required=True,
    help="Target model to test (path or name)",
)
@click.option(
    "--trait",
    type=str,
    required=True,
    help="Personality trait to test (e.g., 'extraversion')",
)
@click.option(
    "--steering-vector",
    type=click.Path(exists=True, path_type=Path),
    help="Path to steering vector file (.safetensors)",
)
@click.option(
    "--questions",
    type=click.Path(exists=True, path_type=Path),
    help="Path to JSON file with questions (optional, generates if not provided)",
)
@click.option(
    "--num-questions",
    type=int,
    default=8,
    help="Number of questions to generate (default: 8)",
)
@click.option(
    "--judge-model",
    type=str,
    default="local-default",
    help="Judge model name from config (default: local-default)",
)
@click.option(
    "--strengths",
    type=str,
    default="1.0",
    help="Comma-separated steering strengths (e.g., '0.5,1.0,2.0')",
)
@click.option(
    "--layers",
    type=str,
    default="all",
    help="Layer specification: numbers (0,5,10), ranges (0-5), or keywords (early,middle,late). Default: 'all'",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./results)",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use (default: auto)",
)
def llm_as_judge(
    model: str,
    trait: str,
    steering_vector: Path | None,
    questions: Path | None,
    num_questions: int,
    judge_model: str,
    strengths: str,
    layers: str,
    output: Path,
    device: str,
):
    """
    Evaluate personality steering using LLM as Judge.
    
    This method complements inventory-based testing by using an external
    LLM to evaluate responses for personality traits and relevance.
    
    Examples:
    
        # Test with generated questions and local judge
        psyctl benchmark llm-as-judge --model "Qwen3-8B" --trait "extraversion" \\
            --steering-vector vector.safetensors --judge-model "local-default"
        
        # Test with custom questions and API judge
        psyctl benchmark llm-as-judge --model "Qwen3-8B" --trait "extraversion" \\
            --questions questions.json --judge-model "gpt-4" \\
            --strengths "0.5,1.0,2.0" --layer-groups "all,top-10"
    """
    console.print("\n[bold cyan]LLM as Judge Personality Evaluation[/bold cyan]\n")

    # Parse strengths
    try:
        strength_list = [float(s.strip()) for s in strengths.split(",")]
    except ValueError:
        console.print("[red]Error: Invalid strength values[/red]")
        raise click.Abort()

    # Load judge config from benchmark settings
    judge_models = benchmark_settings.get_judge_models()
    prompts = benchmark_settings.get_prompts()
    default_questions = benchmark_settings.get_default_questions()
    layer_groups_config = benchmark_settings.get_layer_groups()

    if judge_model not in judge_models:
        console.print(f"[red]Error: Judge model '{judge_model}' not found in config[/red]")
        console.print(f"Available models: {', '.join(judge_models.keys())}")
        raise click.Abort()

    judge_cfg = judge_models[judge_model].copy()

    # Load API key from environment if needed
    if "api_key_env" in judge_cfg:
        api_key_env = judge_cfg.pop("api_key_env")
        api_key = os.getenv(api_key_env)
        if api_key:
            judge_cfg["api_key"] = api_key
        else:
            console.print(f"[yellow]Warning: API key not found in {api_key_env}[/yellow]")

    # Load questions if provided
    questions_list = None
    if questions:
        with open(questions) as f:
            questions_data = json.load(f)
            if isinstance(questions_data, list):
                questions_list = questions_data
            elif isinstance(questions_data, dict) and "questions" in questions_data:
                questions_list = questions_data["questions"]
            else:
                console.print("[red]Error: Invalid questions file format[/red]")
                raise click.Abort()
        console.print(f"[green]Loaded {len(questions_list)} questions from {questions}[/green]")

    # Display configuration
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Model: [cyan]{model}[/cyan]")
    console.print(f"  Trait: [cyan]{trait}[/cyan]")
    console.print(f"  Steering Vector: [cyan]{steering_vector or 'None'}[/cyan]")
    console.print(f"  Judge Model: [cyan]{judge_model}[/cyan]")
    console.print(f"  Strengths: [cyan]{strength_list}[/cyan]")
    console.print(f"  Layers: [cyan]{layers}[/cyan]")
    console.print(f"  Output Directory: [cyan]{output}[/cyan]")
    console.print()

    # Initialize tester with config
    tester = LLMJudgeTester(
        prompts=prompts,
        default_questions=default_questions,
        layer_groups_config=layer_groups_config,
    )

    # Set output directory
    output_dir = output if output else benchmark_settings.get_results_dir()
    
    # Run test
    results = tester.test_with_judge(
        model=model,
        trait=trait,
        questions=questions_list,
        num_questions=num_questions,
        steering_vector_path=steering_vector,
        judge_config=judge_cfg,
        steering_strengths=strength_list,
        layer_spec=layers if layers != "all" else None,
        device=device,
        output_dir=output_dir,
    )

    # Display summary
    _display_results_summary(results, layers)

    console.print("\n[bold green]✓ Evaluation complete![/bold green]")
    console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")


def _display_results_summary(results: list[dict], layer_spec: str):
    """Display results summary in a table."""
    if not results:
        return

    table = Table(title=f"Results Summary - Layers: {layer_spec}")
    table.add_column("Strength", justify="right", style="cyan")
    table.add_column("Baseline\nPersonality", justify="right")
    table.add_column("Steered\nPersonality", justify="right")
    table.add_column("Change", justify="right", style="yellow")
    table.add_column("Baseline\nRelevance", justify="right")
    table.add_column("Steered\nRelevance", justify="right")

    for result in results:
        if result["steered"]:
            table.add_row(
                f"{result['steering_strength']:.1f}",
                f"{result['baseline']['personality_score']:.2f}",
                f"{result['steered']['personality_score']:.2f}",
                f"{result['comparison']['personality_change']:+.2f}",
                f"{result['baseline']['relevance_score']:.2f}",
                f"{result['steered']['relevance_score']:.2f}",
            )
        else:
            table.add_row(
                "0.0",
                f"{result['baseline']['personality_score']:.2f}",
                "N/A",
                "N/A",
                f"{result['baseline']['relevance_score']:.2f}",
                "N/A",
            )

    console.print(table)


