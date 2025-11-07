"""Benchmark commands for inventory testing."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from psyctl.core.benchmark.inventory_tester import InventoryTester
from psyctl.core.logger import get_logger
from psyctl.data.benchmark_settings import get_results_dir

console = Console()
logger = get_logger("benchmark")


@click.command()
@click.option("--model", required=True, help="Model name or path")
@click.option(
    "--steering-vector",
    type=click.Path(exists=True),
    help="Steering vector file path (optional, omit for baseline only)",
)
@click.option(
    "--inventory",
    default="ipip_neo_120",
    help="Inventory name (default: ipip_neo_120)",
)
@click.option(
    "--strength",
    type=float,
    default=1.0,
    help="Steering strength (default: 1.0)",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output JSON file path (optional)",
)
@click.option(
    "--device",
    default="auto",
    help="Device to use (default: auto)",
)
@click.option(
    "--trait",
    type=str,
    default=None,
    help="Specific trait to test (N/E/O/A/C or Neuroticism/Extraversion/etc). Tests all traits if not specified.",
)
@click.option(
    "--layers",
    type=str,
    default=None,
    help="Layer specification for steering. Supports: direct numbers (0,5,10), ranges (0-5), keywords (all/early/middle/late), or combinations. Default: all layers.",
)
def inventory(
    model: str,
    steering_vector: str | None,
    inventory: str,
    strength: float,
    output: str | None,
    device: str,
    trait: str | None,
    layers: str | None,
):
    """Run inventory test to measure personality changes."""
    # Compact output
    console.print("\n[bold blue]🧠 Personality Inventory Benchmark[/bold blue]")
    console.print(f"📦 Model: {model.split('/')[-1]}")
    console.print(f"📋 Inventory: {inventory}")
    if trait:
        console.print(f"🎯 Target Trait: {trait}")
    if steering_vector:
        console.print(f"🔧 Steering: {Path(steering_vector).name} (strength={strength})")
        if layers:
            console.print(f"📍 Layers: {layers}")
        else:
            console.print("📍 Layers: All layers")
    console.print("")

    try:
        tester = InventoryTester()
        results = tester.test_inventory(
            model=model,
            steering_vector_path=Path(steering_vector) if steering_vector else None,
            inventory_name=inventory,
            steering_strength=strength,
            device=device,
            target_trait=trait,
            layer_spec=layers,  # Pass layer spec string directly
        )

        # Print baseline results
        console.print("\n[bold green]📊 Baseline Results (No Steering)[/bold green]\n")
        _print_domain_table(results["baseline"])

        # Print steered results if available
        if results["steered"]:
            console.print(
                "\n[bold yellow]🎯 Steered Results (With Steering)[/bold yellow]\n"
            )
            _print_domain_table(results["steered"])

            # Print comparison
            console.print(
                "\n[bold cyan]📈 Comparison (Steered vs Baseline)[/bold cyan]\n"
            )
            _print_comparison_table(results["comparison"])

        # Save results
        if output:
            output_path = Path(output)
        else:
            results_dir = get_results_dir()
            results_dir.mkdir(parents=True, exist_ok=True)
            model_name = model.split("/")[-1]
            inventory_name = inventory.replace("_", "-")
            output_path = (
                results_dir / f"inventory_{model_name}_{inventory_name}.json"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        console.print(f"\n💾 Results saved to: {output_path}")
        logger.success("Inventory test completed")

    except Exception as e:
        logger.error(f"Failed to run inventory test: {e}")
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        raise


def _print_domain_table(scores: dict[str, dict[str, float]]):
    """Print domain scores as a table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Domain", style="cyan")
    table.add_column("Raw Score", justify="right")
    table.add_column("Population Mean", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("Percentile", justify="right", style="green")

    for domain_code in ["N", "E", "O", "A", "C"]:
        if domain_code in scores:
            s = scores[domain_code]
            table.add_row(
                s["domain_name"],
                f"{s['raw_score']:.2f} / {s['num_items']*5}",
                f"{s['population_mean']:.1f} ± {s['population_std']:.1f}",
                f"{s['z_score']:+.2f}",
                f"{s['percentile']:.1f}%",
            )

    console.print(table)


def _print_comparison_table(comparison: dict[str, dict[str, float]]):
    """Print comparison table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Domain", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Steered", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("% Change", justify="right", style="yellow")
    table.add_column("Z-Score Δ", justify="right", style="magenta")

    for domain_code in ["N", "E", "O", "A", "C"]:
        if domain_code in comparison:
            c = comparison[domain_code]
            change_style = "green" if c["change"] > 0 else "red"
            table.add_row(
                c["domain_name"],
                f"{c['baseline_raw']:.2f}",
                f"{c['steered_raw']:.2f}",
                f"[{change_style}]{c['change']:+.2f}[/{change_style}]",
                f"[{change_style}]{c['percent_change']:+.1f}%[/{change_style}]",
                f"{c['z_change']:+.2f}",
            )

    console.print(table)
