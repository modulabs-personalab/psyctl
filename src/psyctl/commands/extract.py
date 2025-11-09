"""Steering vector extraction commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.logger import get_logger
from psyctl.core.steering_extractor import SteeringExtractor

console = Console()
logger = get_logger("extract")


@click.command()
@click.option(
    "--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-3B-Instruct)"
)
@click.option(
    "--layer",
    multiple=True,
    help="Target layer path (can be repeated for multi-layer extraction). "
    "Example: --layer 'model.layers[13].mlp.down_proj' --layer 'model.layers[14].mlp.down_proj'",
)
@click.option(
    "--layers",
    type=str,
    help="Comma-separated list of layer paths. "
    "Example: --layers 'model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj'",
)
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True),
    help="Dataset directory path",
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output safetensors file path"
)
@click.option(
    "--batch-size",
    type=int,
    help="Batch size for inference (default: from config)",
)
@click.option(
    "--normalize",
    is_flag=True,
    help="Normalize steering vectors to unit length",
)
@click.option(
    "--method",
    type=str,
    default="mean_diff",
    help="Extraction method: mean_diff, denoised_mean_diff (PCA-based denoising), or bipo (preference optimization)",
)
@click.option(
    "--lr",
    type=float,
    default=5e-4,
    help="Learning rate for BiPO optimization (default: 5e-4)",
)
@click.option(
    "--beta",
    type=float,
    default=0.1,
    help="Beta parameter for BiPO loss (default: 0.1)",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="Number of training epochs for BiPO (default: 10)",
)
@click.option(
    "--variance-threshold",
    type=float,
    default=0.95,
    help="PCA variance threshold for denoised_mean_diff method (default: 0.95, keeps 95%% of variance)",
)
@click.option(
    "--use-chat-template/--no-chat-template",
    default=True,
    help="Use chat template for prompt formatting (default: enabled)",
)
def steering(
    model: str,
    layer: tuple[str],
    layers: str | None,
    dataset: str,
    output: str,
    batch_size: int | None,
    normalize: bool,
    method: str,
    lr: float,
    beta: float,
    epochs: int,
    variance_threshold: float,
    use_chat_template: bool,
):
    """
    Extract steering vectors using various methods.

    Supports single or multi-layer extraction. Specify layers using either:
    - Multiple --layer flags: --layer "model.layers[13].mlp.down_proj" --layer "model.layers[14].mlp.down_proj"
    - Comma-separated --layers: --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj"

    Methods:
    - mean_diff: Fast statistical method using mean activation difference (default)
    - denoised_mean_diff: PCA-based denoising for noise reduction and improved robustness
    - bipo: Optimization-based method using preference learning

    Examples:

    \b
    # mean_diff method (fast, statistical - default)
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layer "model.layers[13].mlp.down_proj" \\
      --dataset "./dataset/steering" \\
      --output "./steering_vector/out.safetensors"

    \b
    # denoised_mean_diff method (PCA-based denoising, more robust)
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layer "model.layers[13].mlp.down_proj" \\
      --dataset "./dataset/steering" \\
      --output "./steering_vector/out.safetensors" \\
      --method denoised_mean_diff \\
      --variance-threshold 0.95

    \b
    # BiPO method (optimization-based)
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layer "model.layers[13].mlp" \\
      --dataset "./dataset/caa" \\
      --output "./steering_vector/out.safetensors" \\
      --method bipo \\
      --lr 5e-4 \\
      --beta 0.1 \\
      --epochs 10

    \b
    # Multi-layer extraction with denoised mean difference
    psyctl extract.steering \\
      --model "meta-llama/Llama-3.2-3B-Instruct" \\
      --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj" \\
      --dataset "./dataset/steering" \\
      --output "./steering_vector/out.safetensors" \\
      --method denoised_mean_diff
    """
    logger.info("Starting steering vector extraction")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Output: {output}")
    logger.info(f"Method: {method}")
    logger.info(f"Use chat template: {use_chat_template}")

    # Parse layer arguments
    layer_list = []

    # Add layers from --layer flags (multiple)
    if layer:
        layer_list.extend(layer)

    # Add layers from --layers (comma-separated)
    if layers:
        layer_list.extend([
            layer_str.strip() for layer_str in layers.split(",") if layer_str.strip()
        ])

    # Validate that at least one layer is specified
    if not layer_list:
        raise click.UsageError(
            "No layers specified. Use --layer or --layers to specify target layers."
        )

    logger.info(f"Target layers ({len(layer_list)}): {layer_list}")

    console.print("[blue]Extracting steering vectors...[/blue]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    console.print(f"Layers ({len(layer_list)}):")
    for idx, layer_name in enumerate(layer_list, 1):
        console.print(f"  {idx}. [yellow]{layer_name}[/yellow]")
    console.print(f"Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"Output: [cyan]{output}[/cyan]")
    console.print(f"Method: [cyan]{method}[/cyan]")
    console.print(f"Use chat template: [cyan]{use_chat_template}[/cyan]")
    if normalize:
        console.print("[yellow]Normalization: Enabled[/yellow]")
    if batch_size:
        console.print(f"Batch size: {batch_size}")

    # Display method-specific parameters
    if method == "bipo":
        console.print("[yellow]BiPO Parameters:[/yellow]")
        console.print(f"  - Learning rate: {lr}")
        console.print(f"  - Beta: {beta}")
        console.print(f"  - Epochs: {epochs}")
    elif method == "denoised_mean_diff":
        console.print("[yellow]Denoised Mean Difference Parameters:[/yellow]")
        console.print(f"  - Variance threshold: {variance_threshold:.2%}")

    try:
        extractor = SteeringExtractor()

        # Prepare method-specific parameters
        method_params = {}
        if method == "bipo":
            method_params = {
                "lr": lr,
                "beta": beta,
                "epochs": epochs,
            }
        elif method == "denoised_mean_diff":
            method_params = {
                "variance_threshold": variance_threshold,
            }

        vectors = extractor.extract_steering_vector(
            model_name=model,
            layers=layer_list,
            dataset_path=Path(dataset),
            output_path=Path(output),
            batch_size=batch_size,
            normalize=normalize,
            method=method,
            use_chat_template=use_chat_template,
            **method_params,  # type: ignore[arg-type]
        )

        logger.info(f"Steering vectors extracted successfully to {output}")
        console.print(
            f"[green]Extracted {len(vectors)} steering vectors to {output}[/green]"
        )

        # Display vector info
        console.print("\n[bold]Extracted Vectors:[/bold]")
        for layer_name, vector in vectors.items():
            norm = vector.norm().item()
            console.print(
                f"  - [cyan]{layer_name}[/cyan]: shape={list(vector.shape)}, norm={norm:.4f}"
            )

    except Exception as e:
        logger.error(f"Failed to extract steering vector: {e}")
        raise
