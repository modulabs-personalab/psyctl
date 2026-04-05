"""Dataset generation commands."""

from pathlib import Path

import click
from rich.console import Console

from psyctl.core.dataset_builder import DatasetBuilder
from psyctl.core.logger import get_logger

console = Console()
logger = get_logger("dataset")


@click.command()
@click.option(
    "--model", required=False, help="Model name (e.g., google/gemma-3-27b-it)"
)
@click.option(
    "--personality",
    required=True,
    help="Personality traits (e.g., Extroversion, Machiavellism)",
)
@click.option(
    "--output", required=True, type=click.Path(), help="Output directory path"
)
@click.option(
    "--limit-samples",
    required=False,
    type=int,
    default=0,
    help="Maximum number of samples to generate",
)
@click.option(
    "--dataset",
    required=False,
    default="allenai/soda",
    help="Hugging Face dataset name (e.g., allenai/soda, username/custom-dataset)",
)
@click.option(
    "--openrouter-api-key",
    required=False,
    help="OpenRouter API key (format: sk-or-xxxx). If provided, uses OpenRouter instead of local model.",
)
@click.option(
    "--openrouter-max-workers",
    required=False,
    type=int,
    default=1,
    help="Number of parallel workers for OpenRouter API (1 = sequential, higher = parallel)",
)
@click.option(
    "--roleplay-prompt-template",
    required=False,
    type=click.Path(exists=True),
    help="Path to custom Jinja2 template for roleplay prompts (.j2 file)",
)
def build_steer(
    model: str,
    personality: str,
    output: str,
    limit_samples: int,
    dataset: str,
    openrouter_api_key: str,
    openrouter_max_workers: int,
    roleplay_prompt_template: str,
):
    """Build steering dataset for steering vector extraction."""
    # Determine if using OpenRouter or local model
    use_openrouter = bool(openrouter_api_key)

    # Validate configuration
    if use_openrouter:
        logger.info("Using OpenRouter API mode")
        console.print("[yellow]Using OpenRouter API mode[/yellow]")
        if not model:
            model = "openrouter"  # Placeholder when using OpenRouter
    else:
        if not model:
            logger.error("--model is required when not using OpenRouter")
            console.print(
                "[red]Error: --model is required when not using --openrouter-api-key[/red]"
            )
            raise click.BadParameter("--model is required when not using OpenRouter")
        logger.info("Using local model mode")

    logger.info("Starting steering dataset build")
    logger.info(f"Model: {model}")
    logger.info(f"Personality: {personality}")
    logger.info(f"Output: {output}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Limit samples: {limit_samples}")

    console.print("[blue]Building steering dataset...[/blue]")

    if use_openrouter:
        console.print(f"OpenRouter Model: {model}")
        console.print(f"OpenRouter Workers: {openrouter_max_workers}")
    else:
        console.print(f"Local Model: {model}")

    console.print(f"Personality: {personality}")
    console.print(f"Output: {output}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Limit samples: {limit_samples}")

    try:
        builder = DatasetBuilder(
            use_openrouter=use_openrouter,
            openrouter_api_key=openrouter_api_key,
            openrouter_max_workers=openrouter_max_workers,
            roleplay_prompt_template=roleplay_prompt_template,
        )
        output_file = builder.build_steer_dataset(
            model, personality, Path(output), limit_samples, dataset
        )

        logger.info(f"Dataset built successfully: {output_file}")
        console.print(f"[green]Dataset built successfully: {output_file}[/green]")
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise


@click.command()
@click.option(
    "--dataset-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSONL dataset file to upload",
)
@click.option(
    "--repo-id",
    required=True,
    help="HuggingFace repository ID (format: username/repo-name)",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Make repository private (default: public)",
)
@click.option(
    "--commit-message",
    default="Upload steering dataset via PSYCTL",
    help="Commit message for upload",
)
@click.option(
    "--license",
    required=False,
    help="License identifier (e.g., 'mit', 'apache-2.0', 'cc-by-4.0')",
)
@click.option(
    "--personality",
    required=False,
    help="Personality trait for dataset card (e.g., 'Extroversion', 'Rudeness')",
)
@click.option(
    "--model",
    required=False,
    help="Model name used to generate dataset (e.g., 'meta-llama/Llama-3.2-3B-Instruct')",
)
@click.option(
    "--dataset-source",
    required=False,
    help="Source dataset used (e.g., 'allenai/soda', 'CaveduckAI/simplified_soda_kr')",
)
def upload(
    dataset_file: str,
    repo_id: str,
    private: bool,
    commit_message: str,
    license: str,
    personality: str,
    model: str,
    dataset_source: str,
):
    """Upload steering dataset to HuggingFace Hub."""
    from psyctl.core.utils import validate_hf_token

    logger.info("Starting dataset upload to HuggingFace Hub")
    console.print("[blue]Uploading steering dataset to HuggingFace Hub...[/blue]")

    # Validate HF_TOKEN early
    try:
        token = validate_hf_token()
        masked_token = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
        logger.info(f"HF_TOKEN found: {masked_token}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise click.ClickException(str(e)) from e

    console.print(f"Dataset File: {dataset_file}")
    console.print(f"Repository: {repo_id}")
    console.print(f"Privacy: {'Private' if private else 'Public'}")
    if license:
        console.print(f"License: {license}")

    try:
        builder = DatasetBuilder()
        repo_url = builder.upload_to_hub(
            jsonl_file=Path(dataset_file),
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
            token=token,
            license=license,
            personality=personality,
            model=model,
            dataset_source=dataset_source,
        )

        logger.info("Upload completed successfully")
        console.print(f"[green]Successfully uploaded to: {repo_url}[/green]")
        console.print(f"\n[blue]View your dataset at:[/blue]\n{repo_url}")

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        console.print(f"[red]Upload failed: {e}[/red]")
        raise
