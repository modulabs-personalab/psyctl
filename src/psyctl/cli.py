#!/usr/bin/env python3
"""Main CLI entry point for psyctl."""

import click
import torch
from rich.console import Console
from rich.traceback import install

from psyctl import config
from psyctl.commands import (
    benchmark,
    benchmark_judge,
    benchmark_list,
    dataset,
    extract,
    layer,
    steering,
)
from psyctl.core.logger import get_logger, setup_logging

# Disable PyTorch compiler to avoid Triton issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Rich 설정
install(show_locals=True)
console = Console()

# Setup logging and directories
setup_logging()
config.create_directories()
logger = get_logger("cli")


@click.group()
@click.version_option(prog_name="psyctl")
def main():
    """PSYCTL - LLM Personality Steering Tool."""
    logger.info("PSYCTL CLI started")
    pass


# Create benchmark command group
@click.group(name="benchmark")
def benchmark_group():
    """Benchmark personality steering effects."""
    pass


# Register benchmark subcommands
benchmark_group.add_command(benchmark.inventory, name="inventory")
benchmark_group.add_command(benchmark_judge.llm_as_judge, name="llm-as-judge")

# 명령어 등록
main.add_command(dataset.build_steer, name="dataset.build.steer")
main.add_command(dataset.upload, name="dataset.upload")
main.add_command(extract.steering, name="extract.steering")
main.add_command(layer.analyze, name="layer.analyze")
main.add_command(steering.apply, name="steering")
main.add_command(benchmark_group)
main.add_command(benchmark_list.list_inventories, name="inventory.list")

if __name__ == "__main__":
    main()
