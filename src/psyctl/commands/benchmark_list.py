"""List available inventories command."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from psyctl.core.logger import get_logger
from psyctl.data.inventories.ipip_neo import IPIPNEO

console = Console()
logger = get_logger("benchmark.list")


@click.command()
@click.option(
    "--inventory",
    type=str,
    default=None,
    help="Specific inventory to show info for (e.g., ipip_neo_120). Shows all if not specified.",
)
def list_inventories(inventory: str | None):
    """List available personality inventories and their supported traits."""
    console.print("\n[bold blue]📚 Available Personality Inventories[/bold blue]\n")

    # Currently only IPIP-NEO is supported, but this is extensible
    available_inventories = ["ipip_neo_120", "ipip_neo_300"]

    if inventory:
        # Show detailed info for specific inventory
        try:
            version = inventory.split("_")[-1] if "_" in inventory else "120"
            inv = IPIPNEO(version=version)
            info = inv.get_inventory_info()

            console.print(f"[bold green]{info['name']}[/bold green]")
            console.print(f"Version: {info['version']}")
            console.print(f"Total Questions: {info['total_questions']}")
            console.print(
                f"Questions per Trait: {info['questions_per_trait']}\n"
            )

            # Traits table
            table = Table(title="Supported Traits", show_header=True)
            table.add_column("Code", style="cyan", width=6)
            table.add_column("Full Name", style="magenta")

            for trait in info["traits"]:
                table.add_row(trait["code"], trait["name"])

            console.print(table)
            console.print(
                "\n[dim]Usage: psyctl benchmark --trait <code> ...[/dim]\n"
            )

        except Exception as e:
            console.print(f"[red]Error loading inventory '{inventory}': {e}[/red]")
            logger.error(f"Failed to load inventory: {e}")

    else:
        # Show list of all available inventories
        table = Table(show_header=True)
        table.add_column("Inventory", style="cyan", width=20)
        table.add_column("Version", style="green", width=10)
        table.add_column("Questions", style="yellow", width=12)
        table.add_column("Traits", style="magenta")

        for inv_name in available_inventories:
            try:
                version = inv_name.split("_")[-1] if "_" in inv_name else "120"
                inv = IPIPNEO(version=version)
                info = inv.get_inventory_info()

                trait_codes = ", ".join([t["code"] for t in info["traits"]])
                table.add_row(
                    inv_name,
                    info["version"],
                    str(info["total_questions"]),
                    trait_codes,
                )
            except Exception:
                # Skip inventories that can't be loaded
                continue

        console.print(table)
        console.print(
            "\n[dim]For detailed info: psyctl inventory list --inventory <name>[/dim]\n"
        )

