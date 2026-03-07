"""Inventory registry for dynamic inventory discovery and creation."""

from __future__ import annotations

from typing import Any

from psyctl.data.inventories.base import BaseInventory

INVENTORY_REGISTRY: dict[str, type[BaseInventory]] = {}


def register_inventory(name: str):
    """Decorator to register an inventory class.

    Args:
        name: Registry key for the inventory (e.g., "ipip_neo")

    Returns:
        Decorator function
    """

    def decorator(cls: type[BaseInventory]) -> type[BaseInventory]:
        INVENTORY_REGISTRY[name] = cls
        return cls

    return decorator


def create_inventory(inventory_name: str) -> BaseInventory:
    """Factory function to create inventory by name.

    Parses inventory_name into prefix and version.
    For example, "ipip_neo_120" -> prefix="ipip_neo", version="120".

    Args:
        inventory_name: Full inventory name (e.g., "ipip_neo_120")

    Returns:
        Instantiated inventory object

    Raises:
        ValueError: If inventory type is not registered
    """
    parts = inventory_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        prefix, version = parts[0], parts[1]
    else:
        prefix, version = inventory_name, None

    if prefix not in INVENTORY_REGISTRY:
        available = ", ".join(INVENTORY_REGISTRY.keys())
        raise ValueError(
            f"Unknown inventory: '{inventory_name}'. Available: {available}"
        )

    cls = INVENTORY_REGISTRY[prefix]
    return cls(version=version) if version else cls()


def get_available_inventories() -> list[str]:
    """Get list of all registered inventory names with versions.

    Returns:
        List of inventory name strings (e.g., ["ipip_neo_120", "ipip_neo_300"])
    """
    result: list[str] = []
    for prefix, cls in INVENTORY_REGISTRY.items():
        try:
            inv = cls()
            result.append(f"{prefix}_{inv.version}" if inv.version else prefix)
        except Exception:
            result.append(prefix)
    return result


def get_registry_info() -> list[dict[str, Any]]:
    """Get detailed info about all registered inventories.

    Returns:
        List of dicts with inventory metadata
    """
    info: list[dict[str, Any]] = []
    for prefix, cls in INVENTORY_REGISTRY.items():
        try:
            inv = cls()
            inv_info = inv.get_inventory_info()
            inv_info["registry_key"] = prefix
            info.append(inv_info)
        except Exception:
            info.append({"registry_key": prefix, "name": prefix, "error": True})
    return info
