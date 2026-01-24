# Adding a New Inventory Test

This guide walks through how to add a new psychological inventory to PSYCTL, step by step. The system uses a registry pattern that allows new inventories to be added without modifying existing code.

## Architecture Overview

The inventory system consists of these components:

```
src/psyctl/data/
├── benchmark_config.json          # Centralized configuration (population norms, metadata)
└── inventories/
    ├── base.py                    # BaseInventory abstract class
    ├── registry.py                # @register_inventory decorator + factory
    ├── __init__.py                # Package exports
    ├── ipip_neo.py                # IPIP-NEO implementation (Big Five)
    ├── ipip_neo_120_items_en.json # IPIP-NEO question data
    ├── rei.py                     # REI implementation (Dual-Process)
    └── rei_40_items_en.json       # REI question data
```

**Key Design:**
- Each inventory is a class inheriting from `BaseInventory`
- The `@register_inventory("name")` decorator registers it to the global registry
- The `create_inventory("name_version")` factory instantiates by name
- Configuration (population norms, metadata) lives in `benchmark_config.json`
- Question items live in separate JSON files

---

## Step-by-Step Tutorial

We'll use a hypothetical "Rosenberg Self-Esteem Scale (RSES-10)" as an example. This is a 10-item scale measuring global self-esteem with a single domain.

### Step 1: Prepare the Question Data File

Create `src/psyctl/data/inventories/rses_10_items_en.json`:

```json
[
  {
    "id": "rses-001",
    "text": "On the whole, I am satisfied with myself.",
    "keyed": "plus",
    "domain": "SE",
    "facet": null,
    "num": 1,
    "choices": [
      {"text": "Strongly Disagree", "score": 1},
      {"text": "Disagree", "score": 2},
      {"text": "Agree", "score": 3},
      {"text": "Strongly Agree", "score": 4}
    ]
  },
  {
    "id": "rses-002",
    "text": "At times I think I am no good at all.",
    "keyed": "minus",
    "domain": "SE",
    "facet": null,
    "num": 2,
    "choices": [
      {"text": "Strongly Disagree", "score": 1},
      {"text": "Disagree", "score": 2},
      {"text": "Agree", "score": 3},
      {"text": "Strongly Agree", "score": 4}
    ]
  }
]
```

**Required fields per item:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique item identifier (e.g., `"rses-001"`) |
| `text` | string | The statement presented to the respondent |
| `keyed` | `"plus"` or `"minus"` | Scoring direction. `"minus"` items are reverse-scored |
| `domain` | string | Domain/subscale code (e.g., `"SE"` for Self-Esteem) |
| `facet` | string or null | Optional sub-facet identifier |
| `num` | int | Item sequence number (1-indexed) |
| `choices` | array | Response options with text labels and numeric scores |

**Notes on `keyed`:**
- `"plus"`: Higher score on the Likert scale = higher trait score
- `"minus"`: The score is reversed during calculation: `score = (max + 1) - raw_score`
  - For a 1-5 scale: reversed = `6 - score`
  - For a 1-4 scale: reversed = `5 - score`

---

### Step 2: Add Configuration to benchmark_config.json

Open `src/psyctl/data/benchmark_config.json` and add an entry under the `"inventories"` section:

```json
{
  "inventories": {
    "ipip_neo_120": { ... },
    "rei_40": { ... },
    "rses_10": {
      "name": "RSES-10",
      "description": "Rosenberg Self-Esteem Scale (10 items)",
      "version": "10",
      "language": "en",
      "num_questions": 10,
      "data_file": "rses_10_items_en.json",
      "source": "Rosenberg, M. (1965). Society and the adolescent self-image.",
      "license": "Public Domain",
      "domains": {
        "SE": {
          "name": "Self-Esteem",
          "num_items": 10,
          "population_mean": 22.0,
          "population_std": 5.0
        }
      }
    }
  }
}
```

**Required configuration fields:**

| Field | Description |
|-------|-------------|
| `name` | Display name of the inventory |
| `description` | Brief description |
| `version` | Version string (used in registry key parsing) |
| `language` | Language code (e.g., `"en"`) |
| `num_questions` | Total number of items |
| `data_file` | Filename of the items JSON (relative to inventories/) |
| `source` | Citation or URL |
| `license` | License type |
| `domains` | Dict mapping domain codes to their configuration |

**Domain configuration fields:**

| Field | Description |
|-------|-------------|
| `name` | Full domain name |
| `num_items` | Number of items in this domain |
| `population_mean` | Population mean for z-score calculation |
| `population_std` | Population standard deviation for z-score calculation |

**Where to find population norms:**
- Original publication of the scale
- Meta-analysis papers
- Test manual or technical documentation
- If unavailable, use mean = (num_items * midpoint) and std estimated from similar scales

---

### Step 3: Implement the Inventory Class

Create `src/psyctl/data/inventories/rses.py`:

```python
"""RSES (Rosenberg Self-Esteem Scale) implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("rses")
class RSES(BaseInventory):
    """RSES (Rosenberg Self-Esteem Scale) - Global self-esteem.

    Measures global self-esteem based on Rosenberg (1965).

    Domains:
        SE: Self-Esteem
    """

    def __init__(self, version: str = "10"):
        """Initialize RSES inventory.

        Args:
            version: Version of the inventory ("10")
        """
        self.name = f"RSES-{version}"
        self.domain = "Self-Esteem"
        self.license = "Public Domain"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"rses_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(
                f"RSES version '{self.version}' not found in config"
            )

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported traits.

        Returns:
            List of dicts with trait code and full name
        """
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code.

        Args:
            trait: Trait code or full name

        Returns:
            Normalized trait code

        Raises:
            ValueError: If trait is not recognized
        """
        trait_map: dict[str, str] = {
            "SE": "SE",
            "Self-Esteem": "SE",
            "self-esteem": "SE",
            "self esteem": "SE",
        }

        trait_code = trait_map.get(trait) or trait_map.get(trait.lower())

        if trait_code is None:
            valid_traits = list(self.config["domains"].keys())
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores from responses.

        Args:
            responses: Dict mapping domain codes to list of scores

        Returns:
            Dict with domain scores including raw score, z-score, percentile
        """
        results: dict[str, dict[str, float]] = {}

        for domain, scores in responses.items():
            if domain not in self.config["domains"]:
                continue

            domain_config = self.config["domains"][domain]
            raw_score = sum(scores)
            mean = domain_config["population_mean"]
            std = domain_config["population_std"]

            z_score = (raw_score - mean) / std if std > 0 else 0.0
            percentile = self._z_to_percentile(z_score)

            results[domain] = {
                "domain_name": domain_config["name"],
                "raw_score": raw_score,
                "mean_score": raw_score / len(scores) if scores else 0.0,
                "population_mean": mean,
                "population_std": std,
                "z_score": z_score,
                "percentile": percentile,
                "num_items": float(len(scores)),
            }

        return results

    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile using normal CDF approximation.

        Args:
            z_score: Standard score

        Returns:
            Percentile (0-100)
        """
        if abs(z_score) < 3:
            percentile = 50 + 34.13 * z_score
        elif z_score >= 3:
            percentile = 99.87
        else:
            percentile = 0.13

        return max(0.0, min(100.0, percentile))
```

**Key implementation points:**

1. **`@register_inventory("rses")`** - The decorator name becomes the registry prefix. Combined with version, the full key is `"rses_10"`.

2. **`__init__`** - Set `self.name`, `self.domain`, `self.license` before calling `super().__init__()`. The parent class calls `_load_config()` and `_load_questions()`.

3. **`_load_config`** - Reads `benchmark_config.json` and extracts the section for this inventory.

4. **`_load_questions`** - Reads the items JSON file specified in the config's `data_file` field.

5. **`_normalize_trait`** - Maps various trait representations (codes, full names, lowercase) to the canonical domain code.

6. **`calculate_scores`** - Receives pre-organized responses as `{"SE": [3.0, 4.0, 2.0, ...]}` and computes raw scores, z-scores, and percentiles.

---

### Step 4: Register in __init__.py

Edit `src/psyctl/data/inventories/__init__.py`:

```python
"""Psychological inventory modules."""

from .base import BaseInventory
from .ipip_neo import IPIPNEO
from .registry import (
    INVENTORY_REGISTRY,
    create_inventory,
    get_available_inventories,
    get_registry_info,
)
from .rei import REI
from .rses import RSES  # Add this import

__all__ = [
    "INVENTORY_REGISTRY",
    "IPIPNEO",
    "REI",
    "RSES",              # Add to __all__
    "BaseInventory",
    "create_inventory",
    "get_available_inventories",
    "get_registry_info",
]
```

The import triggers the `@register_inventory` decorator, registering the class in `INVENTORY_REGISTRY`.

---

### Step 5: Verify

Run these checks to confirm the inventory is properly integrated:

```powershell
# 1. Lint check
uv run ruff check src/psyctl/data/inventories/rses.py

# 2. Type check
uv run pyright src/psyctl/data/inventories/rses.py

# 3. Verify registration
uv run python -c "
from psyctl.data.inventories import create_inventory, get_available_inventories
print('Available:', get_available_inventories())
inv = create_inventory('rses_10')
print('Name:', inv.name)
print('Traits:', inv.get_supported_traits())
print('Questions:', len(inv.get_questions()))
"

# 4. Run existing tests (ensure nothing is broken)
uv run pytest tests/ -v
```

Expected output:
```
Available: ['ipip_neo_120', 'rei_40', 'rses_10']
Name: RSES-10
Traits: [{'code': 'SE', 'name': 'Self-Esteem'}]
Questions: 10
```

---

## Advanced Topics

### Multi-Domain Inventories with Parent Scales

For inventories with hierarchical structure (subscales + parent scales), like REI-40:

```python
def calculate_scores(self, responses):
    results = {}

    # 1. Calculate subscale scores
    for domain, scores in responses.items():
        # ... standard calculation ...

    # 2. Calculate parent scales by combining subscales
    parent_scales = {
        "R": ["RA", "RE"],  # Rationality = Rational Ability + Engagement
        "E": ["EA", "EE"],  # Experientiality = Experiential Ability + Engagement
    }

    for parent, subscales in parent_scales.items():
        combined = []
        for sub in subscales:
            if sub in responses:
                combined.extend(responses[sub])
        if combined:
            # Calculate using parent scale norms from config
            ...

    return results
```

Also override `get_questions()` to support parent-scale filtering:

```python
def get_questions(self, trait=None):
    if trait is None:
        return self.questions.copy()

    trait_code = self._normalize_trait(trait)

    # Parent scale: return questions from all child subscales
    if trait_code == "R":
        return [q for q in self.questions if q.get("domain") in ("RA", "RE")]

    # Subscale: filter directly
    return [q for q in self.questions if q.get("domain") == trait_code]
```

### Different Likert Scales

The scoring system supports any scale range. The `choices` array in the items JSON defines the scale:

- **5-point** (IPIP-NEO, REI): scores 1-5, reverse = `6 - score`
- **4-point** (RSES): scores 1-4, reverse = `5 - score`
- **7-point**: scores 1-7, reverse = `8 - score`

The reverse scoring formula is: `reversed = (max_score + 1) - raw_score`

This is handled in the test runner (`inventory_tester.py`) which checks the `keyed` field:
```python
if question["keyed"] == "minus":
    score = 6.0 - score  # Adjust based on your scale's max
```

### Testing with OpenRouter API Models

For API-based testing (no logit access), use the chat-based approach:

```python
from psyctl.data.inventories import create_inventory
from psyctl.models.openrouter_client import OpenRouterClient

inventory = create_inventory("rses_10")
questions = inventory.get_questions()
client = OpenRouterClient(api_key="...")

domain_responses = {}
for question in questions:
    _, response = client.generate(
        prompt=f'Statement: "{question["text"]}"\nYour rating (1-4):',
        model="anthropic/claude-sonnet-4",
        system_prompt="Respond with ONLY a number 1-4...",
    )
    score = parse_score(response)
    if question["keyed"] == "minus":
        score = 5.0 - score  # 4-point scale reversal

    domain = question["domain"]
    domain_responses.setdefault(domain, []).append(score)

scores = inventory.calculate_scores(domain_responses)
```

See `examples/09_openrouter_inventory_test.py` for a complete working example.

### Testing with Local Models (Logit-Based)

The built-in `InventoryTester` uses logit probabilities for more precise scoring:

```powershell
psyctl benchmark --model "gemma-3-270m-it" --inventory rses_10
```

This uses `_get_score_from_logits()` which extracts softmax probabilities over score tokens, producing continuous-valued scores.

---

## Checklist for Adding a New Inventory

- [ ] Research the inventory: item texts, domain structure, scoring rules, population norms
- [ ] Create items JSON file: `src/psyctl/data/inventories/{name}_{version}_items_en.json`
- [ ] Add config to `benchmark_config.json` under `inventories.{name}_{version}`
- [ ] Create inventory class: `src/psyctl/data/inventories/{name}.py`
  - [ ] Inherit from `BaseInventory`
  - [ ] Add `@register_inventory("{name}")` decorator
  - [ ] Implement `_load_config()`, `_load_questions()`
  - [ ] Implement `get_supported_traits()`, `_normalize_trait()`
  - [ ] Implement `calculate_scores()` with z-score and percentile calculation
- [ ] Add import to `__init__.py` and update `__all__`
- [ ] Run lint: `uv run ruff check src/psyctl/data/inventories/{name}.py`
- [ ] Run type check: `uv run pyright src/psyctl/data/inventories/{name}.py`
- [ ] Verify registration: `create_inventory("{name}_{version}")` works
- [ ] Run full test suite: `uv run pytest tests/ -v`

---

## Existing Inventories

| Registry Key | Class | Items | Domains | Measure |
|---|---|---|---|---|
| `ipip_neo_120` | `IPIPNEO` | 120 | N, E, O, A, C | Big Five Personality |
| `rei_40` | `REI` | 40 | RA, RE, EA, EE, R, E | Dual-Process Thinking |

## References

- Pacini, R., & Epstein, S. (1999). The relation of rational and experiential information processing styles. *Journal of Personality and Social Psychology, 76*, 972-987.
- Johnson, J. A. (2014). Measuring thirty facets of the five factor model with a 120-item public domain inventory. *Journal of Research in Personality, 51*, 78-89.
