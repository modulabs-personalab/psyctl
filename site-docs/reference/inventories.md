# Supported Inventories

PSYCTL includes several validated psychological inventories for measuring personality traits.

## Available Inventories

| Inventory | Items | Domain | License |
|-----------|-------|--------|---------|
| `ipip_neo_120` | 120 | Big Five (N, E, O, A, C) | Public Domain |
| `ipip_neo_300` | 300 | Big Five (detailed facets) | Public Domain |
| `rei_40` | 40 | Rational-Experiential thinking styles | Research use |
| `sd4_28` | 28 | Short Dark Tetrad (Mach, Narc, Psycho, Sadism) | Research use |
| `vgq_14` | 14 | Victim Gaslighting Questionnaire | Research use |
| `indcol_32` | 32 | Individualism-Collectivism | Research use |

## Usage

### CLI

```bash
# List all inventories
psyctl inventory.list

# Run inventory test
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --inventory "ipip_neo_120"
```

### Python API

```python
from psyctl.data.inventories import create_inventory

inventory = create_inventory("ipip_neo_120")
questions = inventory.get_questions()
traits = inventory.get_supported_traits()
```

## Big Five Domains (IPIP-NEO)

| Code | Domain | Description |
|------|--------|-------------|
| N | Neuroticism | Emotional instability, anxiety, moodiness |
| E | Extraversion | Sociability, assertiveness, positive emotions |
| O | Openness | Curiosity, imagination, preference for variety |
| A | Agreeableness | Cooperation, trust, compassion |
| C | Conscientiousness | Discipline, organization, goal-directed behavior |

## Adding New Inventories

See [Adding Inventories](adding-inventories.md) for a step-by-step guide on contributing new psychological instruments.
