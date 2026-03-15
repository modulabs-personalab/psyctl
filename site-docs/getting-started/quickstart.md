# Quick Start

This guide walks through the core PSYCTL workflow using the CLI.

For an interactive version, try the [01_quickstart notebook](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/01_quickstart.ipynb).

## 1. Generate a Steering Dataset

```bash
psyctl dataset.build.steer \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion" \
  --output "./dataset/steering" \
  --limit-samples 100
```

## 2. Extract a Steering Vector

```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/steering" \
  --output "./steering_vector/out.safetensors"
```

## 3. Apply Steering

```bash
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "Tell me about yourself"
```

## 4. Measure with Inventory

```bash
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Extraversion"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `dataset.build.steer` | Generate steering datasets |
| `dataset.upload` | Upload datasets to HuggingFace |
| `extract.steering` | Extract steering vectors |
| `steering` | Apply steering to generation |
| `benchmark inventory` | Test with psychological inventories |
| `benchmark llm-as-judge` | Test with LLM as Judge |
| `inventory.list` | List available inventories |

## Next Steps

- [Build Steering Dataset](../guides/dataset.md) — Detailed dataset generation guide
- [Extract Steering Vectors](../guides/extraction.md) — Extraction methods explained
- [Benchmark](../guides/benchmark.md) — Systematic evaluation
