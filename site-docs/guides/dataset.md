# Build Steering Dataset

This document describes how to build steering datasets using the `psyctl dataset.build.steer` command. These datasets are compatible with multiple steering vector extraction methods including mean_diff (Mean Difference) and BiPO (Bi-Directional Preference Optimization).

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [OpenRouter Integration](#openrouter-integration)
- [Dataset Source](#dataset-source)
- [Output Format](#output-format)
- [Performance Optimization](#performance-optimization)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Adding Custom Datasets](#adding-custom-datasets)

## Overview

Steering datasets contain paired prompts designed to elicit contrasting personality-driven responses from language models. These datasets serve as training data for extracting steering vectors that can modify model behavior.

The dataset stores raw components (situation, character name, positive/neutral responses) in a clean format, allowing different extraction methods (mean_diff, BiPO, etc.) to build prompts as needed at training time.

The dataset building process involves:

1. Loading a base conversational dataset (e.g., SODA)
2. Generating personality-specific character descriptions using P2 (Personality Prompt)
3. Creating positive/neutral response pairs for each scenario
4. Saving raw components in JSONL format for steering vector extraction

## Usage

### Basic Command

Generate a steering dataset for a specific personality trait:

```bash
psyctl dataset.build.steer \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/extroversion"
```

### With Custom Dataset

Use a different Hugging Face dataset as the base:

```bash
psyctl dataset.build.steer \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Machiavellianism" \
  --dataset "username/custom-conversations" \
  --output "./dataset/machiavellianism"
```

### Limit Sample Count

Generate a specific number of samples for testing:

```bash
psyctl dataset.build.steer \
  --model "google/gemma-3-270m-it" \
  --personality "Extroversion" \
  --output "./dataset/test" \
  --limit-samples 100
```

### Multiple Personalities

Generate datasets for multiple personality traits:

```bash
psyctl dataset.build.steer \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellianism" \
  --output "./dataset/multi"
```

## OpenRouter Integration

PSYCTL supports OpenRouter API for dataset generation without local GPU requirements.

**Basic usage:**
```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --personality "Extroversion" \
  --output "./dataset/openrouter"
```

For detailed documentation, parallel processing, and Python API examples, see **[OpenRouter Guide](./OPENROUTER.md)**.

## Dataset Source

### Default Dataset: SODA

By default, the command uses the [SODA dataset](https://huggingface.co/datasets/allenai/soda) (Social Dialogue dataset) which contains:

- Over 1.5M dialogue turns
- Diverse conversational scenarios
- Natural social interaction patterns
- High-quality human annotations

### Using Custom Datasets

You can use any Hugging Face dataset that provides conversational contexts:

**Requirements:**
- Must be accessible via Hugging Face Datasets library
- Should contain dialogue or scenario information
- Recommended: Conversational or social interaction data

**Example custom datasets:**
```bash
# Using DailyDialog
psyctl dataset.build.steer \
  --dataset "daily_dialog" \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/daily"

# Using custom dataset
psyctl dataset.build.steer \
  --dataset "myusername/my-conversations" \
  --model "google/gemma-2-2b-it" \
  --personality "Introversion" \
  --output "./dataset/custom"
```

## Output Format

### Directory Structure

```
output_directory/
├── caa_dataset_20250107_143022.jsonl            # Main dataset file (timestamped)
└── caa_dataset_20250107_143022.checkpoint.json  # Checkpoint file
```

### JSONL Format

Each line in the dataset file contains:

```json
{
  "situation": "Alice is at a party.\nBob: Hi, how are you?",
  "char_name": "Alice",
  "positive": "I'm so excited to be here! Want to dance?",
  "neutral": "I'm fine, thanks. Just looking around."
}
```

**Fields:**
- `situation`: The conversational scenario and context
- `char_name`: Character name for the response
- `positive`: Full text of the positive personality answer
- `neutral`: Full text of the neutral personality answer

### Checkpoint Format

The checkpoint file contains:

```json
{
  "num_generated": 500,
  "output_file": "c:/work/psyctl/dataset/caa_dataset_20250107_143022.jsonl",
  "timestamp": "2025-01-07T14:35:22.123456"
}
```

## Performance Optimization

### Batch Processing

The dataset builder uses batch processing (default: 16). Adjust based on GPU memory:

```bash
export PSYCTL_INFERENCE_BATCH_SIZE="32"  # Linux/macOS
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"  # Windows
```

**Recommended batch sizes:**
- High-end GPU (24GB+): 32-64
- Mid-range GPU (8-16GB): 16-32
- Low-end GPU (4-8GB): 8-16

### Checkpoint and Resume

Checkpoints are saved every 100 samples by default. If interrupted, simply re-run the same command to resume automatically.

**Configure checkpoint interval:**
```bash
export PSYCTL_CHECKPOINT_INTERVAL="50"   # Save more frequently
$env:PSYCTL_CHECKPOINT_INTERVAL = "200"  # Save less frequently (Windows)
```

Checkpoint files (`*.checkpoint.json`) are stored alongside the dataset JSONL file.

## Uploading to HuggingFace Hub

Upload generated datasets to HuggingFace Hub with automatic dataset card generation.

### Prerequisites

Set your HuggingFace token ([get one here](https://huggingface.co/settings/tokens)):
```bash
export HF_TOKEN="hf_xxxxxxxxxxxx"  # Linux/macOS
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"  # Windows
```

### CLI Usage

```bash
psyctl dataset.upload \
  --dataset-file "./results/caa_dataset_*.jsonl" \
  --repo-id "username/extroversion-dataset" \
  --license "mit"
```

**Options:**
- `--dataset-file`: Path to JSONL file (required)
- `--repo-id`: HuggingFace repo `username/name` (required)
- `--private`: Make repository private
- `--license`: License (e.g., 'mit', 'apache-2.0', 'cc-by-4.0')

**Features:**
- Automatic dataset card generation with PSYCTL branding
- Metadata tracking (personality, model, sample count, source dataset)
- Usage instructions and paper references included

### Share with Community

Share your dataset by adding it to [COMMUNITY.DATASETS.md](./COMMUNITY.DATASETS.md) via pull request.

## Adding Custom Datasets

To use a custom Hugging Face dataset as the source:

### 1. Upload Dataset to Hugging Face

```python
from datasets import Dataset
import pandas as pd

# Create your conversational dataset
data = {
    'dialogue': ['conversation 1...', 'conversation 2...'],
    'narrative': ['narrative 1...', 'narrative 2...'],
    # ... other fields
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Upload to Hugging Face
dataset.push_to_hub("username/my-conversations")
```

### 2. Use in Dataset Building

```bash
psyctl dataset.build.steer \
  --dataset "username/my-conversations" \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/custom"
```

### 3. Dataset Requirements

**Minimum requirements:**
- Accessible via Hugging Face Datasets
- Contains conversational or scenario data
- Sufficient samples for meaningful extraction

### 4. Testing with Small Samples

Test your custom dataset with limited samples first:

```bash
psyctl dataset.build.steer \
  --dataset "username/my-conversations" \
  --model "google/gemma-3-270m-it" \
  --personality "Extroversion" \
  --output "./dataset/test" \
  --limit-samples 10
```

## Community Datasets

Pre-built steering datasets and source conversation datasets are available from the community. See **[Community Datasets Registry](./COMMUNITY.DATASETS.md)** for the complete list.

## Implementation Details

The dataset builder uses P2 (Personality Prompt) to generate personality-specific character descriptions and creates contrastive positive/neutral response pairs. Data is processed in batches with automatic checkpointing.

### Custom Templates

PSYCTL supports custom Jinja2 templates for roleplay prompts to support different languages or prompt styles.

**Usage:**
```bash
psyctl dataset.build.steer \
  --roleplay-prompt-template "./templates/roleplay.j2" \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/custom"
```

Default template is `src/psyctl/templates/roleplay_prompt.j2`. See the template file for available variables (char_name, user_name, p2, situation).


## References

- [SODA Dataset](https://huggingface.co/datasets/allenai/soda)
- [CAA Paper: Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [P2 Paper: Evaluating and Inducing Personality](https://arxiv.org/abs/2206.07550)
- [PSYCTL Steering Extraction](./EXTRACT.STEERING.md)
