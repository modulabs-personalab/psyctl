# Configuration Guide

This document describes all configuration options for PSYCTL using environment variables.

## Environment Variables

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | None | **Required** - Hugging Face API token for model access |

### Directory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PSYCTL_OUTPUT_DIR` | `./output` | General output directory |
| `PSYCTL_DATASET_DIR` | `./dataset` | Dataset storage directory |
| `PSYCTL_STEERING_VECTOR_DIR` | `./steering_vector` | Steering vector storage |
| `PSYCTL_RESULTS_DIR` | `./results` | Results and test output storage |
| `PSYCTL_CACHE_DIR` | `./temp` | Cache directory for HuggingFace models/datasets |

### Logging Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PSYCTL_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PSYCTL_LOG_FILE` | None | Optional log file path |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PSYCTL_INFERENCE_BATCH_SIZE` | `16` | Batch size for model inference |
| `PSYCTL_MAX_WORKERS` | `4` | Maximum number of worker threads |
| `PSYCTL_CHECKPOINT_INTERVAL` | `100` | Save checkpoint every N samples |

---

## Setting Environment Variables

### Windows (PowerShell)

#### Basic Setup
```powershell
# Required: HuggingFace token
$env:HF_TOKEN = "your_huggingface_token_here"

# Optional: Adjust logging
$env:PSYCTL_LOG_LEVEL = "DEBUG"
```

#### Custom Directories
```powershell
# Custom directory configuration
$env:PSYCTL_CACHE_DIR = "D:\ml_cache"
$env:PSYCTL_RESULTS_DIR = "C:\projects\results"
$env:PSYCTL_DATASET_DIR = "C:\datasets"
```

#### Performance Tuning
```powershell
# For high-end GPUs (24GB+ VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "32"
$env:PSYCTL_CHECKPOINT_INTERVAL = "50"

# For mid-range GPUs (8-16GB VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "16"
$env:PSYCTL_CHECKPOINT_INTERVAL = "100"

# For low-end GPUs (4-8GB VRAM)
$env:PSYCTL_INFERENCE_BATCH_SIZE = "8"
$env:PSYCTL_CHECKPOINT_INTERVAL = "200"
```

### Linux/macOS

#### Basic Setup
```bash
# Required: HuggingFace token
export HF_TOKEN="your_huggingface_token_here"

# Optional: Adjust logging
export PSYCTL_LOG_LEVEL="DEBUG"
```

#### Custom Directories
```bash
# Custom directory configuration
export PSYCTL_CACHE_DIR="/data/ml_cache"
export PSYCTL_RESULTS_DIR="/projects/results"
export PSYCTL_DATASET_DIR="/datasets"
```

#### Performance Tuning
```bash
# For high-end GPUs (24GB+ VRAM)
export PSYCTL_INFERENCE_BATCH_SIZE="32"
export PSYCTL_CHECKPOINT_INTERVAL="50"

# For mid-range GPUs (8-16GB VRAM)
export PSYCTL_INFERENCE_BATCH_SIZE="16"
export PSYCTL_CHECKPOINT_INTERVAL="100"

# For low-end GPUs (4-8GB VRAM)
export PSYCTL_INFERENCE_BATCH_SIZE="8"
export PSYCTL_CHECKPOINT_INTERVAL="200"
```

---

## Hugging Face Token Setup

Some models require a Hugging Face token for access.

### Get Your Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permission
3. Copy the token

### Set the Token

**Windows:**
```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
```

**Linux/macOS:**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

### Verify Setup

```bash
python -c "import os; print('HF_TOKEN:', 'Set' if os.getenv('HF_TOKEN') else 'Not Set')"
```

---

## Performance Optimization

### Batch Size Guidelines

Choose batch size based on your GPU memory:

| GPU Memory | Recommended Batch Size |
|------------|------------------------|
| 24GB+ | 32-64 |
| 16GB | 16-32 |
| 8GB | 8-16 |
| 4GB | 4-8 |
| CPU only | 2-4 |

### Checkpoint Interval

- **Faster checkpointing (50-100)**: Better for unstable connections, slightly slower
- **Slower checkpointing (200-500)**: Better for stable runs, slightly faster

### Tips

- Monitor GPU memory usage with `nvidia-smi`
- Start with smaller batch sizes and increase gradually
- Larger batch sizes improve throughput but require more VRAM
- Checkpoint intervals balance performance vs recovery time

---

## Directory Structure

All directories are automatically created when needed. Default structure:

```
project_root/
├── output/           # General outputs
├── dataset/          # Generated datasets
├── steering_vector/  # Extracted vectors
├── results/          # Test results
└── temp/            # HuggingFace cache
```

Customize any path using environment variables before running PSYCTL.

---

## Examples

### Minimal Setup (CPU)
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
# That's it! Use defaults for everything else
```

### Production Setup (GPU Server)
```bash
# Token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Performance
export PSYCTL_INFERENCE_BATCH_SIZE="32"
export PSYCTL_CHECKPOINT_INTERVAL="100"
export PSYCTL_MAX_WORKERS="8"

# Directories (on fast SSD)
export PSYCTL_CACHE_DIR="/nvme/ml_cache"
export PSYCTL_RESULTS_DIR="/nvme/results"

# Logging
export PSYCTL_LOG_LEVEL="INFO"
export PSYCTL_LOG_FILE="./psyctl.log"
```

### Development Setup
```bash
# Token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Debug logging
export PSYCTL_LOG_LEVEL="DEBUG"

# Small batches for testing
export PSYCTL_INFERENCE_BATCH_SIZE="4"
export PSYCTL_CHECKPOINT_INTERVAL="10"

# Local directories
export PSYCTL_CACHE_DIR="./dev_cache"
export PSYCTL_RESULTS_DIR="./dev_results"
```
