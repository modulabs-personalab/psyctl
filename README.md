<p align="center">
  <img src="./docs/images/logo.png" alt="PSYCTL Logo" width="120"/>
</p>

# PSYCTL - LLM Personality Steering Tool

> **⚠️ Project Under Development**
> This project is currently under development and only supports limited functionality. Please check the release notes for stable features.

A project by [Persona Lab](https://modulabs.co.kr/labs/337) at ModuLabs.

A tool that supports steering LLMs to exhibit specific personalities. The goal is to automatically generate datasets and work with just a model and personality specification.



---

## 📚 Documentation

### Core Guides

- **[Build Steering Dataset](./docs/DATASET.BUILD.STEER.md)** - Generate steering datasets for vector extraction
- **[Extract Steering Vectors](./docs/EXTRACT.STEERING.md)** - Extract steering vectors using mean_diff or BiPO methods
- **[Steering Experiment](./docs/STEERING.md)** - Apply steering vectors using CAA (Contrastive Activation Addition)

### Additional Resources

- **[Configuration](./docs/CONFIGURATION.md)** - Environment variables and performance tuning
- **[OpenRouter Integration](./docs/OPENROUTER.md)** - Use cloud APIs instead of local GPU
- **[Community Datasets](./docs/COMMUNITY.DATASETS.md)** - Pre-built datasets and registry
- **[Troubleshooting](./docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](./docs/CONTRIBUTING.md)** - Development guide and contribution guidelines

---

## 📖 User Guide

### 🚀 Quick Start

#### Installation

**Basic Installation (CPU Version)**
```bash
# Install uv (Windows)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# Project setup
uv venv
& .\.venv\Scripts\Activate.ps1
uv sync
```

**Installation in Google Colab**
```python
# Install directly from GitHub
!pip install git+https://github.com/modulabs-personalab/psyctl.git

# Or install from specific branch
!pip install git+https://github.com/modulabs-personalab/psyctl.git@main

# Set environment variables
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
os.environ['PSYCTL_LOG_LEVEL'] = 'INFO'

# Usage example
from psyctl import DatasetBuilder, P2, LLMLoader
```

**GPU Acceleration Installation (CUDA Support)**
```bash
# Install CUDA-enabled PyTorch after basic installation
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **Important**: The `transformers` package has `torch` as a dependency, so running `uv sync` will automatically install the CPU version. For GPU usage, you need to run the CUDA installation command above again.

#### Basic Usage

```bash
# 1. Generate dataset
psyctl dataset.build.steer \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/steering"

# 2. Upload dataset to HuggingFace Hub (optional)
psyctl dataset.upload \
  --dataset-file "./dataset/steering/steering_dataset_*.jsonl" \
  --repo-id "username/extroversion-steering"

# 3. Extract steering vector
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/steering" \
  --output "./steering_vector/out.safetensors"

# 4. Steering experiment
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "Tell me about yourself"

# 5. Inventory test
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Neuroticism"
```

### 📋 Commands Overview

PSYCTL provides 5 main commands. See documentation links above for detailed usage.

| Command | Description | Documentation |
|---------|-------------|---------------|
| `dataset.build.steer` | Generate steering datasets | [Guide](./docs/DATASET.BUILD.STEER.md) |
| `dataset.upload` | Upload datasets to HuggingFace | [Guide](./docs/DATASET.BUILD.STEER.md#uploading-to-huggingface-hub) |
| `extract.steering` | Extract steering vectors | [Guide](./docs/EXTRACT.STEERING.md) |
| `steering` | Apply steering to generation | [Guide](./docs/STEERING.md) |
| `benchmark inventory` | Test with psychological inventories (logit-based) | See below |
| `benchmark llm-as-judge` | Test with LLM as Judge (situation-based questions) | See below |
| `inventory.list` | List available inventories | See below |

**Benchmark Methods:**
- **Inventory**: Uses standardized psychological inventories (e.g., IPIP-NEO) with logit-based scoring. More objective and reproducible.
- **LLM as Judge**: Generates situation-based questions and uses an LLM to evaluate responses. More flexible and context-aware.
  - For API-based judges (OpenAI, OpenRouter), set environment variables:
    - `OPENAI_API_KEY` for OpenAI models
    - `OPENROUTER_API_KEY` for OpenRouter models
  - For local models, use `local-default` (reuses target model) or configure custom model path in `benchmark_config.json`
  - For custom API servers, edit `benchmark_config.json` to add your server configuration

### 📊 Supported Inventories

| Inventory | Domain | License | Notes |
|-----------|--------|---------|-------|
| IPIP-NEO-300/120 | Big Five | Public Domain | Full & short forms |
| NPI-40 | Narcissism | Free research use | Forced-choice |
| PNI-52 | Pathological narcissism | CC-BY-SA | Likert 1–6 |
| NARQ-18 | Admiration & Rivalry | CC-BY-NC | Two sub-scales |
| MACH-IV | Machiavellianism | Public Domain | Likert 1–5 |
| LSRP-26 | Psychopathy | Public Domain | Primary & secondary |
| PPI-56 | Psychopathy | Free research use | Short form |

### ⚙️ Configuration

PSYCTL uses environment variables for configuration. **Required:**

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token_here"  # Linux/macOS
$env:HF_TOKEN = "your_token_here"              # Windows
```

For detailed configuration options (directories, performance tuning, logging), see [Configuration Guide](./docs/CONFIGURATION.md).

### 📝 Complete Workflow Example

```bash
# 1. Generate dataset for extroversion personality
# Set batch size for optimal performance
export PSYCTL_INFERENCE_BATCH_SIZE="16"

psyctl dataset.build.steer \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Extroversion" \
  --output "./dataset/extroversion" \
  --limit-samples 1000

# 2. Extract steering vector
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/extroversion" \
  --output "./steering_vector/extroversion.safetensors"

# 3. Apply steering to generate text
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --input-text "Tell me about yourself"

# 4. Measure personality changes with inventory
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Extraversion"

# 5. Measure personality changes with LLM as Judge
# Note: For API-based judges, set environment variables:
#   export OPENAI_API_KEY="your-key"        # For OpenAI models
#   export OPENROUTER_API_KEY="your-key"    # For OpenRouter models
# Or configure custom API server in benchmark_config.json
psyctl benchmark llm-as-judge \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --trait "Extraversion" \
  --judge-model "local-default" \
  --num-questions 10 \
  --strengths "1.0,2.0,3.0"
```

**More Examples:**
- See [examples/](examples/) directory for Python library usage
- Check documentation links above for detailed guides

---

## 🤝 Contributing

Contributions are welcome! See [Contributing Guide](./docs/CONTRIBUTING.md) for:
- Development environment setup
- Code style and standards
- Testing guidelines
- Pull request process

## Key papers
- [Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/pdf/2312.06681)
- [Steering Large Language Model Activations in Sparse Spaces](https://arxiv.org/pdf/2503.00177)
- [Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering](https://arxiv.org/pdf/2412.10427v1)
- [Toy model of superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Personalized Steering of LLMs: Versatile Steering Vectors via Bi-directional Preference Optimization](https://arxiv.org/abs/2406.00045)
- [The dark core of personality](https://psycnet.apa.org/record/2018-32574-001)
- [The Dark Triad of personality: Narcissism, Machiavellianism, and psychopathy. Journal of Research in Personality](https://www.sciencedirect.com/science/article/pii/S0092656602005056)
- [Style-Specific Neurons for Steering LLMs in Text Style Transfer](https://arxiv.org/abs/2410.00593)
- [Between facets and domains: 10 aspects of the Big Five. Journal of Personality and Social Psychology](https://psycnet.apa.org/fulltext/2007-15390-012.html)

---

## Sponsors

<p align="left">
  <a href="https://caveduck.io" target="_blank">
    <img src="https://cdn.caveduck.io/public/assets/logo_white.c2efa9b1d010.svg" alt="Caveduck.io" width="200"/>
  </a>
  <a href="https://modulabs.co.kr/" target="_blank" style="margin-left:20px">
    <img src="https://i.namu.wiki/i/OqOsAbn-fFDwzMVNG3EsqTEMZz13k7wmahhZwkayuE1q9WWWLtMMmaqCM9FJULqHB1CjRGHiQzVoOHkQFQkbHBbNlF9CfB1_yEMVgZWOHnbyHdB-akqxFIOjQ9WXGM7RJF4H0JUGGzxDQxIGXeGtTg.webp" alt="Caveduck.io" width="200"/>
  </a>
</p>

