# Installation

## Google Colab (Recommended for trying out)

No local setup required. Just open a notebook and run:

```python
!pip install -q git+https://github.com/modulabs-personalab/psyctl.git bitsandbytes accelerate
```

Set your HuggingFace token in [Colab Secrets](https://colab.research.google.com/notebooks/secrets.ipynb) (key icon in left sidebar):

- Key: `HF_TOKEN`
- Value: Your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Toggle "Notebook access" ON

## Local Installation

### With uv (recommended)

```bash
# Clone the repository
git clone https://github.com/modulabs-personalab/psyctl.git
cd psyctl

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
uv sync
```

### With pip

```bash
pip install git+https://github.com/modulabs-personalab/psyctl.git
```

### GPU Support (CUDA)

After installation, install CUDA-enabled PyTorch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

!!! note
    `uv sync` installs CPU-only PyTorch by default. Run the CUDA command above to enable GPU support.

## Environment Variables

**Required:**

```bash
export HF_TOKEN="your_huggingface_token"
```

**Optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PSYCTL_INFERENCE_BATCH_SIZE` | 16 | Batch size for inference |
| `PSYCTL_LOG_LEVEL` | INFO | Logging level |
| `PSYCTL_CACHE_DIR` | ./temp | HuggingFace cache location |
| `OPENROUTER_API_KEY` | - | For OpenRouter API integration |

See [Configuration](../reference/configuration.md) for the full list.
