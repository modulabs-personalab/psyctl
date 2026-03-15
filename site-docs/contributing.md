# Contributing to PSYCTL

This guide covers how to extend PSYCTL with new features, particularly adding new steering vector extraction methods.

## Table of Contents

- [Development Setup](#development-setup)
- [Adding New Extraction Methods](#adding-new-extraction-methods)
- [Implementation Details](#implementation-details)
- [Testing Guidelines](#testing-guidelines)

## Development Setup

### Environment Setup

```powershell
# Development environment setup
& .\scripts\install-dev.ps1

# Virtual environment activation
& .\.venv\Scripts\Activate.ps1
```

### Development Workflow

```powershell
# Format code
& .\scripts\format.ps1

# Run tests with coverage
& .\scripts\test.ps1

# Complete build process (format + lint + test + install)
& .\scripts\build.ps1
```

## Adding New Extraction Methods

To implement a new steering vector extraction method, follow these steps:

### 1. Create Extractor Class

Create a new file in `src/psyctl/core/extractors/`:

```python
# src/psyctl/core/extractors/my_method_extractor.py

from typing import Dict
from pathlib import Path
import torch
from torch import nn
from transformers import AutoTokenizer

from psyctl.core.extractors.base import BaseVectorExtractor
from psyctl.core.logger import get_logger


class MyMethodExtractor(BaseVectorExtractor):
    """
    Extract steering vectors using My Custom Method.

    Description of your method and algorithm here.
    """

    def __init__(self):
        self.logger = get_logger("my_method_extractor")

    def extract(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        layers: list[str],
        dataset_path: Path,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract steering vectors from specified layers.

        Args:
            model: Loaded language model
            tokenizer: Model tokenizer
            layers: List of layer paths to extract from
            dataset_path: Path to dataset
            **kwargs: Method-specific parameters

        Returns:
            Dictionary mapping layer names to steering vectors
        """
        self.logger.info(f"Extracting with MyMethod from {len(layers)} layers")

        # Your extraction logic here
        steering_vectors = {}

        for layer_path in layers:
            # 1. Access the layer
            # 2. Collect activations
            # 3. Compute steering vector
            # 4. Store in dictionary
            pass

        return steering_vectors
```

### 2. Register Extractor

Update `src/psyctl/core/steering_extractor.py` to register your method:

```python
from psyctl.core.extractors.my_method_extractor import MyMethodExtractor

class SteeringExtractor:
    EXTRACTORS = {
        'mean_diff': MeanDifferenceActivationVectorExtractor,
        'bipo': BiPOVectorExtractor,
        'my_method': MyMethodExtractor,  # Add your extractor
    }

    def extract(self, method: str = 'mean_diff', **kwargs):
        extractor_class = self.EXTRACTORS.get(method)
        if extractor_class is None:
            raise ValueError(f"Unknown extraction method: {method}")

        extractor = extractor_class()
        return extractor.extract(**kwargs)
```

### 3. Update CLI

Add method selection to CLI command in `src/psyctl/commands/extract.py`:

```python
@click.command()
@click.option("--model", required=True)
@click.option("--layer", multiple=True)
@click.option("--dataset", required=True, type=click.Path())
@click.option("--output", required=True, type=click.Path())
@click.option("--method", default="mean_diff",
              help="Extraction method: mean_diff, bipo, my_method")
@click.option("--lr", type=float, default=5e-4, help="Learning rate for BiPO")
@click.option("--beta", type=float, default=0.1, help="Beta parameter for BiPO")
@click.option("--epochs", type=int, default=10, help="Number of epochs for BiPO")
def steering(model: str, layer: tuple, dataset: str, output: str, method: str,
             lr: float, beta: float, epochs: int):
    # ...
    method_params = {}
    if method == "bipo":
        method_params = {"lr": lr, "beta": beta, "epochs": epochs}

    extractor.extract(method=method, **method_params)
```

### 4. Add Tests

Create tests in `tests/core/extractors/test_my_method_extractor.py`:

```python
import pytest
from psyctl.core.extractors.my_method_extractor import MyMethodExtractor


def test_my_method_basic():
    extractor = MyMethodExtractor()
    # Test basic functionality
    pass


def test_my_method_multi_layer():
    extractor = MyMethodExtractor()
    # Test multi-layer extraction
    pass
```

### 5. Document Your Method

Add documentation to `docs/EXTRACT.STEERING.md` under the "Extraction Methods" section:

```markdown
### MyMethodName

Brief description of the method.

**Algorithm:**
1. Step 1
2. Step 2
3. Step 3

**Key Features:**
- Feature 1
- Feature 2

**When to use:**
- Use case 1
- Use case 2

**Parameters:**
- `param1`: Description
- `param2`: Description

**Example:**
```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors" \
  --method my_method
```
```

## Implementation Details

### Layer Access

The `LayerAccessor` class handles dynamic layer access:

```python
from psyctl.core.layer_accessor import LayerAccessor

accessor = LayerAccessor()
layer_module = accessor.get_layer(model, "model.layers[13].mlp.down_proj")
```

**Layer Path Format:**

Layer paths use dot notation with bracket indexing:
- `model.layers[13].mlp.down_proj` - MLP output projection (recommended)
- `model.layers[0].self_attn.o_proj` - Attention output projection
- `model.language_model.layers[10].mlp.act_fn` - After activation function

### Activation Collection

The `ActivationHookManager` manages forward hooks for collecting activations:

```python
from psyctl.core.hook_manager import ActivationHookManager

hook_manager = ActivationHookManager()
layer_modules = {"layer_13": model.model.layers[13].mlp.down_proj}
hook_manager.register_hooks(layer_modules)

# Run inference
with torch.inference_mode():
    outputs = model(**inputs)

# Get collected activations
activations = hook_manager.get_mean_activations()
hook_manager.remove_all_hooks()
```

### Dataset Format

steering datasets are JSONL files with this structure:

```json
{
  "question": "[Situation]\n...\n[Question]\n...\n1. Answer option 1\n2. Answer option 2\n[Answer]",
  "positive": "(1",
  "neutral": "(2",
  "positive_text": "Full text of personality answer...",
  "neutral_text": "Full text of neutral answer..."
}
```

**Version 2+ datasets** include `positive_text` and `neutral_text` fields for full answer content.

The loader automatically combines `question` with `positive`/`neutral` to create full prompts.

### Output Format

Steering vectors should be saved in safetensors format with embedded metadata:

```python
# File structure
{
    "model.layers[13].mlp.down_proj": torch.Tensor,  # First layer's steering vector
    "model.layers[14].mlp.down_proj": torch.Tensor,  # Second layer's steering vector
    # ... more layers
    "__metadata__": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "method": "mean_diff",  # or "bipo"
        "layers": ["model.layers[13].mlp.down_proj", "model.layers[14].mlp.down_proj"],
        "dataset_path": "./dataset/caa",
        "dataset_samples": 20000,
        "num_layers": 2,
        "normalized": false
    }
}
```

**Loading vectors:**

```python
from safetensors.torch import load_file

data = load_file("steering_vector.safetensors")
layer_13_vector = data["model.layers[13].mlp.down_proj"]
metadata = data["__metadata__"]
```

## Testing Guidelines

### Local Testing Standards

- Always use "gemma-3-270m-it" model for local testing
- Output files go to ./results folder
- Hugging Face cache goes to ./temp folder
- No emoji usage in code or output

### Test Coverage

- Unit tests for all public methods
- Integration tests for end-to-end workflows
- Mock-based testing for external dependencies (HuggingFace models)
- Coverage target: >80%

### Code Quality

- Google-style docstrings for all public functions
- Type hints required
- Snake_case for functions/variables, PascalCase for classes
- Import organization with isort
- Code formatting: Black + isort
- Linting: flake8 + mypy

## Git Workflow

- Feature branches: `feature/issue-number-description`
- Bug fixes: `fix/issue-number-description`
- Semantic versioning (MAJOR.MINOR.PATCH)

## Common Implementation Patterns

### Memory-Efficient Activation Collection

Use incremental mean computation for large datasets:

```python
mean_activation = None
count = 0

for batch in dataset:
    activations = get_activations(batch)
    if mean_activation is None:
        mean_activation = torch.zeros_like(activations[0])

    for act in activations:
        count += 1
        mean_activation += (act - mean_activation) / count
```

### Batch Processing

Process data in batches for GPU efficiency:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for batch in dataloader:
    with torch.inference_mode():
        outputs = model(**batch)
        # Collect activations
```

### Checkpoint Support

Save intermediate results for long-running operations:

```python
if checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
    checkpoint_path = output_path.with_suffix('.checkpoint')
    save_file(intermediate_vectors, checkpoint_path)
    self.logger.info(f"Checkpoint saved: {checkpoint_path}")
```

## References

- [PyTorch Hooks Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Safetensors Format](https://github.com/huggingface/safetensors)
- [CAA Paper](https://arxiv.org/abs/2312.06681)
- [BiPO Paper](https://arxiv.org/abs/2406.00045)
