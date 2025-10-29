# Extract Steering Vectors

This document describes how to extract steering vectors from language models using the `psyctl extract.steering` command.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Extraction Methods](#extraction-methods)
- [Multi-Layer Extraction](#multi-layer-extraction)
- [Output Format](#output-format)
- [Adding New Extraction Methods](#adding-new-extraction-methods)

## Overview

Steering vectors are learned representations that can modify language model behavior to exhibit specific personality traits or characteristics. The extraction process involves:

1. Loading a steering dataset with positive/neutral prompt pairs
2. Running inference on the model to collect internal activations
3. Computing steering vectors from the activation differences
4. Saving vectors in safetensors format for later use

## Usage

### CLI Usage

#### Basic Single-Layer Extraction

Extract a steering vector from a single model layer:

```bash
psyctl extract.steering \
  --model "google/gemma-2-2b-it" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors"
```

#### Multi-Layer Extraction

Extract steering vectors from multiple layers simultaneously:

```bash
# Using repeated --layer flags
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --layer "model.layers[14].mlp.down_proj" \
  --layer "model.layers[15].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/multi_layer.safetensors"

# Or using comma-separated values
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj,model.layers[15].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/multi_layer.safetensors"
```

#### Command-Line Options

- `--model`: Hugging Face model identifier (required)
- `--layer`: Single layer path (can be repeated for multi-layer extraction)
- `--layers`: Comma-separated list of layer paths
- `--dataset`: Path to steering dataset directory containing JSONL file (required)
- `--output`: Output path for safetensors file (required)
- `--batch-size`: Batch size for inference (default: from config)
- `--normalize`: Normalize steering vectors to unit length (optional)

### Python Code Usage

You can use the `SteeringExtractor` class directly in Python code with flexible input options.

#### Basic Example (Using model_name and dataset_path)

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

# Initialize extractor
extractor = SteeringExtractor()

# Extract steering vector using CAA method
vectors = extractor.extract_steering_vector(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp.down_proj"],
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/bipo_steering.safetensors"),
    normalize=False,
    method="mean_diff"
)

# vectors is a dict: {"model.layers.13.mlp.down_proj": torch.Tensor}
print(f"Extracted {len(vectors)} vectors")
for layer_name, vector in vectors.items():
    print(f"  {layer_name}: shape={vector.shape}, norm={vector.norm().item():.4f}")
```

#### Using Pre-loaded Model

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from psyctl.core.steering_extractor import SteeringExtractor

# Load model and tokenizer manually
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Initialize extractor
extractor = SteeringExtractor()

# Extract using pre-loaded model
vectors = extractor.extract_steering_vector(
    model=model,
    tokenizer=tokenizer,
    layers=["model.layers.13.mlp.down_proj"],
    dataset_path=Path("./results/caa_dataset.jsonl"),
    output_path=Path("./results/steering.safetensors"),
    method="mean_diff"
)
```

#### Using Pre-loaded Dataset

```python
from pathlib import Path
from datasets import load_dataset
from psyctl.core.steering_extractor import SteeringExtractor

# Load dataset from HuggingFace
hf_dataset = load_dataset("CaveduckAI/steer-personality-rudeness-ko", split="train")

# Convert to required format
dataset = [
    {
        "situation": item["situation"],
        "char_name: item["char_name"],
        "positive": item["positive"],
        "neutral": item["neutral"]
    }
    for item in hf_dataset
]

# Initialize extractor
extractor = SteeringExtractor()

# Extract using pre-loaded dataset
vectors = extractor.extract_steering_vector(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp.down_proj"],
    dataset=dataset,
    output_path=Path("./results/steering.safetensors"),
    method="mean_diff"
)
```

#### Using Both Pre-loaded Model and Dataset

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from psyctl.core.steering_extractor import SteeringExtractor

# Load model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Load and prepare dataset
hf_dataset = load_dataset("CaveduckAI/steer-personality-rudeness-ko", split="train")
dataset = [
    {
        "situation": item["situation"],
        "char_name": item["char_name"],
        "positive": item["positive"],
        "neutral": item["neutral"]
    }
    for item in hf_dataset
]

# Initialize extractor
extractor = SteeringExtractor()

# Extract using both pre-loaded model and dataset
vectors = extractor.extract_steering_vector(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    layers=["model.layers.13.mlp.down_proj"],
    output_path=Path("./results/steering.safetensors"),
    method="mean_diff"
)
```

#### BiPO Method Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

extractor = SteeringExtractor()

# Extract using BiPO optimization method
vectors = extractor.extract_steering_vector(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp"],  # BiPO uses layer modules, not projections
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/bipo_steering.safetensors"),
    batch_size=16,
    normalize=False,
    method="bipo",
    lr=5e-4,
    beta=0.1,
    epochs=10
)
```

#### Multi-Layer Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

extractor = SteeringExtractor()

# Extract from multiple layers simultaneously
layers = [
    "model.layers.10.mlp.down_proj",
    "model.layers.12.mlp.down_proj",
    "model.layers.14.mlp.down_proj",
]

vectors = extractor.extract_steering_vector(
    model_name="google/gemma-2-2b-it",
    layers=layers,
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/multi_layer_steering.safetensors"),
    batch_size=16,
    normalize=True,
    method="mean_diff"
)

# Analyze extracted vectors
for layer_name, vector in vectors.items():
    norm = vector.norm().item()
    mean = vector.mean().item()
    std = vector.std().item()
    print(f"{layer_name}:")
    print(f"  Shape: {vector.shape}")
    print(f"  Norm: {norm:.4f}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
```

#### Loading and Using Extracted Vectors

```python
from safetensors.torch import load_file
import json

# Load saved vectors
data = load_file("./results/bipo_steering.safetensors")

# Access vectors
for key, tensor in data.items():
    if key != "__metadata__":
        print(f"Layer: {key}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")

# Load metadata
metadata_json = data.get("__metadata__", {})
if isinstance(metadata_json, bytes):
    metadata = json.loads(metadata_json.decode('utf-8'))
elif isinstance(metadata_json, str):
    metadata = json.loads(metadata_json)
else:
    metadata = metadata_json

print(f"\nMetadata:")
print(f"  Model: {metadata.get('model')}")
print(f"  Method: {metadata.get('method')}")
print(f"  Layers: {metadata.get('num_layers')}")
print(f"  Normalized: {metadata.get('normalized')}")
print(f"  Dataset samples: {metadata.get('dataset_samples')}")
```

#### Complete Workflow Example

```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor
from safetensors.torch import load_file

# 1. Extract steering vector
extractor = SteeringExtractor()

vectors = extractor.extract_steering_vector(
    model_name="google/gemma-2-2b-it",
    layers=["model.layers.13.mlp.down_proj"],
    dataset_path=Path("./results/caa_dataset_20251007_160523.jsonl"),
    output_path=Path("./results/extroversion_steering.safetensors"),
    batch_size=16,
    normalize=False,
    method="mean_diff"
)

# 2. Verify extraction
layer_name = "model.layers.13.mlp.down_proj"
vector = vectors[layer_name]
print(f"Extracted vector: shape={vector.shape}, norm={vector.norm().item():.4f}")

# 3. Load for later use
loaded_data = load_file("./results/extroversion_steering.safetensors")
loaded_vector = loaded_data[layer_name]
print(f"Loaded vector: shape={loaded_vector.shape}")

# 4. Apply to model (see STEERING.md for details)
# ...
```

### Layer Path Format

Layer paths use dot notation with bracket indexing:

```
model.layers[13].mlp.down_proj
model.layers[0].self_attn.o_proj
model.language_model.layers[10].mlp.act_fn
```

Common layer targets:
- `mlp.down_proj`: MLP output projection (recommended)
- `mlp.act_fn`: After activation function
- `self_attn.o_proj`: Attention output projection

## Extraction Methods

### Mean Difference (mean_diff)

The mean_diff method computes steering vectors as the mean difference between positive and neutral activations.
This implements the Mean Difference (MD) algorithm from the CAA paper:

**Algorithm:**
1. Load steering dataset containing positive/neutral prompt pairs
2. For each layer:
   - Collect activations from positive prompts
   - Collect activations from neutral prompts
   - Compute incremental means (memory efficient)
3. Calculate: `steering_vector = mean(positive_activations) - mean(neutral_activations)`
4. Optionally normalize to unit length

**Example:**
```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/steering" \
  --output "./steering_vector/out.safetensors"
```

Note: `mean_diff` is the default method, so `--method mean_diff` can be omitted.

### BiPO (Bi-Directional Preference Optimization)

BiPO is an optimization-based method that learns steering vectors through preference learning:

**Algorithm:**
1. Load steering dataset containing positive/neutral prompt pairs
2. Initialize learnable steering parameters for each layer
3. For each training epoch:
   - Apply steering to model activations
   - Compute preference loss between positive/neutral outputs
   - Update steering parameters via gradient descent
4. Extract final optimized steering vectors
5. Optionally normalize to unit length

**Hyperparameters:**
- `--lr`: Learning rate (default: 5e-4)
- `--beta`: Beta parameter for preference loss (default: 0.1)
- `--epochs`: Number of training epochs (default: 10)

**Example:**
```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp" \
  --dataset "./dataset/caa" \
  --output "./steering_vector/out.safetensors" \
  --method bipo \
  --lr 5e-4 \
  --beta 0.1 \
  --epochs 10
```

**Note:** BiPO requires layer modules (e.g., `model.layers[13].mlp`) rather than specific projections (e.g., `model.layers[13].mlp.down_proj`).

#### BiPO Dataset Format

PSYCTL uses a clean dataset format that stores raw components (situation, character name, and full answer texts). BiPO uses this format for preference learning aligned with the original paper:

**Dataset Format:**
```json
{
  "situation": "Alice is at a party.\nBob: Hi, how are you?",
  "char_name": "Alice",
  "positive": "I'm so excited to be here! Want to dance?",
  "neutral": "I'm fine, thanks. Just looking around."
}
```

**BiPO Training:**
```bash
# Generate dataset
psyctl dataset.build.steer \
  --model "google/gemma-2-2b-it" \
  --personality "Extroversion" \
  --output "./dataset/ext"

# Extract with BiPO (automatically uses full answer texts)
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp" \
  --dataset "./dataset/ext" \
  --output "./vector.safetensors" \
  --method bipo \
  --lr 5e-4 \
  --beta 0.1 \
  --epochs 10
```

**Python API:**
```python
from pathlib import Path
from psyctl.core.steering_extractor import SteeringExtractor

extractor = SteeringExtractor()

# BiPO extraction
vectors = extractor.extract_steering_vector(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    layers=["model.layers[13].mlp"],
    dataset_path=Path("./dataset/ext"),
    output_path=Path("./vector.safetensors"),
    method="bipo",
    lr=5e-4,
    beta=0.1,
    epochs=10
)
```

**How It Works:**
1. Dataset stores raw answer texts (not pre-formatted prompts)
2. BiPO builds prompts at training time: `[situation]...[char_name]...[positive]...[neutral]...{full_text}`
3. Evaluates `P(positive_answer)` vs `P(neutral_answer)`
4. Optimizes steering vector to increase relative preference for positive answers

**Benefits:**
- ✅ **Paper alignment**: Evaluates full answer texts as described in BiPO paper
- ✅ **Clean data**: No template-generated text stored in dataset
- ✅ **Smaller files**: ~40% size reduction compared to old versions
- ✅ **Flexibility**: Prompts can be customized at inference time

### Method Comparison

| Feature | Mean Diff (mean_diff) | BiPO |
|---------|------------------------|------|
| Speed | Fast | Slower (optimization) |
| Resource Usage | Low | Higher (training) |
| Hyperparameters | None | lr, beta, epochs |
| Use Case | Quick steering | High-quality steering |

## Multi-Layer Extraction

Extracting from multiple layers simultaneously offers several advantages:

**Benefits:**
1. **Efficiency**: Single forward pass collects activations from all layers
2. **Consistency**: All vectors extracted from same dataset samples
3. **Experimentation**: Compare steering strength across layers
4. **Ensemble**: Combine vectors from multiple layers during application

**Best Practices:**
- Test layers in middle-to-late transformer blocks (e.g., layers 10-20 for 24-layer models)
- Focus on MLP output projections (`mlp.down_proj`)
- Extract 3-5 consecutive layers for comparison
- Use visualization tools to analyze vector magnitudes across layers

## Output Format

Steering vectors are saved in safetensors format with embedded metadata:

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

## Adding New Extraction Methods

For developers who want to implement custom extraction methods, see the [Contributing Guide](./CONTRIBUTING.md#adding-new-extraction-methods) for detailed implementation instructions.

## Troubleshooting

### Layer Not Found

```
Error: Layer 'model.layers[50].mlp.down_proj' not found in model
```

**Solution:** Check model architecture and available layers:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name")
print(model)  # Inspect structure
```

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution:** Reduce batch size:

```bash
psyctl extract.steering ... --batch-size 8
```

Or set environment variable:

```bash
export PSYCTL_INFERENCE_BATCH_SIZE=8
```

### Token Position Issues

If activations seem incorrect, verify token position detection for your model architecture. Check logs for detected position or implement custom detection logic.

### Padding Handling

**Technical details**: When you see this log message during extraction:
```
Tokenizer uses LEFT padding
  Position -1 always points to last real token (safe)
```
or
```
Tokenizer uses RIGHT padding
  This project uses attention masks to handle this correctly.
  Activation extraction will find the last real token automatically.
```

This confirms the system detected your tokenizer's padding configuration and will handle it appropriately.

**If you see warnings**: The warning "No attention mask set" should never appear during normal usage. If you see it, please report an issue.

For more details, see [Troubleshooting Guide](./TROUBLESHOOTING.md#padding-related-issues).

## References

- [CAA Paper: Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- [BiPO Paper: Bi-Directional Preference Optimization](https://arxiv.org/abs/2406.00045)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [PSYCTL Dataset Building](./DATASET.BUILD.STEER.md)
- [PSYCTL Steering Application](./STEERING.md)
