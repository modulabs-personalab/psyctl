# Steering Experiment

This document describes how to apply steering vectors to language models for text generation using the `psyctl steering` command.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Steering Parameters](#steering-parameters)
- [Steering Methods](#steering-methods)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)

## Overview

The steering experiment applies pre-extracted steering vectors to language models during text generation. This influences the model's personality or behavior according to the training data used during vector extraction.

The steering process involves:

1. Loading a model and its tokenizer
2. Loading steering vectors from a safetensors file
3. Registering forward hooks on target layers
4. Applying steering vectors during text generation
5. Decoding and returning the steered output

## Usage

### CLI Usage

#### Basic Command

Apply a steering vector to generate text:

```bash
psyctl steering \
  --model "google/gemma-2-2b-it" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "Tell me about yourself"
```

#### With Custom Strength

Adjust the steering strength multiplier:

```bash
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello world" \
  --strength 1.5
```

#### Using Orthogonalized Addition

Apply steering with the orthogonalized addition method:

```bash
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "hello" \
  --orthogonal \
  --strength 2.0
```

#### Command-Line Options

- `--model`: Model name or HuggingFace identifier (required)
- `--steering-vector`: Path to steering vector file (.safetensors) (required)
- `--input-text`: Input text for generation (required)
- `--strength`: Steering strength multiplier (default: 1.0)
- `--max-tokens`: Maximum number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature, 0 for greedy (default: 1.0)
- `--top-p`: Top-p (nucleus) sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--orthogonal`: Use orthogonalized addition method
- `--verbose`: Log full prompt after chat template application

### Python Code Usage

You can use the `SteeringApplier` class directly in Python code with flexible input options.


#### Basic Example (Using model_name)

```python
from pathlib import Path
from psyctl.core.steering_applier import SteeringApplier

# Initialize applier
applier = SteeringApplier()

# Apply steering with model_name
result = applier.apply_steering(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/out.safetensors"),
    input_text="Tell me about yourself",
    strength=1.5
)

print(result)
```

#### Using Persistent Steering (Python API Only - Most Efficient for Multiple Generations)

The `get_steering_applied_model()` method returns a model with steering hooks already attached. This is the most efficient way to generate multiple outputs with the same steering configuration:

```python
from pathlib import Path
from psyctl.core.steering_applier import SteeringApplier

# Initialize applier
applier = SteeringApplier()

# Get model with steering hooks attached
model, tokenizer = applier.get_steering_applied_model(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/out.safetensors"),
    strength=2.0,
    orthogonal=True
)

# Use the model multiple times - hooks remain active
test_inputs = ["Hello", "How are you?", "What's your opinion?"]

for prompt in test_inputs:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, use_cache=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {prompt}")
    print(f"Output: {result}\n")

# Remove steering hooks when done
model.remove_steering()
```

#### Using Pre-loaded Model (Efficient for Multiple Generations)

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from psyctl.core.steering_applier import SteeringApplier

# Load model and tokenizer once
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

# Initialize applier
applier = SteeringApplier()

# Apply steering multiple times with different inputs/strengths
# No need to reload the model each time!
test_inputs = [
    "Hello, how are you?",
    "Tell me about yourself",
    "What is your opinion on AI?"
]

for input_text in test_inputs:
    result = applier.apply_steering(
        model=model,
        tokenizer=tokenizer,
        steering_vector_path=Path("./steering_vector/out.safetensors"),
        input_text=input_text,
        strength=1.5
    )
    print(f"Input: {input_text}")
    print(f"Output: {result}\n")
```

#### Experimenting with Different Strengths

```python
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from psyctl.core.steering_applier import SteeringApplier

# Load model once
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

applier = SteeringApplier()
input_text = "Hello, how are you?"

# Test different steering strengths efficiently
for strength in [0.5, 1.0, 1.5, 2.0, 2.5]:
    result = applier.apply_steering(
        model=model,
        tokenizer=tokenizer,
        steering_vector_path=Path("./steering_vector/rudeness.safetensors"),
        input_text=input_text,
        strength=strength
    )
    print(f"Strength {strength}: {result}\n")
```

#### Using Orthogonalized Addition in Python

```python
from pathlib import Path
from psyctl.core.steering_applier import SteeringApplier

applier = SteeringApplier()

# Apply with orthogonalized addition method
result = applier.apply_steering(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/out.safetensors"),
    input_text="What is your personality like?",
    strength=2.0,
    orthogonal=True,  # Enable orthogonalized addition
    temperature=0.7
)

print(result)
```

#### Using Verbose Logging

Enable verbose logging to see the full prompt after chat template application:

```python
from pathlib import Path
from psyctl.core.steering_applier import SteeringApplier

applier = SteeringApplier()

# Enable verbose to log the full formatted prompt
result = applier.apply_steering(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/out.safetensors"),
    input_text="Hello",
    strength=1.5,
    verbose=True  # Logs full prompt with chat template
)
```

#### Using Per-Layer Strength (Python API Only)

Control steering strength individually for each layer:

```python
from pathlib import Path
from psyctl.core.steering_applier import SteeringApplier

applier = SteeringApplier()

# Apply different strengths to different layers
result = applier.apply_steering(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/multi_layer.safetensors"),
    input_text="Tell me about yourself",
    strength={
        "model.layers[10].mlp.down_proj": 1.0,
        "model.layers[13].mlp.down_proj": 2.5,
        "model.layers[16].mlp.down_proj": 1.5,
        # Layers not specified will use default strength of 1.0
    }
)

print(result)
```

You can also use per-layer strength with `get_steering_applied_model()`:

```python
# Get model with per-layer steering
model, tokenizer = applier.get_steering_applied_model(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./steering_vector/multi_layer.safetensors"),
    strength={
        "model.layers[13].mlp.down_proj": 3.0,  # Strong on this layer
        # Other layers use default 1.0
    },
    orthogonal=True
)

# Generate multiple outputs with this configuration
for prompt in ["Hello", "How are you?"]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, use_cache=False)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

model.remove_steering()
```

## Steering Parameters

### Strength

The `strength` parameter controls how strongly the steering vector affects the model.

**Uniform Strength (float):**

Apply the same strength to all layers:

- `0.0`: No steering (baseline model behavior)
- `1.0`: Default steering strength
- `1.5-2.0`: Strong steering (recommended for subtle personalities)
- `>2.0`: Very strong steering (may produce extreme outputs)

**CLI Example:**
```bash
# Subtle steering
psyctl steering --model "google/gemma-3-270m-it" \
  --steering-vector "./vector.safetensors" \
  --input-text "What is your opinion?" \
  --strength 0.5

# Strong steering
psyctl steering --model "google/gemma-3-270m-it" \
  --steering-vector "./vector.safetensors" \
  --input-text "What is your opinion?" \
  --strength 2.5
```

**Per-Layer Strength (Dict[str, float] - Python API Only):**

Control strength for each layer individually:

```python
# Dictionary mapping layer names to strength values
strength = {
    "model.layers[10].mlp.down_proj": 1.0,   # Mild steering
    "model.layers[13].mlp.down_proj": 2.5,   # Strong steering
    "model.layers[16].mlp.down_proj": 1.5,   # Moderate steering
    # Layers not in dict will use default strength of 1.0
}

result = applier.apply_steering(
    model_name="google/gemma-3-270m-it",
    steering_vector_path=Path("./vector.safetensors"),
    input_text="What is your opinion?",
    strength=strength
)
```

**Benefits of per-layer strength:**
- Fine-grained control over steering behavior
- Can emphasize or de-emphasize specific layers
- Useful for experimenting with layer-specific effects
- Layers not specified in the dict automatically use default strength (1.0)

### Temperature

Controls randomness in text generation:

- `0.0`: Greedy decoding (deterministic)
- `0.5-0.8`: More focused and coherent
- `1.0`: Balanced sampling (default)
- `>1.0`: More creative and diverse

**Example:**
```bash
# Deterministic output
psyctl steering --model "google/gemma-3-270m-it" \
  --steering-vector "./vector.safetensors" \
  --input-text "hello" \
  --temperature 0.0

# Creative output
psyctl steering --model "google/gemma-3-270m-it" \
  --steering-vector "./vector.safetensors" \
  --input-text "hello" \
  --temperature 1.5
```

### Top-p and Top-k

Fine-tune sampling behavior:

- `--top-p`: Nucleus sampling threshold (0.0-1.0)
- `--top-k`: Number of top tokens to consider

**Example:**
```bash
psyctl steering --model "google/gemma-3-270m-it" \
  --steering-vector "./vector.safetensors" \
  --input-text "hello" \
  --top-p 0.95 \
  --top-k 100
```

## Steering Methods

### Simple Addition (Default)

The default method adds the steering vector to model activations after the prompt:

```
output[prompt_length:] = output[prompt_length:] + strength * steering_vector
```

This is the standard CAA (Contrastive Activation Addition) method.

### Orthogonalized Addition

The `--orthogonal` flag enables orthogonalized addition method:

1. Calculate projection of output onto steering vector direction
2. Remove the existing component along that direction
3. Add scaled steering vector

```
norm_steer = steering_vector / ||steering_vector||
proj = (output · norm_steer) * norm_steer
output[prompt_length:] = (output[prompt_length:] - proj) + strength * steering_vector
```

This method orthogonalizes the output with respect to the steering direction before applying the steering vector, providing more controlled modification of model behavior.

**When to use:**
- When steering effects are too strong or unpredictable with simple addition
- When you want more precise control over steering magnitude
- When combining multiple steering vectors to avoid interference
- When fine-tuning steering strength for subtle personality changes

**Example:**
```bash
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./vector.safetensors" \
  --input-text "Describe your personality" \
  --orthogonal \
  --strength 1.5
```

## Examples

### Example 1: Extroversion Steering

```bash
# Extract extroversion steering vector (prerequisite)
psyctl extract.steering \
  --model "google/gemma-3-270m-it" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/extroversion" \
  --output "./vectors/extroversion.safetensors"

# Apply with moderate strength
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/extroversion.safetensors" \
  --input-text "Tell me about your weekend plans" \
  --strength 1.2
```

### Example 2: Multiple Personalities

```bash
# Extract multi-layer steering vector
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layers "model.layers[13].mlp.down_proj,model.layers[14].mlp.down_proj" \
  --dataset "./dataset/agreeableness" \
  --output "./vectors/agreeable_multi.safetensors"

# Apply with orthogonalized addition
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./vectors/agreeable_multi.safetensors" \
  --input-text "What do you think about helping others?" \
  --orthogonal \
  --strength 2.0
```

### Example 3: Comparing Strengths

Test different steering strengths on the same input:

```bash
# No steering (baseline)
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/neuroticism.safetensors" \
  --input-text "I got a bad grade on my test" \
  --strength 0.0

# Mild steering
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/neuroticism.safetensors" \
  --input-text "I got a bad grade on my test" \
  --strength 0.8

# Strong steering
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/neuroticism.safetensors" \
  --input-text "I got a bad grade on my test" \
  --strength 2.0
```

## Advanced Usage

### Chat Template Handling

The steering command automatically detects and applies chat templates for instruction-tuned models:

```bash
# For models with chat templates (Llama, Gemma, etc.)
# Input is automatically formatted as:
# <bos><start_of_turn>user
# Your input text<end_of_turn>
# <start_of_turn>model

psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./vectors/out.safetensors" \
  --input-text "hello"
```

For base models without chat templates, the raw input text is used.

### Multi-Layer Steering

When a steering vector file contains multiple layers, all layers are automatically applied:

```bash
# Extract from multiple layers
psyctl extract.steering \
  --model "google/gemma-3-270m-it" \
  --layers "model.layers[10].mlp.down_proj,model.layers[13].mlp.down_proj,model.layers[16].mlp.down_proj" \
  --dataset "./dataset/caa" \
  --output "./vectors/multi.safetensors"

# Apply to all layers at once
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/multi.safetensors" \
  --input-text "hello" \
  --strength 1.5
```

### Greedy vs Sampling

For reproducible results, use greedy decoding:

```bash
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/out.safetensors" \
  --input-text "Tell me a story" \
  --temperature 0.0
```

For creative outputs, use higher temperature with sampling:

```bash
psyctl steering \
  --model "google/gemma-3-270m-it" \
  --steering-vector "./vectors/out.safetensors" \
  --input-text "Tell me a story" \
  --temperature 1.2 \
  --top-p 0.95
```

## Technical Details

### Hook Implementation

The steering mechanism uses PyTorch forward hooks registered on target layers. The hook function:

1. Receives layer output `(batch_size, sequence_length, hidden_dim)`
2. Applies steering only to tokens after the prompt
3. Returns modified output in the same format

**Code reference:** `src/psyctl/core/steering_applier.py:_make_steering_hook()`

### Prompt Length Tracking

The system tracks prompt length to ensure steering is only applied to generated tokens, not the input prompt. This prevents distorting the input context.

**Special case:** Setting `prompt_length=0` internally applies steering to all tokens (BiPO-style), though this is not exposed via CLI.

### Memory Management

Hooks are automatically cleaned up after generation using try/finally blocks to prevent memory leaks.

## Troubleshooting

### Issue: Steering has no effect

**Solution:**
- Increase `--strength` parameter
- Try `--orthogonal` flag for orthogonalized addition method
- Verify steering vector was extracted from the same model
- Check that layer paths match between extraction and application

### Issue: Output is too extreme

**Solution:**
- Decrease `--strength` parameter (try 0.5-1.0)
- Use `--orthogonal` flag for more controlled steering
- Lower `--temperature` for more focused output

### Issue: Model uses too much memory

**Solution:**
- Use a smaller model (e.g., gemma-3-270m-it instead of gemma-3-27b-it)
- Reduce `--max-tokens` parameter
- The steering process uses `use_cache=False` which increases memory during generation

## See Also

- [Extract Steering Vectors](./EXTRACT.STEERING.md) - How to create steering vectors
- [Build Steering Dataset](./DATASET.BUILD.STEER.md) - How to prepare training data
