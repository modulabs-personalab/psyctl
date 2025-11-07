# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PSYCTL is a Python CLI tool for LLM personality steering using Contrastive Activation Addition (CAA) and Bidirectional Preference Optimization (BiPO). The tool enables extracting steering vectors from models to modify personality traits during text generation.

**Core Workflow:**
1. **Dataset Generation**: Create contrastive personality datasets using P2 personality prompts
2. **Vector Extraction**: Extract steering vectors using mean_diff or BiPO methods
3. **Steering Application**: Apply vectors to model activations during inference
4. **Benchmarking**: Test personality changes using psychological inventories (IPIP-NEO, NPI-40, MACH-IV, etc.)

## Development Commands

### Environment Setup
```powershell
# Install uv package manager
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# Create and activate virtual environment
uv venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
uv sync

# Install development dependencies
& .\scripts\install-dev.ps1
```

### Common Development Tasks

**Pre-commit hooks** (automatic code quality checks):
```powershell
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Format code** (must run before commits):
```powershell
& .\scripts\format.ps1
# Runs: ruff check --fix src/ && ruff format src/
```

**Run tests**:
```powershell
& .\scripts\test.ps1
# Runs: pytest -v --cov=psyctl --cov-report=html
```

**Run single test**:
```powershell
uv run pytest tests/test_core/test_layer_accessor.py -v
uv run pytest tests/test_core/test_layer_accessor.py::test_get_layer_basic -v
```

**Complete build** (format + lint + type check + test + install):
```powershell
& .\scripts\build.ps1
# Runs: ruff check --fix, ruff format, pyright, pytest, pip install -e .
```

**Linting and Type Checking**:
```powershell
uv run ruff check src/
uv run pyright src/
```

### Running PSYCTL Commands

**Set required environment variable**:
```powershell
$env:HF_TOKEN = "your_huggingface_token_here"
```

**Generate steering dataset**:
```powershell
psyctl dataset.build.steer --model "gemma-3-270m-it" --personality "Extroversion" --output "./dataset/steering" --limit-samples 100
```

**Extract steering vector**:
```powershell
psyctl extract.steering --model "gemma-3-270m-it" --layer "model.layers[13].mlp.down_proj" --dataset "./dataset/steering" --output "./steering_vector/out.safetensors"
```

**Apply steering**:
```powershell
psyctl steering --model "gemma-3-270m-it" --steering-vector "./steering_vector/out.safetensors" --input-text "Tell me about yourself"
```

## Architecture

### High-Level Structure

```
CLI Layer (cli.py)
    ↓
Commands Layer (commands/)
    ↓
Core Layer (core/)
    ├── Dataset Generation (dataset_builder.py, prompt.py)
    ├── Vector Extraction (steering_extractor.py, extractors/)
    ├── Steering Application (steering_applier.py)
    └── Infrastructure (hook_manager.py, layer_accessor.py)
    ↓
Models Layer (models/)
    ├── LLM Loading (llm_loader.py)
    ├── Vector Storage (vector_store.py)
    └── API Clients (openrouter_client.py)
```

### Key Components

**1. Dataset Builder (`core/dataset_builder.py`)**
- Generates contrastive personality datasets from conversation data (allenai/soda)
- Uses P2 class to create personality prompts ("Extroversion" → detailed personality description)
- Creates positive/neutral response pairs for each situation
- Supports local models and OpenRouter API
- Uses Jinja2 templates for roleplay prompts (`templates/roleplay_prompt.j2`)
- Batch processing with checkpoint support for resumable generation
- Output: JSONL files with `situation`, `char_name`, `positive`, `neutral` fields

**2. Steering Extractor (`core/steering_extractor.py`)**
- Coordinates extraction using pluggable extractor classes
- Three extraction methods:
  - **mean_diff** (`extractors/mean_difference.py`): Computes mean activation difference between positive/neutral responses
  - **denoised_mean_diff** (`extractors/denoised_mean_difference.py`): PCA-based denoising for noise reduction and improved robustness (variance threshold: 0.95)
  - **bipo** (`extractors/bipo.py`): Bidirectional Preference Optimization using DPO loss
- Layer specification via string paths (e.g., `"model.layers[13].mlp.down_proj"`)
- Uses `LayerAccessor` for dynamic layer access
- Uses `ActivationHookManager` to collect activations via PyTorch forward hooks
- Output: safetensors files with embedded metadata

**3. Hook Manager (`core/hook_manager.py`)**
- Manages PyTorch forward hooks for activation collection
- Accumulates activations across batches with incremental mean computation
- Handles padding by using attention masks to exclude padded tokens
- Thread-safe activation storage
- `register_hooks()` → run inference → `get_mean_activations()` → `remove_all_hooks()`

**4. Layer Accessor (`core/layer_accessor.py`)**
- Dynamically accesses model layers via string paths
- Supports bracket indexing: `"model.layers[13].mlp.down_proj"`
- Handles different model architectures (LLaMA, Gemma, Qwen, etc.)
- Used by extractors to access target layers for hook registration

**5. Steering Applier (`core/steering_applier.py`)**
- Applies steering vectors during inference by modifying activations
- Registers hooks that add scaled steering vectors to layer outputs
- Configurable steering strength (default: 1.0)
- Context manager pattern: `with applier.apply_steering(): model.generate(...)`

**6. P2 Personality Prompt Generator (`core/prompt.py`, `core/prompt_openrouter.py`)**
- Converts high-level personality traits into detailed character descriptions
- Uses the model itself to expand traits (meta-prompting)
- Supports local models and OpenRouter API
- Template-based prompt construction with Jinja2

### Directory Structure

```
src/psyctl/
├── cli.py                    # Main CLI entry point
├── config.py                 # Environment variable configuration
├── commands/                 # CLI command implementations
│   ├── dataset.py           # build.steer, upload
│   ├── extract.py           # extract.steering
│   ├── steering.py          # steering (apply)
│   ├── benchmark.py         # benchmark (inventory tests)
│   └── layer.py             # layer.analyze
├── core/                    # Core business logic
│   ├── dataset_builder.py   # Dataset generation
│   ├── steering_extractor.py # Vector extraction coordinator
│   ├── steering_applier.py  # Vector application
│   ├── hook_manager.py      # PyTorch hook management
│   ├── layer_accessor.py    # Dynamic layer access
│   ├── layer_analyzer.py    # Layer analysis utilities
│   ├── prompt.py            # P2 personality prompt (local)
│   ├── prompt_openrouter.py # P2 personality prompt (API)
│   ├── steer_dataset_loader.py # Dataset loading utilities
│   ├── inventory_tester.py  # Psychological inventory testing
│   ├── extractors/          # Extraction methods
│   │   ├── base.py         # Base extractor interface
│   │   ├── mean_difference.py # Mean difference
│   │   ├── denoised_mean_difference.py # PCA-based denoising
│   │   └── bipo.py         # BiPO method
│   ├── analyzers/          # Layer analysis tools
│   │   ├── base.py         # Base analyzer interface
│   │   ├── consensus.py    # Consensus voting across analyzers
│   │   └── svm_analyzer.py # SVM-based separation analysis
│   ├── logger.py           # Logging configuration
│   └── utils.py            # Utility functions
├── models/                 # Model and storage abstractions
│   ├── llm_loader.py      # HuggingFace model loading
│   ├── openrouter_client.py # OpenRouter API client
│   └── vector_store.py    # Safetensors storage
├── data/                  # Static data
│   └── inventories/       # Psychological inventory data
└── templates/             # Jinja2 templates
    ├── roleplay_prompt.j2 # Default roleplay prompt
    └── md_question.j2     # Question template
```

## Key Design Patterns

### 1. Layer Path Specification
Layers are specified as dot-notation strings with bracket indexing:
```python
# Common layer patterns for steering:
"model.layers[13].mlp.down_proj"  # MLP output (recommended)
"model.layers[0].self_attn.o_proj" # Attention output
"model.layers[10].mlp.act_fn"     # After activation function
```

### 2. Hook-Based Activation Collection
```python
# Pattern used throughout extractors:
hook_manager = ActivationHookManager()
layer_modules = {"layer_13": model.model.layers[13].mlp.down_proj}
hook_manager.register_hooks(layer_modules)

# Run inference
with torch.inference_mode():
    outputs = model(**inputs)

# Collect activations
activations = hook_manager.get_mean_activations()
hook_manager.remove_all_hooks()
```

### 3. Extractor Plugin Pattern
New extraction methods can be added by:
1. Create class inheriting from `BaseVectorExtractor` in `core/extractors/`
2. Implement `extract()` method
3. Register in `SteeringExtractor.EXTRACTORS` dict
4. Add CLI option in `commands/extract.py`

### 4. Dataset Format (JSONL)
Steering datasets use JSONL with this structure:
```json
{
  "situation": "Context and dialogue...",
  "char_name": "Character name",
  "positive": "Response exhibiting target personality",
  "neutral": "Neutral baseline response"
}
```

Version 2+ datasets also include:
```json
{
  "positive_text": "Full answer with personality trait",
  "neutral_text": "Full neutral answer"
}
```

### 5. Safetensors Storage
Steering vectors are stored in safetensors format with metadata:
```python
{
  "model.layers[13].mlp.down_proj": torch.Tensor,  # Vector for layer 13
  "model.layers[14].mlp.down_proj": torch.Tensor,  # Vector for layer 14
  "__metadata__": {
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "method": "mean_diff",
    "layers": ["model.layers[13].mlp.down_proj", ...],
    "normalized": false
  }
}
```

### 6. Batch Processing with Checkpoints
Dataset generation processes data in batches and saves checkpoints:
```python
# Batch size from config
batch_size = INFERENCE_BATCH_SIZE // 2

# Save checkpoint every N samples
if num_generated % CHECKPOINT_INTERVAL == 0:
    save_checkpoint(output_file, num_generated)
```

## Configuration System

All configuration is done via environment variables (see `config.py`):

**Required:**
- `HF_TOKEN`: HuggingFace API token (required for model access)

**Optional:**
- `PSYCTL_INFERENCE_BATCH_SIZE`: Batch size for inference (default: 16)
- `PSYCTL_MAX_WORKERS`: Max worker threads (default: 4)
- `PSYCTL_CHECKPOINT_INTERVAL`: Checkpoint frequency (default: 100)
- `PSYCTL_LOG_LEVEL`: Logging level (default: INFO)
- `PSYCTL_CACHE_DIR`: HuggingFace cache location (default: ./temp)
- `PSYCTL_DATASET_DIR`: Dataset storage (default: ./dataset)
- `PSYCTL_STEERING_VECTOR_DIR`: Vector storage (default: ./steering_vector)
- `PSYCTL_RESULTS_DIR`: Results storage (default: ./results)

**GPU Memory Tuning:**
- 24GB+ VRAM: `PSYCTL_INFERENCE_BATCH_SIZE=32`
- 8-16GB VRAM: `PSYCTL_INFERENCE_BATCH_SIZE=16`
- 4-8GB VRAM: `PSYCTL_INFERENCE_BATCH_SIZE=8`

## Testing Standards

**Test Model**: Always use `"gemma-3-270m-it"` for local testing (small, fast model)

**Test Directories**:
- Output: `./results`
- Cache: `./temp`

**Test Structure**:
```
tests/
├── test_cli.py                      # CLI smoke tests
├── test_commands/                   # Command-level tests
│   ├── test_dataset_builder.py
│   ├── test_dataset_upload.py
│   └── test_prompt.py
└── test_core/                       # Core component tests
    ├── test_extractors/
    ├── test_hook_manager.py
    ├── test_layer_accessor.py
    ├── test_layer_analyzer.py
    ├── test_steer_dataset_loader.py
    └── test_steering_applier.py
```

**Running Tests**:
```powershell
# All tests with coverage
& .\scripts\test.ps1

# Specific test file
uv run pytest tests/test_core/test_hook_manager.py -v

# Specific test function
uv run pytest tests/test_core/test_hook_manager.py::test_register_hooks -v

# Skip slow tests
uv run pytest -m "not slow"
```

## Code Style

**Tools**:
- Ruff (linting and formatting, line length: 88)
- Pyright (type checking in strict mode)
- Pre-commit hooks (automatic quality checks on commit)

**Conventions**:
- Google-style docstrings
- Type hints required (strict mode enabled)
- Snake_case for functions/variables
- PascalCase for classes
- No emojis in code or commits

**Import Organization** (ruff handles this automatically):
1. Standard library
2. Third-party packages
3. Local imports



## Type Annotations

The codebase uses Python 3.11+ modern type syntax:
- Uses `X | Y` union syntax with `from __future__ import annotations`
- External library imports (datasets, dotenv, sklearn) have `# type: ignore[import-not-found]` for missing type stubs
- Some type ignores are used for legitimate pyright false positives (e.g., **kwargs unpacking)

**Before Committing**:
```powershell
& .\scripts\format.ps1  # Format with ruff
uv run ruff check src/  # Lint
uv run pyright src/     # Type check
uv run pytest           # Test
```

Note: Pre-commit hooks will automatically run ruff formatting and checks on staged files.

## Common Development Tasks

### Adding a New Extraction Method

1. **Create extractor class** in `src/psyctl/core/extractors/my_method.py`:
```python
from psyctl.core.extractors.base import BaseVectorExtractor

class MyMethodExtractor(BaseVectorExtractor):
    def extract(self, model, tokenizer, layers, dataset_path, **kwargs):
        # Implementation here
        steering_vectors = {}
        for layer_path in layers:
            # Extract vector for this layer
            steering_vectors[layer_path] = extracted_vector
        return steering_vectors
```

2. **Register in `steering_extractor.py`**:
```python
from psyctl.core.extractors.my_method import MyMethodExtractor

class SteeringExtractor:
    EXTRACTORS = {
        'mean_diff': MeanDifferenceActivationVectorExtractor,
        'denoised_mean_diff': DenoisedMeanDifferenceVectorExtractor,
        'bipo': BiPOVectorExtractor,
        'my_method': MyMethodExtractor,  # Add here
    }
```

3. **Add CLI option** in `commands/extract.py`:
```python
@click.option("--method", default="mean_diff",
              help="Extraction method: mean_diff, denoised_mean_diff, bipo, my_method")
```

4. **Add tests** in `tests/test_core/test_extractors/test_my_method.py`

5. **Document** in `docs/EXTRACT.STEERING.md`

### Working with Templates

Jinja2 templates are in `src/psyctl/templates/`:

**Roleplay prompt template** (`roleplay_prompt.j2`):
- Used by `_get_answer()` in dataset builder
- Variables: `user_name`, `char_name`, `p2` (personality description), `situation`
- Can be customized via `DatasetBuilder(roleplay_prompt_template="path/to/custom.j2")`

**Modify templates programmatically**:
```python
builder = DatasetBuilder()
custom_template = """
# Character: {{ char_name }}
Personality: {{ p2 }}
Situation: {{ situation }}
"""
builder.set_roleplay_prompt_template(custom_template)
```

### Memory-Efficient Patterns

**Incremental mean computation** (avoids storing all activations):
```python
mean_activation = None
count = 0
for batch in dataset:
    activations = get_activations(batch)
    for act in activations:
        count += 1
        if mean_activation is None:
            mean_activation = torch.zeros_like(act)
        mean_activation += (act - mean_activation) / count
```

**Batch processing with DataLoader**:
```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
for batch in dataloader:
    with torch.inference_mode():
        outputs = model(**batch)
```

## Important Implementation Details

### PyTorch Dynamo Disabled
The CLI disables PyTorch compiler to avoid Triton issues:
```python
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
```

### Tokenizer Padding Validation
The system validates that tokenizers have proper padding configuration:
- `pad_token` must be set
- `padding_side` should be "left" for decoder-only models
- Automatically adds pad_token if missing
- See `core/utils.py::validate_tokenizer_padding()`

### Chat Template Handling
Dataset generation uses chat templates when available:
```python
try:
    tokenized_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors=None,
    )
except Exception:
    # Fallback to direct tokenization
    tokenized = tokenizer(prompt, return_tensors="pt")
```

### Activation Hook Pattern
Hooks must be removed after use to prevent memory leaks:
```python
try:
    hook_manager.register_hooks(layer_modules)
    # ... run inference ...
    activations = hook_manager.get_mean_activations()
finally:
    hook_manager.remove_all_hooks()  # Always clean up
```

### OpenRouter API Support
Dataset generation can use cloud APIs instead of local models:
```python
builder = DatasetBuilder(
    use_openrouter=True,
    openrouter_api_key="your_key",
    openrouter_max_workers=4  # Parallel requests
)
```

## Known Issues and Limitations

1. **Model Compatibility**: Layer paths vary by architecture. Common patterns:
   - LLaMA/Mistral: `model.layers[N].*`
   - Gemma: `model.layers[N].*`
   - Qwen: `model.layers[N].*`

2. **Memory Management**: Large models require careful batch size tuning. Monitor with `nvidia-smi`.

3. **Dataset Format**: Older datasets may not have `positive_text`/`neutral_text` fields. The loader handles both formats.

4. **Checkpoint Recovery**: If generation is interrupted, delete the `.checkpoint.json` file to start fresh.

## Debugging Tips

**Enable debug logging**:
```powershell
$env:PSYCTL_LOG_LEVEL = "DEBUG"
```

**Check model device placement**:
```python
device = next(model.parameters()).device
print(f"Model is on: {device}")
```

**Verify layer paths**:
```python
from psyctl.core.layer_accessor import LayerAccessor
accessor = LayerAccessor()
layer = accessor.get_layer(model, "model.layers[13].mlp.down_proj")
print(f"Found layer: {layer}")
```

**Test activation collection**:
```python
from psyctl.core.hook_manager import ActivationHookManager
hook_manager = ActivationHookManager()
hook_manager.register_hooks({"test": model.model.layers[13]})
# Run inference and check
print(f"Collected activations: {hook_manager.get_mean_activations()}")
```

## Related Documentation

- **[DATASET.BUILD.STEER.md](./docs/DATASET.BUILD.STEER.md)**: Dataset generation guide
- **[EXTRACT.STEERING.md](./docs/EXTRACT.STEERING.md)**: Vector extraction guide
- **[STEERING.md](./docs/STEERING.md)**: Steering application guide
- **[CONTRIBUTING.md](./docs/CONTRIBUTING.md)**: Development guidelines
- **[CONFIGURATION.md](./docs/CONFIGURATION.md)**: Configuration reference
- **[COMMUNITY.DATASETS.md](./docs/COMMUNITY.DATASETS.md)**: Pre-built datasets

## Key Papers

The implementation is based on these research papers:

- **CAA**: [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- **BiPO**: [Personalized Steering via Bi-directional Preference Optimization](https://arxiv.org/abs/2406.00045)
- **P2**: [Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)
- **Refusal**: [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
