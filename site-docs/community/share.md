# Share Your Vectors and Datasets

Contributing to the PSYCTL community is straightforward. Extract a vector or generate a dataset, upload to HuggingFace Hub, and submit a pull request to register it.

---

## Share a Steering Vector

### Step 1: Extract your vector

Use any of the three extraction methods:

```bash
psyctl extract.steering \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./my_dataset/data.jsonl" \
  --output "./my_vector.safetensors" \
  --method bipo
```

Or use the [04_extract_vector notebook](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/04_extract_vector.ipynb) for an interactive walkthrough.

### Step 2: Upload to HuggingFace Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Create a new model repository
api.create_repo("your-username/steering-vector-agreeableness", repo_type="model")

# Upload your vector file
api.upload_file(
    path_or_fileobj="./my_vector.safetensors",
    path_in_repo="steering_vector.safetensors",
    repo_id="your-username/steering-vector-agreeableness",
    repo_type="model",
)
```

### Step 3: Register in the community

Open a pull request on [GitHub](https://github.com/modulabs-personalab/psyctl) adding your vector to `docs/COMMUNITY.DATASETS.md`.

**Naming convention:** `{username}/steering-vector-{personality}-{language}`

---

## Share a Steering Dataset

### Step 1: Generate your dataset

```bash
# With OpenRouter API (no GPU needed)
psyctl dataset.build.steer \
  --model "qwen/qwen3-next-80b-a3b-instruct" \
  --personality "Agreeableness" \
  --output "./my_dataset" \
  --limit-samples 100

# Or with a local model
psyctl dataset.build.steer \
  --model "google/gemma-3-270m-it" \
  --personality "Agreeableness" \
  --output "./my_dataset" \
  --limit-samples 100
```

Or use the [03_generate_dataset notebook](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/03_generate_dataset.ipynb).

### Step 2: Upload to HuggingFace Hub

```python
from psyctl.core.dataset_builder import DatasetBuilder

builder = DatasetBuilder()
repo_url = builder.upload_to_hub(
    jsonl_file="./my_dataset/caa_dataset_*.jsonl",
    repo_id="your-username/steer-personality-agreeableness-en",
    private=False,
    license="mit",
)
print(f"Uploaded: {repo_url}")
```

### Step 3: Register in the community

Open a pull request adding your dataset to `docs/COMMUNITY.DATASETS.md`.

**Naming convention:** `{username}/steer-personality-{trait}-{language}`

**Dataset format (JSONL):**

```json
{
  "situation": "Context and dialogue...",
  "char_name": "Character name",
  "positive": "Response exhibiting the target personality",
  "neutral": "Neutral baseline response"
}
```

---

## Tips for high-quality contributions

- **More samples = better vectors.** 100+ samples recommended, 500+ for best results.
- **Use diverse source data.** The allenai/soda dataset covers many social situations.
- **Test before sharing.** Run a quick benchmark with `psyctl benchmark inventory` to verify your vector actually shifts the target personality.
- **Include metadata.** Document the model, method, layer, and personality in your HuggingFace repo README.
- **License clearly.** MIT or CC-BY recommended for maximum reusability.
