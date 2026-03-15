# Community Hub

PSYCTL is built around community contributions. Share your steering vectors and datasets so others can steer LLM personalities without training from scratch.

---

## Pre-trained Steering Vectors

Ready-to-use vectors extracted with BiPO. Download and apply immediately.

| Vector | Personality | Model | Language | Method |
|--------|-------------|-------|----------|--------|
| [agreeableness](https://huggingface.co/dalekwon/bipo-steering-vectors) | Agreeableness | Llama-3.1-8B-Instruct | English | BiPO |
| [neuroticism](https://huggingface.co/dalekwon/bipo-steering-vectors) | Neuroticism | Llama-3.1-8B-Instruct | English | BiPO |
| [awfully_sweet](https://huggingface.co/dalekwon/bipo-steering-vectors) | Extremely Kind | Llama-3.1-8B-Instruct | English | BiPO |
| [paranoid](https://huggingface.co/dalekwon/bipo-steering-vectors) | Paranoid | Llama-3.1-8B-Instruct | English | BiPO |
| [very_lascivious](https://huggingface.co/dalekwon/bipo-steering-vectors) | Bold/Sensation-seeking | Llama-3.1-8B-Instruct | English | BiPO |
| [awfully_sweet_kr](https://huggingface.co/dalekwon/bipo-steering-vectors) | Extremely Kind | EXAONE-3.5-7.8B | Korean | BiPO |
| [rude_kr](https://huggingface.co/dalekwon/bipo-steering-vectors) | Rude | EXAONE-3.5-7.8B | Korean | BiPO |
| [lewd_kr](https://huggingface.co/dalekwon/bipo-steering-vectors) | Lewd | EXAONE-3.5-7.8B | Korean | BiPO |

**Use a vector in 3 lines:**

```python
from psyctl.core.steering_applier import SteeringApplier
applier = SteeringApplier()
result = applier.apply_steering(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    steering_vector_path="./vectors/bipo_steering_english_agreeableness.safetensors",
    input_text="Tell me about yourself.",
    strength=2.0,
)
```

---

## Steering Datasets

Contrastive pair datasets for extracting new vectors. Each sample contains a personality-exhibiting (positive) response and a neutral baseline.

| Dataset | Personality | Language | Samples | Model |
|---------|-------------|----------|---------|-------|
| [steer-personality-extroversion-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-extroversion-ko) | Extroversion | Korean | 100 | kimi-k2 |
| [steer-personality-rudeness-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-rudeness-ko) | Rudeness | Korean | 500 | kimi-k2 |
| [steer-personality-lewd-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-lewd-ko) | Lewd | Korean | - | kimi-k2 |

**Source dialogue datasets:**

| Dataset | Language | Samples | Description |
|---------|----------|---------|-------------|
| [allenai/soda](https://huggingface.co/datasets/allenai/soda) | English | ~1.5M | Social dialogue scenarios |
| [CaveduckAI/simplified_soda_kr](https://huggingface.co/datasets/CaveduckAI/simplified_soda_kr) | Korean | - | Korean SoDA |

---

## Want to contribute?

See [How to Share](share.md) for a step-by-step guide on sharing your vectors and datasets with the community.
