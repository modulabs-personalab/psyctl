<p align="center">
  <img src="assets/logo.png" alt="PSYCTL Logo" width="120"/>
</p>

# PSYCTL

**LLM Personality Steering with Psychology**

Steer LLM personalities by directly modifying model activations — not prompts.
Measure changes with real psychology instruments. Extract, apply, and benchmark steering vectors.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary } [Try in Colab](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/01_quickstart.ipynb){ .md-button }

---

## What is PSYCTL?

PSYCTL is a Python toolkit for steering LLM personalities using **Contrastive Activation Addition (CAA)** and **Bidirectional Preference Optimization (BiPO)**. Unlike prompt engineering, PSYCTL modifies model activations directly — making personality changes consistent and measurable.

- **Extract Steering Vectors** — Generate contrastive datasets and extract personality vectors using mean_diff, denoised mean_diff, or BiPO methods.
- **Apply Personality Steering** — Apply vectors to model activations during inference with configurable strength from -3.0 to +3.0.
- **Measure with Psychology** — Score personality profiles using IPIP-NEO-120 (Big Five), REI-40, SD4-28 (Dark Tetrad), and more.
- **Benchmark and Compare** — Systematically evaluate vectors across multiple strengths and inventories with cross-impact analysis.

## Quick Demo

```python
from psyctl.core.steering_applier import SteeringApplier

applier = SteeringApplier()

# Apply agreeableness steering to any prompt
result = applier.apply_steering(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    steering_vector_path="agreeableness.safetensors",
    input_text="My coworker keeps taking credit for my ideas.",
    strength=2.0,
)
print(result)
```

## Notebooks

Open any notebook directly in Google Colab — no local setup required.

### English

| Notebook | Description | Time |
|----------|-------------|------|
| [01_quickstart](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/01_quickstart.ipynb) | Instant personality steering | ~5 min |
| [02_measure_personality](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/02_measure_personality.ipynb) | Measure with IPIP-NEO-120 | ~8 min |
| [03_generate_dataset](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/03_generate_dataset.ipynb) | Generate steering dataset | ~5 min |
| [04_extract_vector](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/04_extract_vector.ipynb) | Extract with 3 methods | ~10 min |
| [05_layer_analysis](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/05_layer_analysis.ipynb) | Find optimal layers | ~10 min |
| [06_benchmark_vectors](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/06_benchmark_vectors.ipynb) | Benchmark vectors | ~15 min |

### Korean

| 노트북 | 설명 | 소요 시간 |
|--------|------|----------|
| [01_quickstart](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/01_quickstart.ipynb) | 사전학습 벡터로 성격 즉시 조향 | ~5분 |
| [02_measure_personality](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/02_measure_personality.ipynb) | IPIP-NEO-120 심리 검사 | ~8분 |
| [03_generate_dataset](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/03_generate_dataset.ipynb) | 스티어링 데이터셋 생성 | ~5분 |
| [04_extract_vector](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/04_extract_vector.ipynb) | 3가지 방법으로 벡터 추출 | ~10분 |
| [05_layer_analysis](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/05_layer_analysis.ipynb) | 최적 레이어 탐색 | ~10분 |
| [06_benchmark_vectors](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/06_benchmark_vectors.ipynb) | 벡터 벤치마크 | ~15분 |

## Community Hub

PSYCTL is a community-driven project. Use pre-trained vectors or share your own.

**Pre-trained Vectors** — Ready to use, no training needed:

| Personality | Model | Language |
|-------------|-------|----------|
| [Agreeableness](https://huggingface.co/dalekwon/bipo-steering-vectors) | Llama-3.1-8B | English |
| [Neuroticism](https://huggingface.co/dalekwon/bipo-steering-vectors) | Llama-3.1-8B | English |
| [Awfully Sweet](https://huggingface.co/dalekwon/bipo-steering-vectors) | Llama-3.1-8B | English |
| [Paranoid](https://huggingface.co/dalekwon/bipo-steering-vectors) | Llama-3.1-8B | English |
| [Awfully Sweet (KR)](https://huggingface.co/dalekwon/bipo-steering-vectors) | EXAONE-3.5-7.8B | Korean |

[Browse all vectors and datasets](community/index.md){ .md-button } [Share your own](community/share.md){ .md-button .md-button--primary }

---

## Key Papers

- [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (CAA)
- [Personalized Steering via Bi-directional Preference Optimization](https://arxiv.org/abs/2406.00045) (BiPO)
- [Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550) (P2)
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

## Sponsors

<p>
  <a href="https://caveduck.io"><img src="https://cdn.caveduck.io/public/assets/logo_white.c2efa9b1d010.svg" alt="Caveduck.io" width="90"/></a>
</p>

A project by [Persona Lab](https://modulabs.co.kr/labs/337) at ModuLabs.
