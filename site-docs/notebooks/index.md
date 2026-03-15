# Notebooks

Interactive Jupyter notebooks that run directly in Google Colab. No local setup required — just click and run.

!!! tip "Requirements"
    Add `HF_TOKEN` to Colab Secrets (key icon in the left sidebar) and enable notebook access before running.

## English

| # | Notebook | Description | GPU | Time |
|---|----------|-------------|-----|------|
| 01 | [Quickstart](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/01_quickstart.ipynb) | Load pre-trained vectors, compare baseline vs steered, explore 5 personalities | T4 | ~5 min |
| 02 | [Measure Personality](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/02_measure_personality.ipynb) | IPIP-NEO-120 scoring, radar chart visualization, multiple inventories | T4 | ~8 min |
| 03 | [Generate Dataset](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/03_generate_dataset.ipynb) | Create contrastive pairs via OpenRouter API or local model | Optional | ~5 min |
| 04 | [Extract Vector](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/04_extract_vector.ipynb) | Compare mean_diff, denoised mean_diff, and BiPO extraction | T4 | ~10 min |
| 05 | [Layer Analysis](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/05_layer_analysis.ipynb) | SVM-based layer ranking with bar chart visualization | T4 | ~10 min |
| 06 | [Benchmark Vectors](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/06_benchmark_vectors.ipynb) | IPIP-NEO-120 benchmark at multiple strengths, cross-impact analysis | T4 | ~15 min |

## 한국어

| # | 노트북 | 설명 | GPU | 소요 시간 |
|---|--------|------|-----|----------|
| 01 | [빠른 시작](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/01_quickstart.ipynb) | 사전학습 벡터 로드, 기본 vs 조향 비교, 5가지 성격 탐색 | T4 | ~5분 |
| 02 | [성격 측정](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/02_measure_personality.ipynb) | IPIP-NEO-120 스코어링, 레이더 차트, 여러 인벤토리 | T4 | ~8분 |
| 03 | [데이터셋 생성](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/03_generate_dataset.ipynb) | OpenRouter API 또는 로컬 모델로 대조 쌍 생성 | 선택 | ~5분 |
| 04 | [벡터 추출](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/04_extract_vector.ipynb) | mean_diff, denoised, BiPO 3가지 추출 비교 | T4 | ~10분 |
| 05 | [레이어 분석](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/05_layer_analysis.ipynb) | SVM 기반 레이어 순위 + 시각화 | T4 | ~10분 |
| 06 | [벡터 벤치마크](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/ko/06_benchmark_vectors.ipynb) | 다양한 강도에서 IPIP-NEO-120 벤치마크 | T4 | ~15분 |
