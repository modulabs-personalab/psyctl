<p align="center">
  <img src="./docs/images/logo.png" alt="PSYCTL 로고" width="120"/>
</p>

# PSYCTL - LLM 성격 조향 도구

> **개발 중인 프로젝트입니다**
> 현재 개발 중이며 제한된 기능만 지원합니다. 안정적인 기능은 릴리스 노트를 확인해 주세요.

[ModuLabs](https://modulabs.co.kr/labs/337)의 [Persona Lab](https://modulabs.co.kr/labs/337) 프로젝트입니다.

LLM이 특정 성격을 나타내도록 조향(steering)하는 것을 지원하는 도구입니다. 모델과 성격 지정만으로 데이터셋을 자동 생성하고 작업할 수 있는 것이 목표입니다.

---

## 문서

### 핵심 가이드

- **[조향 데이터셋 생성](./docs/DATASET.BUILD.STEER.md)** - 벡터 추출을 위한 조향 데이터셋 생성
- **[조향 벡터 추출](./docs/EXTRACT.STEERING.md)** - mean_diff 또는 BiPO 방법을 사용한 조향 벡터 추출
- **[조향 실험](./docs/STEERING.md)** - CAA(Contrastive Activation Addition)를 사용한 조향 벡터 적용

### 추가 자료

- **[설정](./docs/CONFIGURATION.md)** - 환경 변수 및 성능 튜닝
- **[OpenRouter 연동](./docs/OPENROUTER.md)** - 로컬 GPU 대신 클라우드 API 사용
- **[커뮤니티 데이터셋](./docs/COMMUNITY.DATASETS.md)** - 미리 만들어진 데이터셋 및 레지스트리
- **[문제 해결](./docs/TROUBLESHOOTING.md)** - 일반적인 문제 및 해결 방법
- **[기여 가이드](./docs/CONTRIBUTING.md)** - 개발 가이드 및 기여 지침

---

## 사용자 가이드

### 빠른 시작

#### 설치

**기본 설치 (CPU 버전)**
```bash
# uv 설치 (Windows)
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile "install_uv.ps1"
& .\install_uv.ps1

# 프로젝트 설정
uv venv
& .\.venv\Scripts\Activate.ps1
uv sync
```

**Google Colab에서 설치**
```python
# GitHub에서 직접 설치
!pip install git+https://github.com/modulabs-personalab/psyctl.git

# 또는 특정 브랜치에서 설치
!pip install git+https://github.com/modulabs-personalab/psyctl.git@main

# 환경 변수 설정
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
os.environ['PSYCTL_LOG_LEVEL'] = 'INFO'

# 사용 예시
from psyctl import DatasetBuilder, P2, LLMLoader
```

**GPU 가속 설치 (CUDA 지원)**
```bash
# 기본 설치 후 CUDA 지원 PyTorch 설치
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 설치 확인
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **중요**: `transformers` 패키지가 `torch`를 의존성으로 가지고 있어서 `uv sync` 실행 시 CPU 버전이 자동으로 설치됩니다. GPU를 사용하려면 위의 CUDA 설치 명령을 다시 실행해야 합니다.

#### 기본 사용법

```bash
# 1. 데이터셋 생성
psyctl dataset.build.steer \
  --model "google/gemma-3-27b-it" \
  --personality "Extroversion, Machiavellism" \
  --output "./dataset/steering"

# 2. HuggingFace Hub에 데이터셋 업로드 (선택사항)
psyctl dataset.upload \
  --dataset-file "./dataset/steering/steering_dataset_*.jsonl" \
  --repo-id "username/extroversion-steering"

# 3. 조향 벡터 추출
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/steering" \
  --output "./steering_vector/out.safetensors"

# 4. 조향 실험
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --input-text "Tell me about yourself"

# 5. 인벤토리 테스트
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/out.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Neuroticism"
```

### 명령어 개요

PSYCTL은 5개의 주요 명령어를 제공합니다. 자세한 사용법은 위의 문서 링크를 참조하세요.

| 명령어 | 설명 | 문서 |
|--------|------|------|
| `dataset.build.steer` | 조향 데이터셋 생성 | [가이드](./docs/DATASET.BUILD.STEER.md) |
| `dataset.upload` | HuggingFace에 데이터셋 업로드 | [가이드](./docs/DATASET.BUILD.STEER.md#uploading-to-huggingface-hub) |
| `extract.steering` | 조향 벡터 추출 | [가이드](./docs/EXTRACT.STEERING.md) |
| `steering` | 생성에 조향 적용 | [가이드](./docs/STEERING.md) |
| `benchmark inventory` | 심리학적 인벤토리 테스트 (로짓 기반) | 아래 참조 |
| `benchmark llm-as-judge` | LLM as Judge 테스트 (상황 기반 질문) | 아래 참조 |
| `inventory.list` | 사용 가능한 인벤토리 목록 | 아래 참조 |

**벤치마크 방법:**
- **Inventory**: 표준화된 심리학적 인벤토리(예: IPIP-NEO)를 로짓 기반 점수로 사용합니다. 더 객관적이고 재현 가능합니다.
- **LLM as Judge**: 상황 기반 질문을 생성하고 LLM을 사용하여 응답을 평가합니다. 더 유연하고 맥락 인식이 가능합니다.
  - API 기반 판정자(OpenAI, OpenRouter)의 경우 환경 변수를 설정하세요:
    - OpenAI 모델용 `OPENAI_API_KEY`
    - OpenRouter 모델용 `OPENROUTER_API_KEY`
  - 로컬 모델의 경우 `local-default`(대상 모델 재사용) 또는 `benchmark_config.json`에서 사용자 지정 모델 경로 설정
  - 사용자 지정 API 서버의 경우 `benchmark_config.json`을 편집하여 서버 구성 추가

### 지원 인벤토리

| 인벤토리 | 도메인 | 라이선스 | 비고 |
|----------|--------|----------|------|
| IPIP-NEO-300/120 | Big Five | Public Domain | 전체 및 단축 형식 |
| NPI-40 | 나르시시즘 | 연구용 무료 | 강제 선택형 |
| PNI-52 | 병리적 나르시시즘 | CC-BY-SA | 리커트 1-6 |
| NARQ-18 | 존경 및 경쟁 | CC-BY-NC | 두 개의 하위 척도 |
| MACH-IV | 마키아벨리즘 | Public Domain | 리커트 1-5 |
| LSRP-26 | 사이코패시 | Public Domain | 1차 및 2차 |
| PPI-56 | 사이코패시 | 연구용 무료 | 단축 형식 |

### 설정

PSYCTL은 환경 변수를 통해 설정합니다. **필수:**

```bash
# https://huggingface.co/settings/tokens 에서 토큰 발급
export HF_TOKEN="your_huggingface_token_here"  # Linux/macOS
$env:HF_TOKEN = "your_token_here"              # Windows
```

자세한 설정 옵션(디렉토리, 성능 튜닝, 로깅)은 [설정 가이드](./docs/CONFIGURATION.md)를 참조하세요.

### 전체 워크플로우 예시

```bash
# 1. 외향성 성격을 위한 데이터셋 생성
# 최적의 성능을 위해 배치 크기 설정
export PSYCTL_INFERENCE_BATCH_SIZE="16"

psyctl dataset.build.steer \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --personality "Extroversion" \
  --output "./dataset/extroversion" \
  --limit-samples 1000

# 2. 조향 벡터 추출
psyctl extract.steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --layer "model.layers[13].mlp.down_proj" \
  --dataset "./dataset/extroversion" \
  --output "./steering_vector/extroversion.safetensors"

# 3. 조향을 적용하여 텍스트 생성
psyctl steering \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --input-text "Tell me about yourself"

# 4. 인벤토리로 성격 변화 측정
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Extraversion"

# 5. LLM as Judge로 성격 변화 측정
# 참고: API 기반 판정자의 경우 환경 변수 설정:
#   export OPENAI_API_KEY="your-key"        # OpenAI 모델용
#   export OPENROUTER_API_KEY="your-key"    # OpenRouter 모델용
# 또는 benchmark_config.json에서 사용자 지정 API 서버 설정
psyctl benchmark llm-as-judge \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --steering-vector "./steering_vector/extroversion.safetensors" \
  --trait "Extraversion" \
  --judge-model "local-default" \
  --num-questions 10 \
  --strengths "1.0,2.0,3.0"
```

**더 많은 예시:**
- Python 라이브러리 사용법은 [examples/](examples/) 디렉토리 참조
- 자세한 가이드는 위의 문서 링크 확인

---

## PoC 노트북 (Google Colab)

브라우저에서 psyctl을 직접 실행해 볼 수 있는 Colab 노트북입니다:

| 노트북 | 설명 |
|--------|------|
| [PsyCtl로 Gemma-3-4b-it 성격 조향하기 (한국어)](https://colab.research.google.com/drive/12x92LqwshlDlxH9xn1JkeZTK7JkdTvzJ) | 전체 조향 워크플로우 한국어 버전: 레이어 분석, BiPO 추출, 텍스트 생성 |
| [Steering Gemma-3-4b-it With PsyCtl (English)](https://colab.research.google.com/drive/1h84G02UYAgJ_GSm_1mhnerpK_HhMzO46) | 전체 조향 워크플로우: 레이어 분석, BiPO 추출, 텍스트 생성 |
| [Steering Gemma-3-270m-it With PsyCtl](https://colab.research.google.com/drive/1TJCXjwuYd_IRghpg-uKQc1T4eDNkKdrl) | 270M 소형 모델을 사용한 빠른 테스트 버전 |
| [CAA Extraction (Incremental Mean)](https://colab.research.google.com/drive/1uAFDbxjUXJKuH8CbdDZtGki74aFNajxz) | 메모리 효율적인 Incremental Mean 방식의 수동 CAA 벡터 추출 |
| [BiPO PoC](https://colab.research.google.com/drive/1mK5_VXb8AWX9NzOe93BRSUJDoRbQWh5y) | distilgpt2를 사용한 BiPO 알고리즘 개념 증명 |
| [Build CAA Dataset by P2](https://colab.research.google.com/drive/1AG2sqixvNTZWKCGYqEquPRhPCreA3hdo) | P2 성격 프롬프팅을 활용한 데이터셋 생성 |

> **참고**: 이 노트북들은 HuggingFace 토큰이 필요합니다 (Colab 시크릿에 `HF_TOKEN`으로 설정). GPU 런타임 (T4 이상) 권장.

---

## 기여하기

기여를 환영합니다! [기여 가이드](./docs/CONTRIBUTING.md)에서 다음 내용을 확인하세요:
- 개발 환경 설정
- 코드 스타일 및 표준
- 테스트 가이드라인
- 풀 리퀘스트 프로세스

## 주요 논문
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

## 후원사

<p align="left">
  <a href="https://caveduck.io" target="_blank">
    <img src="https://cdn.caveduck.io/public/assets/logo_white.c2efa9b1d010.svg" alt="Caveduck.io" width="200"/>
  </a>
  <a href="https://modulabs.co.kr/" target="_blank" style="margin-left:20px">
    <img src="https://i.namu.wiki/i/OqOsAbn-fFDwzMVNG3EsqTEMZz13k7wmahhZwkayuE1q9WWWLtMMmaqCM9FJULqHB1CjRGHiQzVoOHkQFQkbHBbNlF9CfB1_yEMVgZWOHnbyHdB-akqxFIOjQ9WXGM7RJF4H0JUGGzxDQxIGXeGtTg.webp" alt="Caveduck.io" width="200"/>
  </a>
</p>
