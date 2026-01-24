# 새로운 인벤토리 테스트 추가하기

이 문서는 PSYCTL에 새로운 심리검사 인벤토리를 추가하는 방법을 단계별로 안내합니다. 레지스트리 패턴을 사용하므로 기존 코드를 수정하지 않고도 새 인벤토리를 추가할 수 있습니다.

## 아키텍처 개요

인벤토리 시스템은 다음 구성요소로 이루어져 있습니다:

```
src/psyctl/data/
├── benchmark_config.json          # 중앙 설정 파일 (모집단 규준, 메타데이터)
└── inventories/
    ├── base.py                    # BaseInventory 추상 클래스
    ├── registry.py                # @register_inventory 데코레이터 + 팩토리
    ├── __init__.py                # 패키지 export
    ├── ipip_neo.py                # IPIP-NEO 구현 (Big Five)
    ├── ipip_neo_120_items_en.json # IPIP-NEO 문항 데이터
    ├── rei.py                     # REI 구현 (이중처리)
    └── rei_40_items_en.json       # REI 문항 데이터
```

**핵심 설계:**
- 각 인벤토리는 `BaseInventory`를 상속하는 클래스
- `@register_inventory("name")` 데코레이터로 글로벌 레지스트리에 등록
- `create_inventory("name_version")` 팩토리 함수로 이름 기반 인스턴스 생성
- 설정(모집단 규준, 메타데이터)은 `benchmark_config.json`에 관리
- 문항 데이터는 별도 JSON 파일로 관리

---

## 단계별 튜토리얼

가상의 "Rosenberg Self-Esteem Scale (RSES-10)"을 예시로 사용합니다. 단일 도메인으로 구성된 10문항 자존감 척도입니다.

### 1단계: 문항 데이터 파일 준비

`src/psyctl/data/inventories/rses_10_items_en.json` 파일을 생성합니다:

```json
[
  {
    "id": "rses-001",
    "text": "On the whole, I am satisfied with myself.",
    "keyed": "plus",
    "domain": "SE",
    "facet": null,
    "num": 1,
    "choices": [
      {"text": "Strongly Disagree", "score": 1},
      {"text": "Disagree", "score": 2},
      {"text": "Agree", "score": 3},
      {"text": "Strongly Agree", "score": 4}
    ]
  },
  {
    "id": "rses-002",
    "text": "At times I think I am no good at all.",
    "keyed": "minus",
    "domain": "SE",
    "facet": null,
    "num": 2,
    "choices": [
      {"text": "Strongly Disagree", "score": 1},
      {"text": "Disagree", "score": 2},
      {"text": "Agree", "score": 3},
      {"text": "Strongly Agree", "score": 4}
    ]
  }
]
```

**문항별 필수 필드:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | string | 고유 문항 식별자 (예: `"rses-001"`) |
| `text` | string | 응답자에게 제시되는 문장 |
| `keyed` | `"plus"` 또는 `"minus"` | 채점 방향. `"minus"` 문항은 역채점됨 |
| `domain` | string | 도메인/하위척도 코드 (예: `"SE"`) |
| `facet` | string 또는 null | 선택적 하위요인 식별자 |
| `num` | int | 문항 순서 번호 (1부터 시작) |
| `choices` | array | 텍스트 라벨과 숫자 점수가 포함된 응답 선택지 |

**`keyed` 필드 설명:**
- `"plus"`: 리커트 척도의 높은 점수 = 높은 특성 점수
- `"minus"`: 계산 시 점수가 역전됨: `score = (최대값 + 1) - 원점수`
  - 1-5점 척도: 역전값 = `6 - score`
  - 1-4점 척도: 역전값 = `5 - score`

---

### 2단계: benchmark_config.json에 설정 추가

`src/psyctl/data/benchmark_config.json`을 열고 `"inventories"` 섹션에 항목을 추가합니다:

```json
{
  "inventories": {
    "ipip_neo_120": { ... },
    "rei_40": { ... },
    "rses_10": {
      "name": "RSES-10",
      "description": "Rosenberg Self-Esteem Scale (10 items)",
      "version": "10",
      "language": "en",
      "num_questions": 10,
      "data_file": "rses_10_items_en.json",
      "source": "Rosenberg, M. (1965). Society and the adolescent self-image.",
      "license": "Public Domain",
      "domains": {
        "SE": {
          "name": "Self-Esteem",
          "num_items": 10,
          "population_mean": 22.0,
          "population_std": 5.0
        }
      }
    }
  }
}
```

**필수 설정 필드:**

| 필드 | 설명 |
|------|------|
| `name` | 인벤토리 표시 이름 |
| `description` | 간단한 설명 |
| `version` | 버전 문자열 (레지스트리 키 파싱에 사용) |
| `language` | 언어 코드 (예: `"en"`) |
| `num_questions` | 총 문항 수 |
| `data_file` | 문항 JSON 파일명 (inventories/ 기준 상대 경로) |
| `source` | 출처 인용 또는 URL |
| `license` | 라이선스 유형 |
| `domains` | 도메인 코드와 설정을 매핑하는 딕셔너리 |

**도메인 설정 필드:**

| 필드 | 설명 |
|------|------|
| `name` | 도메인 전체 이름 |
| `num_items` | 해당 도메인의 문항 수 |
| `population_mean` | z-점수 산출을 위한 모집단 평균 |
| `population_std` | z-점수 산출을 위한 모집단 표준편차 |

**모집단 규준을 찾는 방법:**
- 척도의 원 논문
- 메타분석 논문
- 검사 매뉴얼 또는 기술 문서
- 구할 수 없는 경우: 평균 = (문항 수 x 중간값), 표준편차는 유사 척도에서 추정

---

### 3단계: 인벤토리 클래스 구현

`src/psyctl/data/inventories/rses.py` 파일을 생성합니다:

```python
"""RSES (Rosenberg Self-Esteem Scale) implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseInventory
from .registry import register_inventory


@register_inventory("rses")
class RSES(BaseInventory):
    """RSES (Rosenberg Self-Esteem Scale) - Global self-esteem.

    Measures global self-esteem based on Rosenberg (1965).

    Domains:
        SE: Self-Esteem
    """

    def __init__(self, version: str = "10"):
        """Initialize RSES inventory.

        Args:
            version: Version of the inventory ("10")
        """
        self.name = f"RSES-{version}"
        self.domain = "Self-Esteem"
        self.license = "Public Domain"
        super().__init__(version=version)

    def _load_config(self) -> dict[str, Any]:
        """Load inventory configuration."""
        config_path = Path(__file__).parent.parent / "benchmark_config.json"
        with open(config_path, encoding="utf-8") as f:
            all_configs = json.load(f)

        inventory_key = f"rses_{self.version}"
        if (
            "inventories" not in all_configs
            or inventory_key not in all_configs["inventories"]
        ):
            raise ValueError(
                f"RSES version '{self.version}' not found in config"
            )

        return all_configs["inventories"][inventory_key]

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load questions from data file."""
        data_file = self.config["data_file"]
        data_path = Path(__file__).parent / data_file

        with open(data_path, encoding="utf-8") as f:
            questions = json.load(f)

        return questions

    def get_supported_traits(self) -> list[dict[str, str]]:
        """Get list of supported traits.

        Returns:
            List of dicts with trait code and full name
        """
        traits = []
        for code, info in self.config["domains"].items():
            traits.append({"code": code, "name": info["name"]})
        return traits

    def _normalize_trait(self, trait: str) -> str:
        """Normalize trait name to domain code.

        Args:
            trait: Trait code or full name

        Returns:
            Normalized trait code

        Raises:
            ValueError: If trait is not recognized
        """
        trait_map: dict[str, str] = {
            "SE": "SE",
            "Self-Esteem": "SE",
            "self-esteem": "SE",
            "self esteem": "SE",
        }

        trait_code = trait_map.get(trait) or trait_map.get(trait.lower())

        if trait_code is None:
            valid_traits = list(self.config["domains"].keys())
            raise ValueError(
                f"Unrecognized trait '{trait}'. Valid traits: {valid_traits}"
            )

        return trait_code

    def calculate_scores(
        self, responses: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """Calculate scores from responses.

        Args:
            responses: Dict mapping domain codes to list of scores

        Returns:
            Dict with domain scores including raw score, z-score, percentile
        """
        results: dict[str, dict[str, float]] = {}

        for domain, scores in responses.items():
            if domain not in self.config["domains"]:
                continue

            domain_config = self.config["domains"][domain]
            raw_score = sum(scores)
            mean = domain_config["population_mean"]
            std = domain_config["population_std"]

            z_score = (raw_score - mean) / std if std > 0 else 0.0
            percentile = self._z_to_percentile(z_score)

            results[domain] = {
                "domain_name": domain_config["name"],
                "raw_score": raw_score,
                "mean_score": raw_score / len(scores) if scores else 0.0,
                "population_mean": mean,
                "population_std": std,
                "z_score": z_score,
                "percentile": percentile,
                "num_items": float(len(scores)),
            }

        return results

    def _z_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile using normal CDF approximation.

        Args:
            z_score: Standard score

        Returns:
            Percentile (0-100)
        """
        if abs(z_score) < 3:
            percentile = 50 + 34.13 * z_score
        elif z_score >= 3:
            percentile = 99.87
        else:
            percentile = 0.13

        return max(0.0, min(100.0, percentile))
```

**구현 핵심 포인트:**

1. **`@register_inventory("rses")`** - 데코레이터 이름이 레지스트리 접두사가 됩니다. 버전과 결합하면 전체 키는 `"rses_10"`입니다.

2. **`__init__`** - `super().__init__()`을 호출하기 전에 `self.name`, `self.domain`, `self.license`를 설정합니다. 부모 클래스가 `_load_config()`과 `_load_questions()`를 호출합니다.

3. **`_load_config`** - `benchmark_config.json`을 읽고 해당 인벤토리 섹션을 추출합니다.

4. **`_load_questions`** - config의 `data_file` 필드에 지정된 문항 JSON 파일을 읽습니다.

5. **`_normalize_trait`** - 다양한 특성 표현(코드, 전체 이름, 소문자)을 정규 도메인 코드로 매핑합니다.

6. **`calculate_scores`** - 정리된 응답을 `{"SE": [3.0, 4.0, 2.0, ...]}` 형태로 받아 원점수, z-점수, 백분위를 계산합니다.

---

### 4단계: __init__.py에 등록

`src/psyctl/data/inventories/__init__.py`를 수정합니다:

```python
"""Psychological inventory modules."""

from .base import BaseInventory
from .ipip_neo import IPIPNEO
from .registry import (
    INVENTORY_REGISTRY,
    create_inventory,
    get_available_inventories,
    get_registry_info,
)
from .rei import REI
from .rses import RSES  # 이 import 추가

__all__ = [
    "INVENTORY_REGISTRY",
    "IPIPNEO",
    "REI",
    "RSES",              # __all__에 추가
    "BaseInventory",
    "create_inventory",
    "get_available_inventories",
    "get_registry_info",
]
```

import 문이 `@register_inventory` 데코레이터를 트리거하여 클래스를 `INVENTORY_REGISTRY`에 등록합니다.

---

### 5단계: 검증

인벤토리가 정상적으로 통합되었는지 확인합니다:

```powershell
# 1. 린트 검사
uv run ruff check src/psyctl/data/inventories/rses.py

# 2. 타입 검사
uv run pyright src/psyctl/data/inventories/rses.py

# 3. 등록 확인
uv run python -c "
from psyctl.data.inventories import create_inventory, get_available_inventories
print('Available:', get_available_inventories())
inv = create_inventory('rses_10')
print('Name:', inv.name)
print('Traits:', inv.get_supported_traits())
print('Questions:', len(inv.get_questions()))
"

# 4. 기존 테스트 실행 (기존 기능 정상 동작 확인)
uv run pytest tests/ -v
```

예상 출력:
```
Available: ['ipip_neo_120', 'rei_40', 'rses_10']
Name: RSES-10
Traits: [{'code': 'SE', 'name': 'Self-Esteem'}]
Questions: 10
```

---

## 고급 주제

### 상위척도가 있는 다중 도메인 인벤토리

REI-40처럼 계층 구조(하위척도 + 상위척도)를 가진 인벤토리의 경우:

```python
def calculate_scores(self, responses):
    results = {}

    # 1. 하위척도 점수 계산
    for domain, scores in responses.items():
        # ... 표준 계산 ...

    # 2. 하위척도를 결합하여 상위척도 계산
    parent_scales = {
        "R": ["RA", "RE"],  # 합리성 = 합리적 능력 + 합리적 관여
        "E": ["EA", "EE"],  # 경험성 = 경험적 능력 + 경험적 관여
    }

    for parent, subscales in parent_scales.items():
        combined = []
        for sub in subscales:
            if sub in responses:
                combined.extend(responses[sub])
        if combined:
            # config의 상위척도 규준을 사용하여 계산
            ...

    return results
```

또한 `get_questions()`를 오버라이드하여 상위척도 필터링을 지원합니다:

```python
def get_questions(self, trait=None):
    if trait is None:
        return self.questions.copy()

    trait_code = self._normalize_trait(trait)

    # 상위척도: 모든 하위척도의 문항을 반환
    if trait_code == "R":
        return [q for q in self.questions if q.get("domain") in ("RA", "RE")]

    # 하위척도: 직접 필터링
    return [q for q in self.questions if q.get("domain") == trait_code]
```

### 다양한 리커트 척도

채점 시스템은 모든 척도 범위를 지원합니다. 문항 JSON의 `choices` 배열이 척도를 정의합니다:

- **5점** (IPIP-NEO, REI): 점수 1-5, 역전 = `6 - score`
- **4점** (RSES): 점수 1-4, 역전 = `5 - score`
- **7점**: 점수 1-7, 역전 = `8 - score`

역채점 공식: `역전값 = (최대점수 + 1) - 원점수`

테스트 실행기(`inventory_tester.py`)에서 `keyed` 필드를 확인하여 처리합니다:
```python
if question["keyed"] == "minus":
    score = 6.0 - score  # 척도의 최대값에 맞게 조정
```

### OpenRouter API 모델로 테스트

API 기반 테스트(logit 접근 불가)의 경우, 채팅 기반 방식을 사용합니다:

```python
from psyctl.data.inventories import create_inventory
from psyctl.models.openrouter_client import OpenRouterClient

inventory = create_inventory("rses_10")
questions = inventory.get_questions()
client = OpenRouterClient(api_key="...")

domain_responses = {}
for question in questions:
    _, response = client.generate(
        prompt=f'Statement: "{question["text"]}"\nYour rating (1-4):',
        model="anthropic/claude-sonnet-4",
        system_prompt="Respond with ONLY a number 1-4...",
    )
    score = parse_score(response)
    if question["keyed"] == "minus":
        score = 5.0 - score  # 4점 척도 역전

    domain = question["domain"]
    domain_responses.setdefault(domain, []).append(score)

scores = inventory.calculate_scores(domain_responses)
```

전체 동작 예시는 `examples/09_openrouter_inventory_test.py`를 참조하세요.

### 로컬 모델로 테스트 (Logit 기반)

내장된 `InventoryTester`는 logit 확률을 사용하여 더 정밀한 채점을 수행합니다:

```powershell
psyctl benchmark --model "gemma-3-270m-it" --inventory rses_10
```

이 방식은 `_get_score_from_logits()`를 사용하여 점수 토큰에 대한 softmax 확률을 추출하므로 연속값 점수를 생성합니다.

---

## 새 인벤토리 추가 체크리스트

- [ ] 인벤토리 조사: 문항 텍스트, 도메인 구조, 채점 규칙, 모집단 규준
- [ ] 문항 JSON 파일 생성: `src/psyctl/data/inventories/{name}_{version}_items_en.json`
- [ ] `benchmark_config.json`의 `inventories.{name}_{version}`에 설정 추가
- [ ] 인벤토리 클래스 생성: `src/psyctl/data/inventories/{name}.py`
  - [ ] `BaseInventory` 상속
  - [ ] `@register_inventory("{name}")` 데코레이터 추가
  - [ ] `_load_config()`, `_load_questions()` 구현
  - [ ] `get_supported_traits()`, `_normalize_trait()` 구현
  - [ ] `calculate_scores()`에서 z-점수, 백분위 계산 구현
- [ ] `__init__.py`에 import 추가 및 `__all__` 업데이트
- [ ] 린트 실행: `uv run ruff check src/psyctl/data/inventories/{name}.py`
- [ ] 타입 검사: `uv run pyright src/psyctl/data/inventories/{name}.py`
- [ ] 등록 확인: `create_inventory("{name}_{version}")`가 동작하는지 확인
- [ ] 전체 테스트 실행: `uv run pytest tests/ -v`

---

## 기존 인벤토리 목록

| 레지스트리 키 | 클래스 | 문항 수 | 도메인 | 측정 영역 |
|---|---|---|---|---|
| `ipip_neo_120` | `IPIPNEO` | 120 | N, E, O, A, C | Big Five 성격 |
| `rei_40` | `REI` | 40 | RA, RE, EA, EE, R, E | 이중처리 사고양식 |

## 참고문헌

- Pacini, R., & Epstein, S. (1999). The relation of rational and experiential information processing styles. *Journal of Personality and Social Psychology, 76*, 972-987.
- Johnson, J. A. (2014). Measuring thirty facets of the five factor model with a 120-item public domain inventory. *Journal of Research in Personality, 51*, 78-89.
