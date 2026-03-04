# PSYCTL Benchmark Results: BiPO Steering Vectors

**Source:** [dalekwon/bipo-steering-vectors](https://huggingface.co/dalekwon/bipo-steering-vectors)
**Date:** 2026-03-03
**psyctl version:** main (latest)

**Test Environment:**
- GPU: NVIDIA H100 80GB HBM3
- VRAM: 81559 MiB
- PyTorch: 2.5.1+cu124
- CUDA: 12.4
- Python: 3.11

---

## Models Used

| Language | Model | Parameters | Steering Layer |
|----------|-------|------------|----------------|
| English | `meta-llama/Llama-3.1-8B-Instruct` | 8B | `model.layers.13.mlp` |
| Korean | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 7.8B | `transformer.h.13.mlp` / `h.11.mlp` |

---

## Steering Vectors

All vectors extracted using **BiPO** (Bi-directional Preference Optimization), 1000 dataset samples each, BF16, shape `[4096]`.

| # | Filename | Personality | Language | Model | Layer | Norm |
|---|----------|-------------|----------|-------|-------|------|
| 1 | `bipo_steering_english_agreeableness.safetensors` | Agreeableness | EN | Llama-3.1-8B | `model.layers.13.mlp` | 2.54 |
| 2 | `bipo_steering_english_neuroticism.safetensors` | Neuroticism | EN | Llama-3.1-8B | `model.layers.13.mlp` | 3.29 |
| 3 | `bipo_steering_english_awfully_sweet.safetensors` | Awfully Sweet | EN | Llama-3.1-8B | `model.layers.13.mlp` | 2.45 |
| 4 | `bipo_steering_english_paranoid.safetensors` | Paranoid | EN | Llama-3.1-8B | `model.layers.13.mlp` | 3.49 |
| 5 | `bipo_steering_english_very_lascivious.safetensors` | Very Lascivious | EN | Llama-3.1-8B | `model.layers.13.mlp` | 2.05 |
| 6 | `bipo_steering_korean_awfully_sweet.safetensors` | Awfully Sweet | KR | EXAONE-3.5-7.8B | `transformer.h.13.mlp` | - |
| 7 | `bipo_steering_korean_rude.safetensors` | Rude | KR | EXAONE-3.5-7.8B | `transformer.h.13.mlp` | - |
| 8 | `bipo_steering_korean_lewd.safetensors` | Lewd | KR | EXAONE-3.5-7.8B | `transformer.h.11.mlp` | - |

**Vector Cosine Similarity (English):**

|  | agree | neuro | paranoid | sweet | lasciv |
|--|-------|-------|----------|-------|--------|
| agreeableness | 1.00 | 0.25 | 0.14 | **0.57** | 0.44 |
| neuroticism | 0.25 | 1.00 | **0.55** | 0.27 | 0.20 |
| paranoid | 0.14 | **0.55** | 1.00 | 0.14 | 0.18 |
| awfully_sweet | **0.57** | 0.27 | 0.14 | 1.00 | **0.56** |
| very_lascivious | 0.44 | 0.20 | 0.18 | **0.56** | 1.00 |

**CaveduckAI Dataset Overlap:**
- `korean_rude` <-> [CaveduckAI/steer-personality-rudeness-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-rudeness-ko)
- `korean_lewd` <-> [CaveduckAI/steer-personality-lewd-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-lewd-ko)

---

## Part 1: IPIP-NEO-120 Inventory Benchmark (English)

Target: `meta-llama/Llama-3.1-8B-Instruct` | Method: logprob-weighted scoring | 120 items (24 per domain)

**Baseline (no steering):** N=66.3 | E=71.7 | O=77.2 | A=81.8 | C=79.7

### Agreeableness Vector (target trait only)

| Strength | Baseline | Steered | Change | Note |
|----------|----------|---------|--------|------|
| -3.0 | 81.8 | 71.8 | -10.0 | model degradation |
| -2.0 | 81.8 | 70.9 | -10.9 | model degradation |
| -1.0 | 81.8 | 76.4 | -5.4 | |
| -0.5 | 81.8 | 79.5 | -2.3 | |
| **+0.5** | **81.8** | **83.5** | **+1.7** | **sweet spot** |
| +1.0 | 81.8 | 79.0 | -2.8 | logprob distortion starts |
| +2.0 | 81.8 | 53.0 | -28.8 | model broken |
| +3.0 | 81.8 | 62.9 | -18.9 | model broken |

### Neuroticism Vector (target trait only)

| Strength | Baseline | Steered | Change | Note |
|----------|----------|---------|--------|------|
| -3.0 | 66.3 | 73.5 | +7.2 | model degradation |
| -2.0 | 66.3 | 71.8 | +5.5 | model degradation |
| -1.0 | 66.3 | 67.6 | +1.3 | |
| **-0.5** | **66.3** | **65.9** | **-0.4** | **reduces neuroticism** |
| +0.5 | 66.3 | 67.3 | +1.0 | |
| +1.0 | 66.3 | 70.4 | +4.1 | |
| **+2.0** | **66.3** | **84.7** | **+18.4** | **peak effect** |
| +3.0 | 66.3 | 80.0 | +13.7 | starts degrading |

---

## Cross-Impact Analysis: Non-Big-Five Vectors on All Big Five Domains

### Paranoid Vector

| Strength | N | E | O | A | C | Pattern |
|----------|---|---|---|---|---|---------|
| -3.0 | +21.3 | +18.3 | -5.9 | -25.3 | -11.1 | model broken |
| -2.0 | +7.9 | +7.3 | -2.4 | -10.0 | -4.9 | model broken |
| -1.0 | +1.5 | +3.1 | -1.3 | -3.2 | -1.2 | |
| -0.5 | +0.2 | +1.8 | -0.7 | -1.0 | -0.0 | mild reverse |
| **+0.5** | **+0.6** | **-3.5** | **-0.8** | **-1.5** | **-2.5** | **expected direction** |
| **+1.0** | **+2.4** | **-5.2** | **-3.8** | **-5.8** | **-6.4** | **clear paranoid profile** |
| +2.0 | +20.6 | +14.7 | -3.0 | -22.7 | -10.8 | model broken |
| +3.0 | +5.6 | -0.4 | -5.1 | -9.6 | -7.9 | degraded |

**Pattern:** Paranoid increases N, decreases E/A/C (withdrawn, distrustful, disorganized)

### Awfully Sweet Vector

| Strength | N | E | O | A | C | Pattern |
|----------|---|---|---|---|---|---------|
| -3.0 | +10.6 | +9.1 | -3.5 | -13.1 | -6.5 | model broken |
| -2.0 | +6.4 | +5.2 | -3.5 | -8.7 | -4.3 | model broken |
| -1.0 | +0.2 | -1.1 | -2.6 | -2.1 | -1.6 | mild reverse |
| -0.5 | -0.5 | -1.6 | -1.7 | -0.7 | -0.9 | mild reverse |
| **+0.5** | **+0.4** | **+4.6** | **+3.1** | **-0.4** | **+1.6** | **E/O boost** |
| **+1.0** | **+0.4** | **+14.6** | **+6.3** | **-4.5** | **+5.1** | **strong E/O/C boost** |
| +2.0 | +9.8 | +23.3 | +4.8 | -14.4 | +2.9 | overdriven |
| +3.0 | +24.0 | +23.3 | -3.1 | -27.0 | -10.3 | model broken |

**Pattern:** Sweet dramatically increases E (extraversion), also boosts O and C. Paradoxically decreases A at high strengths.

### Very Lascivious Vector

| Strength | N | E | O | A | C | Pattern |
|----------|---|---|---|---|---|---------|
| -3.0 | +3.0 | +1.7 | -3.8 | -4.0 | -3.5 | degraded |
| -2.0 | -1.3 | -2.4 | -3.2 | +1.0 | -1.0 | |
| **-1.0** | **-2.2** | **-4.0** | **-2.2** | **+1.8** | **-0.4** | **reverse: reserved, agreeable** |
| -0.5 | -1.7 | -3.0 | -1.3 | +1.4 | -0.3 | |
| **+0.5** | **+2.6** | **+5.2** | **+1.6** | **-3.0** | **-0.3** | **bold, outgoing** |
| **+1.0** | **+6.0** | **+11.2** | **+1.7** | **-8.1** | **-1.6** | **strong extraversion + neuroticism** |
| +2.0 | +13.5 | +17.5 | -0.2 | -16.6 | -4.8 | overdriven |
| +3.0 | +14.8 | +12.9 | -3.7 | -16.4 | -6.5 | starts degrading |

**Pattern:** Lascivious strongly increases E and N, decreases A. Most bidirectionally consistent vector.

---

## Part 2: LLM-as-Judge Benchmark (English)

Target: `meta-llama/Llama-3.1-8B-Instruct` | Judge: same model (self-evaluation, `local-default` config with `model_path="auto"`)
Scale: 0-5 (Personality score = trait presence, Relevance = response quality)
Questions: 8 custom English questions per trait (see `examples/08_benchmark_bipo_vectors.py`)

### Agreeableness (8 questions)

| Strength | BL Pers | ST Pers | P Change | BL Rel | ST Rel | R Change | Note |
|----------|---------|---------|----------|--------|--------|----------|------|
| -3.0 | 3.74 | 1.58 | -2.16 | 4.01 | 1.00 | -3.00 | broken |
| -2.0 | 3.74 | 2.47 | -1.27 | 4.01 | 2.70 | -1.31 | degraded |
| -1.0 | 3.74 | 2.85 | -0.89 | 4.01 | 3.50 | -0.50 | less agreeable |
| -0.5 | 3.74 | 3.49 | -0.25 | 4.01 | 3.80 | -0.21 | slightly less |
| +0.5 | 3.74 | 4.29 | **+0.55** | 4.01 | 4.89 | +0.88 | more agreeable |
| +1.0 | 3.74 | 4.95 | **+1.21** | 4.01 | 4.99 | +0.98 | highly agreeable |
| **+2.0** | **3.74** | **5.00** | **+1.26** | **4.01** | **5.00** | **+0.99** | **max score** |
| +3.0 | 3.74 | 1.97 | -1.77 | 4.01 | 1.54 | -2.47 | **broken** |

### Neuroticism (8 questions)

| Strength | BL Pers | ST Pers | P Change | BL Rel | ST Rel | R Change | Note |
|----------|---------|---------|----------|--------|--------|----------|------|
| -3.0 | 3.77 | 0.63 | -3.15 | 4.74 | 0.91 | -3.84 | broken |
| -2.0 | 3.77 | 0.48 | -3.29 | 4.74 | 1.12 | -3.63 | broken |
| -1.0 | 3.77 | 0.71 | -3.06 | 4.74 | 3.53 | -1.22 | broken |
| -0.5 | 3.77 | 2.67 | -1.10 | 4.74 | 4.48 | -0.26 | less neurotic |
| +0.5 | 3.77 | 3.68 | -0.09 | 4.74 | 4.00 | -0.74 | minimal effect |
| +1.0 | 3.77 | 4.09 | **+0.31** | 4.74 | 3.65 | -1.09 | more neurotic |
| **+2.0** | **3.77** | **4.85** | **+1.08** | **4.74** | **4.79** | **+0.05** | **peak** |
| +3.0 | 3.77 | 4.37 | +0.60 | 4.74 | 3.86 | -0.88 | degrading |

### Paranoid (8 questions)

| Strength | BL Pers | ST Pers | P Change | BL Rel | ST Rel | R Change | Note |
|----------|---------|---------|----------|--------|--------|----------|------|
| -3.0 | 2.60 | 2.88 | +0.28 | 4.19 | 2.08 | -2.11 | broken |
| -2.0 | 2.60 | 1.41 | -1.19 | 4.19 | 0.48 | -3.70 | broken |
| -1.0 | 2.60 | 0.99 | -1.61 | 4.19 | 3.53 | -0.65 | less paranoid |
| -0.5 | 2.60 | 2.15 | -0.45 | 4.19 | 3.93 | -0.25 | slightly less |
| +0.5 | 2.60 | 3.24 | +0.64 | 4.19 | 2.81 | -1.38 | more paranoid |
| +1.0 | 2.60 | 3.96 | **+1.36** | 4.19 | 1.53 | -2.65 | paranoid (R drops) |
| **+2.0** | **2.60** | **4.69** | **+2.09** | **4.19** | **3.24** | **-0.94** | **peak paranoid** |
| +3.0 | 2.60 | 3.61 | +1.00 | 4.19 | 3.41 | -0.77 | degrading |

### Awfully Sweet (8 questions)

| Strength | BL Pers | ST Pers | P Change | BL Rel | ST Rel | R Change | Note |
|----------|---------|---------|----------|--------|--------|----------|------|
| -3.0 | 4.38 | 1.87 | -2.51 | 4.69 | 1.93 | -2.76 | broken |
| -2.0 | 4.38 | 2.49 | -1.89 | 4.69 | 2.72 | -1.97 | degraded |
| -1.0 | 4.38 | 3.63 | -0.75 | 4.69 | 3.69 | -1.01 | less sweet |
| -0.5 | 4.38 | 3.31 | -1.07 | 4.69 | 3.83 | -0.87 | less sweet |
| +0.5 | 4.38 | 4.66 | +0.28 | 4.69 | 4.87 | +0.18 | sweeter |
| +1.0 | 4.38 | 4.91 | **+0.53** | 4.69 | 4.97 | +0.28 | very sweet |
| **+2.0** | **4.38** | **5.00** | **+0.62** | **4.69** | **5.00** | **+0.31** | **max score** |
| **+3.0** | **4.38** | **5.00** | **+0.62** | **4.69** | **5.00** | **+0.31** | **still max!** |

### Very Lascivious (8 questions)

| Strength | BL Pers | ST Pers | P Change | BL Rel | ST Rel | R Change | Note |
|----------|---------|---------|----------|--------|--------|----------|------|
| -3.0 | 3.75 | 1.33 | -2.42 | 4.25 | 2.44 | -1.81 | broken |
| -2.0 | 3.75 | 1.83 | -1.92 | 4.25 | 2.52 | -1.72 | degraded |
| -1.0 | 3.75 | 2.98 | -0.76 | 4.25 | 3.47 | -0.78 | less lascivious |
| -0.5 | 3.75 | 3.80 | +0.06 | 4.25 | 4.11 | -0.13 | near baseline |
| +0.5 | 3.75 | 4.37 | **+0.63** | 4.25 | 4.74 | +0.49 | more expressive |
| +1.0 | 3.75 | 4.65 | **+0.91** | 4.25 | 4.62 | +0.38 | clearly lascivious |
| **+2.0** | **3.75** | **4.93** | **+1.19** | **4.25** | **4.89** | **+0.64** | **peak** |
| +3.0 | 3.75 | 4.83 | +1.08 | 4.25 | 4.24 | -0.00 | slight degradation |

---

## Key Findings

### 1. Breakdown Thresholds

Every vector has a strength threshold beyond which model output degrades:

| Vector | Breakdown Point | Max Effective Strength | Most Robust Range |
|--------|----------------|----------------------|-------------------|
| Agreeableness | +3.0 | +2.0 | +0.5 ~ +2.0 |
| Neuroticism | +3.0 | +2.0 | +1.0 ~ +2.0 |
| Paranoid | +3.0 | +2.0 | +0.5 ~ +2.0 |
| **Awfully Sweet** | **None (up to +3.0)** | **+3.0** | **+0.5 ~ +3.0** |
| Very Lascivious | +3.0 (mild) | +2.0 | +0.5 ~ +2.0 |

### 2. Inventory vs Judge Discrepancy

The IPIP-NEO inventory (logprob-based) and LLM-as-Judge evaluate differently:

| Vector | Inventory sweet spot | Judge sweet spot | Explanation |
|--------|---------------------|------------------|-------------|
| Agreeableness | +0.5 only | +1.0 ~ +2.0 | Logprob distortion at strength >= 1.0 |
| Neuroticism | +2.0 | +2.0 | Both methods agree |

**Conclusion:** LLM-as-Judge is more robust at higher strengths. Inventory evaluation is only reliable at low strengths (< 1.0) where the activation perturbation doesn't distort the next-token probability distribution.

### 3. Cross-Impact Personality Profiles

Each non-Big-Five vector creates a distinctive Big Five signature:

| Vector | N | E | O | A | C | Interpretation |
|--------|---|---|---|---|---|----------------|
| **Paranoid** (+1.0) | +2.4 | **-5.2** | -3.8 | **-5.8** | **-6.4** | Withdrawn, distrustful, disorganized |
| **Awfully Sweet** (+1.0) | +0.4 | **+14.6** | +6.3 | -4.5 | +5.1 | Warm, outgoing, curious, disciplined |
| **Very Lascivious** (+1.0) | **+6.0** | **+11.2** | +1.7 | **-8.1** | -1.6 | Bold, impulsive, disagreeable |

### 4. Vector Robustness Ranking (by Judge)

1. **Awfully Sweet** - Most robust (P:5.0 at both +2.0 and +3.0, no breakdown)
2. **Very Lascivious** - Very robust (P:4.93 at +2.0, only mild degradation at +3.0)
3. **Agreeableness** - Good (P:5.0 at +2.0, breaks at +3.0)
4. **Neuroticism** - Moderate (P:4.85 at +2.0, degrades at +3.0)
5. **Paranoid** - Least robust (R drops significantly even at effective strengths)

---

## How to Reproduce

```bash
# 1. Install psyctl
git clone https://github.com/modulabs-personalab/psyctl.git
cd psyctl
uv venv && source .venv/bin/activate
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Set HuggingFace token
export HF_TOKEN="your_token_here"

# 3. Run full benchmark (downloads vectors automatically)
python examples/08_benchmark_bipo_vectors.py

# 4. Run only English inventory benchmark
python examples/08_benchmark_bipo_vectors.py --english-only --inventory-only

# 5. Custom strengths (bidirectional)
python examples/08_benchmark_bipo_vectors.py --strengths -3.0,-2.0,-1.0,-0.5,0.5,1.0,2.0,3.0

# 6. Disable cross-impact analysis (faster)
python examples/08_benchmark_bipo_vectors.py --no-cross-impact
```

Results are saved to `./results/benchmark_bipo/`:
- `BENCHMARK_RESULTS.md` -- auto-generated markdown summary
- `all_results.json` -- consolidated JSON with all results
- `inventory/` -- per-run inventory JSON files
- `judge/` -- per-run LLM-as-Judge JSON files

---

*Benchmarked with [psyctl](https://github.com/modulabs-personalab/psyctl) on NVIDIA H100 80GB | 2026-03-03*
