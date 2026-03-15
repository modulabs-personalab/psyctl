# Community Datasets Registry

Community-contributed datasets for PSYCTL personality steering.

## 🗂️ Source Datasets

Base conversation datasets used to generate steering datasets. These contain social dialogue scenarios with situational context.

**Format Example**:
```json
{
  "narrative": "Alice is at a party...",
  "speakers": ["Friend", "Alice"],
  "dialogue": ["Want to dance?", "Sure!"]
}
```

| Repository | Language | Samples | License | Description |
|------------|----------|---------|---------|-------------|
| [CaveduckAI/simplified_soda_kr](https://huggingface.co/datasets/CaveduckAI/simplified_soda_kr) | Korean | - | - | Korean version of SoDA dataset |
| [allenai/soda](https://huggingface.co/datasets/allenai/soda) | English | ~1.5M | ODC-BY | Social dialogue dataset |

## 🎯 Steering Datasets

Datasets for extracting personality steering vectors using methods like mean_diff (Mean Difference) and BiPO. Each sample contains personality-specific (positive) and neutral responses to the same situation. The extracted vectors are applied using CAA (Contrastive Activation Addition).

**Format Example**:
```json
{
  "situation": "Alice is at a party...\nFriend: Want to dance?\n",
  "char_name": "Alice",
  "positive": "Absolutely! Let's get everyone together!",
  "neutral": "Sure, I'll join you."
}
```

| Repository | Personality | Language | Samples | Source Dataset | Model | License |
|------------|-------------|----------|---------|----------------|-------|---------|
| [CaveduckAI/steer-personality-extroversion-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-extroversion-ko) | Extroversion (외향성) | Korean | 100 | simplified_soda_kr | kimi-k2-0905 | MIT |
| [CaveduckAI/steer-personality-rudeness-ko](https://huggingface.co/datasets/CaveduckAI/steer-personality-rudeness-ko) | Rudeness (무례함) | Korean | 500 | simplified_soda_kr | kimi-k2-0905 | MIT |

---

## 📝 How to Contribute

1. Generate your dataset using PSYCTL
2. Upload to HuggingFace Hub
3. Add a row to the appropriate table above via pull request

**Dataset Naming Convention**: `{username}/steer-personality-{trait}-{lang}`

**Example**:
```bash
psyctl dataset.build.steer \
  --openrouter-api-key "your-key" \
  --openrouter-model "moonshotai/kimi-k2-0905" \
  --personality "Your Trait" \
  --output "./results/dataset" \
  --limit-samples 100 \
  --dataset "CaveduckAI/simplified_soda_kr"
```

---

## 🔗 Resources

- [PSYCTL GitHub](https://github.com/modulabs-personalab/psyctl)
- [CAA Paper](https://arxiv.org/abs/2312.06681)
