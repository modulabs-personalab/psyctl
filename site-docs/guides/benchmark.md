# Benchmark Steering Vectors

PSYCTL provides two methods to evaluate steering vectors.

## Inventory Benchmark (Logprob-based)

Uses standardized psychological inventories with token log-probability scoring. More objective and reproducible.

```bash
psyctl benchmark inventory \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --steering-vector "./vector.safetensors" \
  --inventory "ipip_neo_120" \
  --trait "Agreeableness"
```

### Supported Options

| Flag | Description |
|------|-------------|
| `--inventory` | Inventory name (e.g., `ipip_neo_120`, `rei_40`, `sd4_28`) |
| `--trait` | Specific trait to test (e.g., `N`, `E`, `O`, `A`, `C`) |
| `--strengths` | Comma-separated strengths (e.g., `0.5,1.0,2.0,3.0`) |
| `--layer-spec` | Layer specification (e.g., `15`, `middle`, `0-5`) |

## LLM-as-Judge Benchmark

Generates situation-based questions and uses an LLM to evaluate personality alignment in responses.

```bash
psyctl benchmark llm-as-judge \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --steering-vector "./vector.safetensors" \
  --trait "Extraversion" \
  --judge-model "local-default" \
  --num-questions 10 \
  --strengths "1.0,2.0,3.0"
```

### Judge Configuration

- **Local model**: `local-default` reuses the target model
- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **OpenRouter**: Set `OPENROUTER_API_KEY` environment variable
- **Custom API**: Edit `benchmark_config.json`

## Interactive Notebook

For a hands-on walkthrough, try the [06_benchmark_vectors notebook](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/06_benchmark_vectors.ipynb).
