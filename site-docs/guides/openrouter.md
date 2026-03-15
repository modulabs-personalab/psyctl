# OpenRouter Integration Guide

This document describes how to use OpenRouter API with PSYCTL for dataset generation without requiring local GPU resources.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Performance Optimization](#performance-optimization)
- [Cost Management](#cost-management)
- [Troubleshooting](#troubleshooting)

## Overview

OpenRouter integration allows you to generate steering datasets using cloud-based LLMs without downloading models locally. This is particularly useful when:

- You don't have GPU resources available
- You want to use large models (70B+, 405B) without local infrastructure
- You need to test multiple models quickly
- You want to reduce local disk space usage

### Key Features

- **No Local Model Required**: Generate datasets using cloud API
- **Parallel Processing**: Speed up generation with multiple concurrent workers
- **Cost Tracking**: Monitor API usage and costs
- **Model Flexibility**: Easy switching between different models
- **Backward Compatible**: Works alongside existing local model workflow

## Getting Started

### 1. Get OpenRouter API Key

Sign up at [OpenRouter](https://openrouter.ai/) and obtain your API key.

### 2. Set Environment Variable (Optional)

```powershell
# Windows
$env:OPENROUTER_API_KEY = "sk-or-v1-xxxx"

# Linux/macOS
export OPENROUTER_API_KEY="sk-or-v1-xxxx"
```

### 3. Run Dataset Generation

```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --personality "Extroversion" \
  --output "./results/openrouter" \
  --limit-samples 100
```

## Usage

### Basic Command

Generate dataset using OpenRouter with default settings:

```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --personality "Extroversion" \
  --output "./results/extroversion"
```

### Custom Model Selection

Use a specific OpenRouter model:

```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-model "meta-llama/llama-3.1-405b-instruct" \
  --personality "Machiavellianism" \
  --output "./results/mach"
```

### Parallel Processing

Speed up generation with multiple workers:

```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-max-workers 4 \
  --personality "Extroversion" \
  --output "./results/fast" \
  --limit-samples 1000
```

### Using Environment Variable

If `OPENROUTER_API_KEY` is set in environment:

```bash
psyctl dataset.build.steer \
  --openrouter-api-key $env:OPENROUTER_API_KEY \
  --personality "Extroversion" \
  --output "./results/extroversion"
```

### Command-Line Options

#### OpenRouter-Specific Options

- `--openrouter-api-key`: OpenRouter API key (format: sk-or-v1-xxxx)
  - Required for OpenRouter mode
  - Can be set via environment variable

- `--openrouter-model`: Model identifier on OpenRouter
  - Default: `qwen/qwen3-next-80b-a3b-instruct`
  - Examples: `meta-llama/llama-3.1-405b-instruct`, `google/gemini-2.5-flash-preview-09-2025`

- `--openrouter-max-workers`: Number of parallel workers
  - Default: `1` (sequential)
  - Range: 1-10 (higher = more parallel requests)
  - Recommended: 4-8 for optimal performance

#### Standard Options (Same as Local Mode)

- `--personality`: Target personality trait (required)
- `--output`: Output directory path (required)
- `--dataset`: Hugging Face dataset name (default: allenai/soda)
- `--limit-samples`: Maximum samples to generate (default: unlimited)

## Performance Optimization

### Parallel Workers Configuration

The `--openrouter-max-workers` option controls how many API requests run in parallel:

**Sequential (Default)**
```bash
--openrouter-max-workers 1
```
- One request at a time
- Slowest but most reliable
- Good for testing

**Moderate Parallelism**
```bash
--openrouter-max-workers 4
```
- 4 requests in parallel
- Balanced speed and reliability
- Recommended for production

**High Parallelism**
```bash
--openrouter-max-workers 8
```
- 8 requests in parallel
- Fastest generation
- May hit rate limits

### Performance Comparison

| Workers | Speed Multiplier | Use Case |
|---------|-----------------|----------|
| 1 | 1x (baseline) | Testing, debugging |
| 4 | ~3.5x faster | Production use |
| 8 | ~6x faster | Large datasets |

### Recommended Settings

**For Small Datasets (< 100 samples)**
```bash
--openrouter-max-workers 1
--limit-samples 100
```

**For Medium Datasets (100-1000 samples)**
```bash
--openrouter-max-workers 4
--limit-samples 1000
```

**For Large Datasets (1000+ samples)**
```bash
--openrouter-max-workers 8
--limit-samples 5000
```

## Cost Management

### Cost Tracking

PSYCTL automatically tracks OpenRouter API costs:

```
2025-10-07 13:13:45,227 - dataset_builder - INFO - Total OpenRouter requests: 10
2025-10-07 13:13:45,227 - dataset_builder - INFO - Total OpenRouter cost: $0.000123
```

### Cost Estimation

Typical costs per sample (approximate):

| Model | Cost per Sample | 1000 Samples |
|-------|----------------|--------------|
| qwen/qwen3-next-80b-a3b-instruct | $0.000015 | $0.015 |
| meta-llama/llama-3.1-405b-instruct | $0.000050 | $0.050 |
| google/gemini-2.5-flash-preview | $0.000008 | $0.008 |

### Cost Optimization Tips

1. **Start Small**: Test with `--limit-samples 10` first
2. **Use Efficient Models**: Start with smaller models for testing
3. **Monitor Costs**: Check logs for cost tracking
4. **Set Limits**: Use `--limit-samples` to cap costs

## Supported Models

### Recommended Models

**Best Performance/Cost Ratio**
```bash
--openrouter-model "qwen/qwen3-next-80b-a3b-instruct"
```

**Highest Quality**
```bash
--openrouter-model "meta-llama/llama-3.1-405b-instruct"
```

**Fastest/Cheapest**
```bash
--openrouter-model "google/gemini-2.5-flash-lite-preview-09-2025"
```

### Model Comparison

| Model | Size | Speed | Cost | Quality |
|-------|------|-------|------|---------|
| qwen/qwen3-next-80b-a3b-instruct | 80B | Fast | Low | High |
| meta-llama/llama-3.1-405b-instruct | 405B | Slow | High | Highest |
| google/gemini-2.5-flash-preview | N/A | Fastest | Lowest | Good |

## Troubleshooting

### API Key Issues

**Error: OpenRouter API key is required**
```
ValueError: OpenRouter API key is required when use_openrouter=True
```

**Solution**: Provide API key via CLI or environment:
```bash
--openrouter-api-key "sk-or-v1-xxxx"
# OR
$env:OPENROUTER_API_KEY = "sk-or-v1-xxxx"
```

### Rate Limiting

**Error: Too many requests**
```
API Error: 429 - Rate limit exceeded
```

**Solution**: Reduce parallel workers:
```bash
--openrouter-max-workers 2  # Reduce from higher value
```

### Connection Timeout

**Error: Request timeout**
```
OpenRouter API request timeout
```

**Solution**:
1. Check internet connection
2. Retry with fewer workers
3. Use a different model

### Slow Performance

**Issue**: Generation takes too long

**Solution**:
1. Increase parallel workers:
   ```bash
   --openrouter-max-workers 8
   ```

2. Use faster model:
   ```bash
   --openrouter-model "google/gemini-2.5-flash-preview-09-2025"
   ```

3. Reduce batch size via environment:
   ```powershell
   $env:PSYCTL_INFERENCE_BATCH_SIZE = "8"
   ```

## Comparison: OpenRouter vs Local Model

### OpenRouter Mode

**Advantages:**
- No GPU required
- Access to large models (405B+)
- No model download time
- No disk space needed
- Easy model switching

**Disadvantages:**
- API costs per request
- Network dependency
- Rate limiting possible
- Slower for small datasets

### Local Model Mode

**Advantages:**
- No API costs
- No network dependency
- No rate limits
- Faster for small datasets

**Disadvantages:**
- Requires GPU
- Model download time
- Disk space required
- Limited to available GPU memory

### When to Use Each

**Use OpenRouter When:**
- No GPU available
- Need large models (70B+)
- Testing different models
- One-time dataset generation

**Use Local Model When:**
- GPU available
- Frequent dataset generation
- Large-scale production
- Cost is a concern

## Examples

### Example 1: Quick Test
```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --personality "Extroversion" \
  --output "./results/test" \
  --limit-samples 10
```

### Example 2: Production Dataset
```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-model "qwen/qwen3-next-80b-a3b-instruct" \
  --openrouter-max-workers 4 \
  --personality "Extroversion" \
  --output "./results/production" \
  --limit-samples 5000
```

### Example 3: High-Speed Generation
```bash
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-model "google/gemini-2.5-flash-preview-09-2025" \
  --openrouter-max-workers 8 \
  --personality "Machiavellianism" \
  --output "./results/fast" \
  --limit-samples 1000
```

### Example 4: Multiple Personalities
```bash
# Extroversion
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-max-workers 4 \
  --personality "Extroversion" \
  --output "./results/extroversion" \
  --limit-samples 1000

# Introversion
psyctl dataset.build.steer \
  --openrouter-api-key "sk-or-v1-xxxx" \
  --openrouter-max-workers 4 \
  --personality "Introversion" \
  --output "./results/introversion" \
  --limit-samples 1000
```

## Best Practices

1. **Start Small**: Always test with 10-100 samples first
2. **Monitor Costs**: Check logs for cost tracking
3. **Use Parallel Workers**: 4-8 workers for optimal speed
4. **Choose Right Model**: Balance quality, speed, and cost
5. **Set Limits**: Use `--limit-samples` to control costs
6. **Log Everything**: Keep logs for cost analysis

## References

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [PSYCTL Dataset Building Guide](./DATASET.BUILD.CAA.md)
- [Supported Models List](https://openrouter.ai/models)
