# Troubleshooting Guide

This document provides solutions to common issues you may encounter when using PSYCTL.

## Padding-Related Issues

### Understanding Padding in Batch Processing

When processing multiple prompts in a batch, the tokenizer pads shorter sequences to match the longest sequence in the batch. Different models use different padding directions:

- **Left padding**: `[PAD, PAD, Token1, Token2]` (used by Gemma, GPT-2)
- **Right padding**: `[Token1, Token2, PAD, PAD]` (used by T5, BERT)

### How PSYCTL Handles Padding

**Good news**: PSYCTL automatically handles both padding directions correctly!

The activation extraction system uses **attention masks** to identify the last real (non-padding) token in each sequence. This ensures correct steering vector extraction regardless of which model you use.

#### Technical Details

1. **With attention mask** (automatic):
   - Left-padded: Finds last real token correctly
   - Right-padded: Finds last real token correctly
   - No padding: Works as expected

2. **Without attention mask** (fallback):
   - Uses position -1 (may be incorrect for right-padded models)
   - Logs a warning message
   - Only occurs if internal APIs are used incorrectly

### Symptom: Warning "No attention mask set"

**Warning message**:
```
WARNING: No attention mask set for layer 'model.layers[13].mlp.down_proj'.
Using position -1 (may be incorrect for right-padded models)
```

**What it means**:
The activation collection system could not determine which tokens are real vs padding.

**Why it matters**:
Without attention masks, the system falls back to using position -1, which is:
- Safe for left-padded models (Gemma, GPT-2)
- Potentially incorrect for right-padded models (T5, BERT)

**Solution**:
This should not happen during normal usage. If you see this warning:

1. Make sure you're using the official CLI or public APIs
2. If using Python APIs directly, ensure you're not calling internal functions
3. Report the issue with reproduction steps at: https://github.com/anthropics/psyctl/issues

### Symptom: Poor Steering Performance with T5/BERT Models

**Problem**:
Steering vectors don't seem to work well with T5 or BERT-based models.

**Possible causes**:

1. **Incorrect layer selection**: T5 and BERT have different layer naming than Llama/Gemma
   - T5: `encoder.block[N].layer[1].DenseReluDense.wo`
   - BERT: `encoder.layer[N].intermediate.dense`
   - Llama/Gemma: `model.layers[N].mlp.down_proj`

2. **Model architecture differences**: Encoder-only models (BERT) work differently than decoder-only models (Llama, Gemma)

**Solution**:
- Use the layer inspection tools to find the correct layer paths
- Start with middle layers (e.g., layer 6 out of 12)
- Experiment with different layers to find what works best

### Symptom: High Memory Usage During Extraction

**Problem**:
Memory usage is very high during steering vector extraction.

**Causes**:
- Large batch size with high padding ratio
- Example: Batch size 32 with prompts ranging from 10 to 500 tokens

**Solution**:

1. Reduce batch size:
   ```bash
   export PSYCTL_INFERENCE_BATCH_SIZE=8
   ```

2. Sort prompts by length before batching (future feature)

3. Use a smaller model for initial experimentation

## Model Compatibility

### Supported Models

PSYCTL has been tested with:
- Gemma 2/3 series (270M, 2B, 9B, 27B)
- Llama 3.1/3.2 series
- Qwen series
- Other decoder-only causal LMs

### Known Limitations

1. **Encoder-only models** (BERT, RoBERTa):
   - Require different layer paths
   - May need different steering application strategies

2. **Encoder-decoder models** (T5, BART):
   - Complex architecture with separate encoder/decoder
   - Steering vectors may need to target specific components

3. **Very large models** (70B+):
   - May require device_map="auto" for multi-GPU
   - Batch size needs careful tuning

## Dataset Issues

### Symptom: "No JSONL files found"

**Error**:
```
FileNotFoundError: No JSONL files found in directory: ./dataset/caa
```

**Solution**:
1. Check that the dataset directory exists
2. Verify the file has `.jsonl` extension (not `.json`)
3. Use `ls` to confirm the file is in the expected location

### Symptom: "Invalid JSON format"

**Error**:
```
ValueError: Invalid JSON format at line 42
```

**Solution**:
1. Check that each line is valid JSON
2. Verify required fields: `question`, `positive`, `neutral`
3. Use a JSON validator to check the file

## Performance Issues

### Slow Extraction Speed

**Causes**:
- Batch size too small (underutilizing GPU)
- Batch size too large (memory swapping)
- Model on CPU instead of GPU

**Solutions**:

1. Check GPU usage:
   ```bash
   nvidia-smi
   ```

2. Optimize batch size (start with 16, adjust based on VRAM):
   ```bash
   export PSYCTL_INFERENCE_BATCH_SIZE=16
   ```

3. Ensure model loads on GPU (should see "cuda" in logs)

### Out of Memory (OOM)

**Error**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

1. Reduce batch size:
   ```bash
   export PSYCTL_INFERENCE_BATCH_SIZE=4
   ```

2. Use a smaller model:
   ```bash
   psyctl extract.steering --model "google/gemma-3-270m-it" ...
   ```

3. Clear GPU cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Getting Help

If you encounter issues not covered here:

1. Check the [documentation](https://docs.psyctl.ai)
2. Search [existing issues](https://github.com/anthropics/psyctl/issues)
3. Open a new issue with:
   - Error message and full traceback
   - Command or code that caused the error
   - Model name and configuration
   - System info (GPU, Python version, etc.)

## Debugging Tips

### Enable Debug Logging

```bash
export PSYCTL_LOG_LEVEL=DEBUG
```

This will show detailed information about:
- Tokenizer padding direction
- Attention mask shapes
- Activation collection statistics
- Layer validation results

### Verify Installation

```bash
psyctl --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Test with Minimal Example

Use the smallest model and smallest dataset to isolate issues:

```bash
# Quick test with tiny model
psyctl extract.steering \
  --model "google/gemma-3-270m-it" \
  --layer "model.layers[6].mlp.down_proj" \
  --dataset "path/to/small/dataset" \
  --output "./test.safetensors"
```
