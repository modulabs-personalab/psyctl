# Layer Analysis

Not all layers are equally effective for steering. PSYCTL includes tools to analyze which layers provide the best separation between personality-exhibiting and neutral activations.

## CLI Usage

```bash
psyctl layer.analyze \
  --model "google/gemma-3-270m-it" \
  --layers "model.layers[*].mlp" \
  --dataset "./dataset/steering" \
  --method svm \
  --top-k 5
```

## Analysis Methods

### SVM Analyzer

Trains a linear SVM at each layer to classify positive vs neutral activations:

- **Score**: Overall separation quality (higher is better)
- **Accuracy**: Classification accuracy
- **Margin**: SVM margin (larger = more robust)

### Consensus Analyzer

Combines multiple analysis methods for more robust layer selection.

## Layer Patterns

Wildcard patterns for targeting groups of layers:

| Pattern | Matches |
|---------|---------|
| `model.layers[*].mlp` | All MLP layers |
| `model.layers[5:15].mlp` | Layers 5-14 |
| `model.layers[::2].mlp` | Every other layer |

## Interactive Notebook

Try the [05_layer_analysis notebook](https://colab.research.google.com/github/modulabs-personalab/psyctl/blob/main/examples/en/05_layer_analysis.ipynb) for a visual walkthrough with charts.
