# Experiment 001: 50% KNN redundancy pruning

## Question

Can a K-nearest-neighbor redundancy sampler remove half of TabFM's training
context while reducing end-to-end inference time without materially reducing
classification accuracy?

## Hypothesis

Retaining 50% of the training rows will produce at least a 1.5× prediction
speedup while keeping the accuracy change within one percentage point of the
full-context baseline.

## KNN sampling definition

This experiment uses `knn-redundancy-50`, not an external canonical algorithm:

1. Fit median imputation and standard scaling to numeric training columns.
2. Fit most-frequent imputation and one-hot encoding to categorical columns.
3. Build a 5-nearest-neighbor graph independently within each class.
4. Consider the most locally redundant rows first.
5. Remove a row only while protecting a nearby active representative.
6. Stop at exactly 50% of the original training-row count.

The transformation, graph construction, and subset materialization are all
included in `sampling_seconds`.

## Protocol

- Model: TabFM v1.0.0, PyTorch backend, default estimator preset.
- Ensemble members: 4. The default 32-member configuration was attempted first
  but did not complete one CPU condition after 17 minutes; reducing ensemble
  replication keeps this initial feedback-loop experiment tractable.
- Device: automatic CUDA selection, otherwise CPU.
- Model seed: 0.
- OpenML split: repeat 0, fold 0, sample 0.
- Conditions: full context and KNN 50% context.
- Timing: cold per-condition prediction; model and dataset loading excluded.
- Primary metric: classification accuracy.
- Secondary metrics: balanced accuracy, macro F1, log loss, and ROC AUC.

| Dataset | OpenML task | Type | Train rows | Test rows |
|---|---:|---|---:|---:|
| diabetes | 363629 | binary classification | 512 | 256 |
| maternal_health_risk | 363685 | 3-class classification | 676 | 338 |

## Run

```bash
uv sync --extra pytorch --group dev --no-editable
uv run --no-sync ltm-sampling run \
  --config experiments/001-knn-redundancy-50/config.toml
```

The run writes `results/raw.jsonl` and `results/raw_comparisons.csv` inside this
experiment folder.
