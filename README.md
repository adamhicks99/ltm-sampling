# LTM Sampling

An open research project for measuring whether training-context sampling can
improve the predictive-performance/latency tradeoff of large tabular models.
The first model integration targets Google Research's
[TabFM](https://github.com/google-research/tabfm).

## Experiment contract

Every experiment is a paired comparison:

1. Load an official OpenML task split.
2. Keep its test fold unchanged.
3. evaluate TabFM with the full training context;
4. apply a sampler to the training fold only;
5. evaluate the same TabFM configuration and seed on the sampled context; and
6. report quality deltas, retained rows, and runtime speedups.

The model weights are loaded once per problem type and reused. Model-loading
time is recorded but excluded from the paired total. Sampling, TabFM
preprocessing (`fit`), and prediction are timed separately. Condition order is
randomized within each dataset/seed pair to reduce fixed warm-cache bias.

TabFM performs in-context learning rather than parameter training on each
dataset. Its `fit` step prepares encoders and context views; the selected
training rows are consumed during prediction. This makes context sampling a
direct latency and quality intervention.

## Metrics

Classification runs record accuracy, balanced accuracy, macro F1, log loss,
and ROC AUC. Regression runs record RMSE, MAE, and R². The paired report adds:

- `quality_delta`: sampled accuracy minus full accuracy for classification, or
  full RMSE minus sampled RMSE for regression. Positive is always better.
- `predict_speedup`: full prediction time divided by sampled prediction time.
- `total_speedup`: full measured time divided by sampled measured time.

Raw records also contain dataset split coordinates, seed, execution order,
TabFM revision, model settings, row counts, each timing component, and a
hardware/software fingerprint.

## Experiment archive

Tracked experiment protocols, configurations, results, and notes live under
[`experiments/`](experiments/README.md). A completed experiment is immutable;
material methodology changes receive a new numbered folder so conclusions stay
connected to the exact evidence that produced them.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/). The PyTorch
backend is the default:

```bash
uv sync --extra pytorch --group dev --no-editable
```

The first real run downloads TabFM weights from Hugging Face. TabFM is pinned
to commit `5ee6cd7829b5a4fdfd7e2a266259df733d40d036` because its repository does
not currently publish an immutable `v1.0.0` Git tag.

For JAX, install the alternative extra and set `backend = "jax"` in the TOML
configuration:

```bash
uv sync --extra jax --group dev --no-editable
```

## Run

Inspect the six-condition smoke-test matrix without downloading data or model
weights:

```bash
uv run --no-sync ltm-sampling plan --config configs/quickstart.toml
```

Run it and generate paired comparisons:

```bash
uv run --no-sync ltm-sampling run --config configs/quickstart.toml
```

This writes:

- `outputs/quickstart.jsonl`: one auditable record per condition;
- `outputs/quickstart_comparisons.csv`: full-versus-sampled deltas and speedups.

`configs/benchmark.toml` expands the matrix to six small TabArena datasets,
five context strategies, and three seeds. It uses TabFM's standard 32-member
ensemble count and is intended for a substantive run, not a quick smoke test.

## Add a sampling technique

Implement the `Sampler` protocol in
`src/ltm_sampling/samplers.py`. A sampler receives `X`, `y`, the problem type,
and a random seed, then returns unique positional training-row indices. Register
the method in `build_sampler`, add it to a TOML file, and the existing runner
will apply the same controls, metrics, timing, and paired reporting. Re-run the
setup command with `--reinstall-package ltm-sampling` after changing package
code so the non-editable command-line installation is refreshed.

The included methods are:

- `full`: unsampled paired baseline;
- `random`: uniform sampling without replacement;
- `stratified`: class-preserving sampling for classification and target-quantile
  sampling for regression.
- `knn`: nearest-neighbor redundancy pruning with class or regression-quantile
  preservation.

## Reproducibility boundaries

- OpenML task split coordinates are fixed and recorded.
- Dataset download/loading and model-weight loading are excluded from paired
  totals; subset materialization is included in sampling time.
- Sampling and TabFM ensemble randomness share the configured seed.
- TabFM's internal `max_num_rows` sampling is forcibly disabled because it
  would confound the external sampler.
- Accelerator work is synchronized at timing boundaries.
- `warmup_runs = 0` measures cold per-condition inference. Increase it to
  compare steady-state prediction after an unmeasured warmup policy; warmup
  time remains recorded separately.
- GPU timing should be compared on the same otherwise-idle machine. Do not mix
  hardware types in one aggregate result.

## Development

```bash
uv sync --group dev --no-editable
uv run --no-sync ruff check .
uv run --no-sync pytest
```

Licensed under the [MIT License](LICENSE).
