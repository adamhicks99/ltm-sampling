# Experiment 002: Oracle gap — the value of per-dataset sampling

## Question

Does the best training-context sampling policy change from dataset to dataset,
and if so, how much predictive quality does a single fixed policy leave on the
table versus choosing the best policy per dataset? In short: is *adaptive*
("intelligent") context sampling worth building, or is one cookie-cutter policy
good enough?

## Hypothesis

No single fixed `(method, fraction)` policy is optimal across datasets. Under a
fixed latency target (prediction speedup ≥ 1.5×):

1. the per-dataset oracle is achieved by **different** samplers on different
   datasets; and
2. the oracle retains **measurably more** quality than the best single fixed
   policy — an intelligence gap > 0.005 mean retention.

Experiment 001 already showed the same 50% KNN sampler failing on *opposite*
criteria across two datasets (accuracy-preserving but slow on diabetes; fast
but 6.5 pp worse on maternal_health_risk). This experiment tests whether that
divergence generalizes and quantifies it.

## Definitions

**Retention** — quality relative to the full-context baseline, normalized so it
is comparable across datasets and problem types:

- classification: `accuracy / baseline_accuracy`
- regression: `baseline_rmse / rmse`

Retention `1.0` means parity with full context; `> 1.0` means the sampled
context beat it (possible when pruning removes noise); `< 1.0` means quality was
lost.

**Latency target** — a condition is *admissible* when its mean prediction
speedup is ≥ 1.5× (experiment 001's threshold). The full-context baseline is
always available as the "do not sample" fallback (retention `1.0`, speedup
`1.0×`).

**Per-dataset oracle** — the admissible sampled policy with the highest
retention; the full fallback if no sampler is admissible.

**Best fixed policy** — the single sampler admissible on *every* dataset that
maximizes mean retention across the portfolio. If none is admissible everywhere,
that is itself a result: no cookie-cutter policy clears the bar.

**Intelligence gap** — mean over datasets of
`(oracle retention − best-fixed retention on that dataset)`. This is the
headline "value of adaptivity."

## Protocol

- Model: TabFM v1.0.0, PyTorch backend, default estimator preset.
- Ensemble members: 4, matching experiment 001's tractable setting. This is a
  deliberate departure from TabFM's default 32; the interaction between ensemble
  size and sampling sensitivity is a follow-up, not part of this run.
- Device: automatic CUDA selection, otherwise CPU.
- Seeds: 0 and 1. Sampling and ensemble randomness share the seed. Two seeds
  reduce sampler/ensemble noise; the OpenML split is fixed, so seed variance
  does **not** cover split variance (a stated limitation, see follow-ups).
- OpenML split: repeat 0, fold 0, sample 0 for every dataset.
- Timing: cold per-condition prediction; model and dataset loading excluded.
- Primary metric: accuracy (classification) / RMSE (regression). The runner also
  records balanced accuracy, macro F1, log loss, ROC AUC (classification) and
  MAE, R² (regression); retention is derived from the primary metric.

### Samplers

A full-context baseline plus a fraction sweep. `knn` omits 0.25 because
`KNNRedundancySampler` currently requires `fraction ≥ 0.5`.

| Method | Fractions |
|---|---|
| full | — |
| random | 0.25, 0.50, 0.75 |
| stratified | 0.25, 0.50, 0.75 |
| knn | 0.50, 0.75 |

### Datasets

Four small TabArena tasks spanning binary/multiclass classification and
regression, chosen so the winning policy has room to differ by data type. Exact
train/test row counts are recorded per condition in `results/raw.jsonl`.

| Dataset | OpenML task | Problem |
|---|---:|---|
| diabetes | 363629 | binary classification |
| maternal_health_risk | 363685 | 3-class classification |
| concrete_compressive_strength | 363625 | regression |
| airfoil_self_noise | 363612 | regression |

Conditions: 4 datasets × 9 samplers × 2 seeds = **72**.

## Run

```bash
uv sync --extra pytorch --group dev --no-editable
uv run --no-sync ltm-sampling run \
  --config experiments/002-oracle-gap-sweep/config.toml
```

Writes `results/raw.jsonl` and `results/raw_comparisons.csv`.

## Analyze

The oracle-gap computation is not part of the shared `ltm-sampling` tooling; it
lives here because the metric is still being figured out. It reads only the
paired comparisons and has no third-party dependencies.

```bash
python experiments/002-oracle-gap-sweep/analyze.py
# sweep the latency target:
python experiments/002-oracle-gap-sweep/analyze.py --speedup-target 2.0
```

Prints the report and writes `results/condition_summary.csv` (per dataset ×
sampler retention and speedup) and `results/oracle_gap.csv` (oracle vs best
fixed policy per dataset, and the gap).

## Acceptance / decision rule

- **Adaptivity is worthwhile** (supports the "not cookie cutter" thesis) if the
  oracle's winning sampler differs across datasets, or the intelligence gap is
  ≥ 0.005 mean retention. → motivates a dispatcher experiment (003) that picks
  method and fraction from cheap dataset diagnostics.
- **Cookie-cutter is adequate** if one policy is admissible on every dataset and
  stays within 0.005 retention of the oracle on each. → adopt that policy and
  deprioritize adaptive sampling.

## Runtime note

On CPU with 4 ensemble members, the full 72-condition matrix takes on the order
of an hour; full-context conditions dominate the cost. Trim datasets, seeds, or
fractions for a faster first pass — but a material change to the dataset suite,
seed policy, or sampler set is a new experiment number, not an edit here.
