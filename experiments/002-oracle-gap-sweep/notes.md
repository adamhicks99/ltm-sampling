# Experiment notes

## Status

Complete. All 72 conditions finished with `status: success` (0 failures).
Analysis produced by `analyze.py`; see `results/raw.jsonl`,
`results/raw_comparisons.csv`, `results/condition_summary.csv`, and
`results/oracle_gap.csv`.

## Run log

- Single background run on CPU (Apple arm64), TabFM PyTorch, 4-member ensemble.
- 72/72 conditions succeeded. Total compute 3.10 h (sum of per-condition
  workflow time); model load 18 s per problem type (excluded from paired
  totals). Slowest condition 588 s (airfoil full context), fastest 55 s.
- Resolved split sizes: diabetes 512/256, maternal_health_risk 676/338,
  concrete 686/344, airfoil 1002/501 (train/test).

## Results

Latency target: prediction speedup ≥ 1.5×. Retention is quality relative to
full context (1.0 = parity; > 1.0 beats full).

| Quantity | Value |
|---|---|
| Mean oracle retention | 0.880 |
| **Intelligence gap (mean)** | **+0.026** |
| Distinct oracle winners | 3 of 4 datasets |
| Datasets forced to no-sample | 0 / 4 |
| Best fixed policy (admissible everywhere) | knn-50 (mean retention 0.854) |

Per-dataset oracle vs best fixed policy:

| Dataset | Problem | Oracle sampler | Oracle retention | Speedup | knn-50 retention | Gap |
|---|---|---|---:|---:|---:|---:|
| diabetes | binary cls | stratified-25 | 1.031 | 2.42× | 0.992 | +0.039 |
| maternal_health_risk | 3-class cls | knn-50 | 0.928 | 1.74× | 0.928 | +0.000 |
| concrete_compressive_strength | regression | stratified-50 | 0.873 | 1.84× | 0.807 | +0.065 |
| airfoil_self_noise | regression | knn-50 | 0.690 | 1.81× | 0.690 | +0.000 |

Fixed policies admissible on every dataset (mean retention): knn-50 0.854,
random-50 0.807, stratified-25 0.759, random-25 0.725. Every 0.75-fraction
policy fails the latency target on all four datasets (speedups 1.0–1.4×), so no
gentle-pruning policy is portfolio-admissible.

## Observations

- **Hypothesis supported — sampling is not cookie-cutter.** The oracle is
  achieved by three different samplers across four datasets, and the best single
  fixed policy leaves +0.026 mean retention on the table — well above the 0.005
  decision threshold.
- **The gap is concentrated, not diffuse.** Two datasets (airfoil, maternal)
  have gap 0 because knn-50 happens to be their oracle; all of the value comes
  from concrete (+0.065) and diabetes (+0.039). Adaptivity pays *on specific
  datasets*, it does not uniformly lift every dataset.
- **knn-50 is rehabilitated relative to experiment 001.** With retention
  normalization and the full sweep it is the most robust single policy (best
  fixed, and oracle-optimal on 2/4). Any dispatcher in 003 must beat a knn-50
  baseline, not a naive-random one.
- **Problem type is the dominant axis.** Classification tolerates aggressive
  pruning — diabetes retention stays ≥ 0.98 everywhere and stratified-25
  *beats* full context (1.031) at 2.42×, a noise-removal effect confirmed on
  both seeds. Regression punishes it — airfoil's oracle retains only 0.690
  (RMSE 44% worse); its higher-quality option knn-75 (0.853) is not admissible
  (1.02×).
- **The latency target is itself a binding constraint, especially for
  regression.** Hitting 1.5× forces the retained fraction to ≤ 50%, which is
  exactly where regression quality collapses. For airfoil the strict target
  buys speed at a 31% quality loss; a Pareto (retention-vs-speedup) view or a
  relaxed target may be the more honest framing than a single cutoff.
- **A fixed fraction is not a fixed latency.** random-25 yields 2.95× on airfoil
  but 2.39× on diabetes; some 0.75 conditions are *below* 1.0× (maternal
  random-75 at 0.96×), where sampling overhead exceeds the context savings.
- **Admissibility hides seed variance.** concrete knn-50 straddles the target
  (seed 0 1.40×, seed 1 1.88×; mean 1.64× admits it). Two seeds is thin; some
  cells would flip admissibility under per-seed accounting.

## Decision

The "not cookie-cutter" thesis holds for this portfolio: the optimal
`(method, fraction)` varies by dataset and a single fixed policy forfeits
measurable quality (+0.026 mean, up to +0.065 on concrete) under a 1.5×
latency target. This justifies **experiment 003: a dispatcher** that predicts
the oracle policy from cheap dataset diagnostics.

Two tempering caveats the dispatcher must respect:

1. The realizable prize is modest (~2.6% mean retention) and concentrated on
   two datasets, so the dispatcher has to beat a strong knn-50 default rather
   than a random baseline.
2. The clearest signal is problem type plus a regression/redundancy sensitivity
   proxy — a v1 dispatcher keyed on those two features is the natural first cut,
   before anything more elaborate.

## Follow-ups

- **Experiment 003 (dispatcher):** predict oracle `(method, fraction)` from
  diagnostics (problem type, class balance, a redundancy/noise proxy, size);
  measure the fraction of the +0.026 gap recovered against a knn-50 baseline.
- Report a per-dataset retention-vs-speedup Pareto frontier and sweep
  `--speedup-target`; airfoil shows a single cutoff can force large quality loss
  and may misrepresent the real tradeoff.
- Use per-seed admissibility and ≥ 3 seeds; concrete knn-50 straddles 1.5×
  across the two seeds run here.
- Extend `KNNRedundancySampler` below fraction 0.5 so it can compete in the 25%
  column, where stratified-25 currently wins diabetes and knn is simply absent.
- Isolate the "beats full" effect (diabetes stratified-25 at 1.031) with the
  margin/edited-nearest-neighbor samplers — it looks like label-noise removal.
- Carry over from 001: add fold variation and revisit with the default
  32-member ensemble to confirm the sampler ranking is not an ensemble artifact.
- Add larger tasks (splice, wine_quality) once GPU or longer CPU budget is
  available; airfoil full context already cost ~10 min per condition here.
