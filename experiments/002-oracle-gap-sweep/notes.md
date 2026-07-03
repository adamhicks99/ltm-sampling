# Experiment notes

## Status

Scaffolded, not yet run. Protocol, config, and analysis script are in place and
the analysis script is validated against synthetic fixtures; the benchmark
matrix has not been executed.

## Run log

- Pending execution.

## Results

Filled by `analyze.py` after the run. Headline numbers to record:

| Quantity | Value |
|---|---|
| Mean oracle retention | — |
| Intelligence gap (mean) | — |
| Distinct oracle winners | — |
| Datasets forced to no-sample | — |
| Best fixed policy | — |

Per-dataset oracle vs best fixed policy (from `results/oracle_gap.csv`):

| Dataset | Oracle sampler | Oracle retention | Speedup | Fixed retention here | Gap |
|---|---|---:|---:|---:|---:|
| diabetes | — | — | — | — | — |
| maternal_health_risk | — | — | — | — | — |
| concrete_compressive_strength | — | — | — | — | — |
| airfoil_self_noise | — | — | — | — | — |

## Observations

Pending execution.

## Decision

Pending execution.

## Follow-ups

- If adaptivity pays: build experiment 003 as a dispatcher — predict the oracle
  `(method, fraction)` from cheap dataset diagnostics (size, class imbalance,
  redundancy spectrum, noise estimate) and measure how much of the oracle gap it
  recovers.
- Add fold variation, not just seed variation, so the reported gap reflects
  split noise. The current design fixes the OpenML split.
- Revisit with TabFM's default 32-member ensemble to confirm the ranking of
  samplers is not an artifact of the reduced 4-member ensemble.
- Extend the KNN sampler below fraction 0.5 so it can compete in the aggressive
  25% column instead of being absent there.
- Add larger TabArena tasks (splice, wine_quality) once runtime on CPU or a GPU
  is validated; they were excluded here to keep the matrix tractable.
