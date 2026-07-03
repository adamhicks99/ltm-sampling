# Experiment notes

## Status

Complete. All four conditions finished with `status: success`; see
`results/raw.jsonl` and `results/raw_comparisons.csv`.

## Run log

- Initial execution downloaded the TabFM PyTorch checkpoint successfully but
  failed before the first condition because TabFM's package metadata omitted
  the `safetensors` runtime dependency. The project dependency was added
  explicitly and the run was restarted from the cached checkpoint.
- The 32-member default ensemble was then attempted on CPU. The first
  full-context diabetes condition remained compute-bound after 17 minutes and
  was stopped before producing a result. The experiment was revised to use 4
  ensemble members; all other controls remain unchanged.

## Results

| Dataset | Condition | Rows retained | Accuracy | Accuracy Δ (pp) | Prediction time (s) | Prediction speedup | Total speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| diabetes | full | 512 (100%) | 0.7656 | — | 131.82 | 1.00× | 1.00× |
| diabetes | knn-redundancy-50 | 256 (50%) | 0.7617 | −0.39 | 90.25 | 1.46× | 1.46× |
| maternal_health_risk | full | 676 (100%) | 0.8817 | — | 206.13 | 1.00× | 1.00× |
| maternal_health_risk | knn-redundancy-50 | 338 (50%) | 0.8166 | −6.51 | 107.92 | 1.91× | 1.91× |

Acceptance criteria (from the hypothesis): **≥1.5× prediction speedup** and
**accuracy change within 1 percentage point** of the full-context baseline.

| Dataset | ≥1.5× speedup | Δaccuracy ≤ 1 pp | Both met |
|---|---|---|---|
| diabetes | no (1.46×) | yes (−0.39 pp) | no |
| maternal_health_risk | yes (1.91×) | no (−6.51 pp) | no |

Secondary metrics (full → knn-50):

| Dataset | Balanced acc. | Macro F1 | Log loss | ROC AUC |
|---|---|---|---|---|
| diabetes | 0.7252 → 0.6942 (−3.1 pp) | 0.7326 → 0.7065 | 0.4584 → 0.4831 (worse) | 0.8470 → 0.8343 |
| maternal_health_risk | 0.8806 → 0.8080 (−7.3 pp) | 0.8843 → 0.8177 | 0.3252 → 0.4458 (worse) | 0.9722 → 0.9457 |

## Observations

- **Neither dataset satisfied both acceptance criteria at once.** diabetes
  preserved accuracy but fell short on speed; maternal_health_risk delivered the
  speedup but lost 6.5 points of accuracy. The two datasets miss on opposite
  criteria, so the joint target was not reached on this pilot.
- **Speedup scales with training-context size and is sublinear in the retained
  fraction.** Halving the context gave 1.91× on the larger maternal set
  (676 → 338 rows) but only 1.46× on the smaller diabetes set (512 → 256 rows),
  consistent with a fixed per-prediction overhead that does not shrink with the
  context. Total and prediction speedups are essentially identical because
  sampling and fit time are negligible (all < 0.13 s).
- **The diabetes accuracy delta is a single test row.** −0.0039 = −1/256, i.e.
  one of 256 test examples flipped. maternal's −0.0651 = −22/338, i.e. 22 test
  examples flipped. With one split and one seed these headline deltas carry
  little statistical weight, especially diabetes'.
- **Headline accuracy understates the quality cost.** Even on diabetes, where
  accuracy barely moved, balanced accuracy dropped 3.1 pp and log loss worsened,
  indicating the pruned context hurt minority-class and probability calibration
  more than the top-line number suggests. On maternal every secondary metric
  degraded materially.

## Decision

**Reject the hypothesis for this configuration.** 50% KNN redundancy pruning did
not simultaneously achieve a ≥1.5× prediction speedup and a ≤1 pp accuracy
change on either dataset. The result is not a flat negative, though: it points
to a size-dependent tradeoff — small datasets keep accuracy but do not shrink
enough to hit the speed target, while a larger dataset hits the speed target at
a real accuracy cost. A fixed 50% fraction is therefore the wrong lever to hold
constant across dataset sizes, and accuracy alone is too coarse a quality gate.
This warrants a follow-up experiment rather than adoption or abandonment.

## Follow-ups

- Run multiple seeds and folds regardless of this pilot's outcome: single-split
  deltas (diabetes' is literally one test row) are within noise and cannot
  support a conclusion on their own.
- Compare KNN redundancy pruning against uniform (`random`) and `stratified`
  sampling at the same 50% fraction to isolate whether redundancy-aware pruning
  beats naive subsampling.
- Sweep the retained fraction (e.g. 25 / 50 / 75%) per dataset to locate the
  size-dependent knee where speedup and accuracy cross the acceptance criteria.
- Add balanced accuracy and log loss to the decision gate; accuracy alone missed
  the calibration/minority-class degradation visible even on diabetes.
- Revisit with TabFM's default 32-member ensemble to confirm the effect is not
  an artifact of the reduced 4-member ensemble used to keep this pilot tractable.
- Expand to larger TabArena tasks after validating runtime behavior.
