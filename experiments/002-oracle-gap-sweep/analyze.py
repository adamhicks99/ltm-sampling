#!/usr/bin/env python3
"""Oracle-gap analysis for the intelligent-sampling question.

Reads the paired comparisons produced by ``ltm-sampling`` and quantifies how
much predictive quality a single fixed sampling policy leaves on the table
relative to choosing the best policy per dataset (the hindsight oracle), under
a fixed latency target.

Quality is expressed as *retention* relative to the full-context baseline so it
is comparable across datasets and problem types:

* classification: ``accuracy / baseline_accuracy``
* regression:     ``baseline_rmse / rmse``

Retention 1.0 means parity with full context; > 1.0 means the sampled context
beat it (possible when pruning removes noise); < 1.0 means quality was lost.

A condition is *admissible* when its mean prediction speedup meets the latency
target (default 1.5x, matching experiment 001). The full-context baseline is
always available as the "do not sample" fallback (retention 1.0, speedup 1.0x).

Definitions
-----------
* per-dataset oracle: the admissible sampled policy with the highest retention;
  if none is admissible, the oracle falls back to full context.
* best fixed policy: the single sampler that is admissible on every dataset and
  maximizes mean retention across the portfolio. If no sampler is admissible
  everywhere, that is itself reported (no cookie-cutter policy clears the bar).
* intelligence gap: mean over datasets of (oracle retention - best-fixed
  retention on that dataset). This is the headline "value of adaptivity".

Usage
-----
    python analyze.py [results/raw_comparisons.csv] [--speedup-target 1.5]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
DEFAULT_COMPARISONS = HERE / "results" / "raw_comparisons.csv"


class Condition(NamedTuple):
    dataset: str
    sampler: str
    method: str
    fraction: float
    problem_type: str
    retention: float
    predict_speedup: float


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _retention(row: dict[str, str]) -> float | None:
    """Quality retained relative to the full-context baseline (1.0 = parity)."""
    if row["problem_type"] == "classification":
        value = _to_float(row.get("accuracy"))
        baseline = _to_float(row.get("baseline_accuracy"))
        if value is None or not baseline:
            return None
        return value / baseline
    value = _to_float(row.get("rmse"))
    baseline = _to_float(row.get("baseline_rmse"))
    if not value or baseline is None:
        return None
    return baseline / value


def load_conditions(path: Path) -> list[Condition]:
    """Collapse the per-seed comparison rows into per-(dataset, sampler) means."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    retentions: dict[tuple[str, str], list[float]] = defaultdict(list)
    speedups: dict[tuple[str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str], tuple[str, float, str]] = {}
    for row in rows:
        key = (row["dataset"], row["sampler"])
        retention = _retention(row)
        speedup = _to_float(row.get("predict_speedup"))
        if retention is None or speedup is None:
            continue
        retentions[key].append(retention)
        speedups[key].append(speedup)
        meta[key] = (
            row["sampler_method"],
            float(row["requested_fraction"]),
            row["problem_type"],
        )

    conditions = []
    for key, retention_values in retentions.items():
        dataset, sampler = key
        method, fraction, problem_type = meta[key]
        conditions.append(
            Condition(
                dataset=dataset,
                sampler=sampler,
                method=method,
                fraction=fraction,
                problem_type=problem_type,
                retention=statistics.fmean(retention_values),
                predict_speedup=statistics.fmean(speedups[key]),
            )
        )
    return conditions


class OracleRow(NamedTuple):
    dataset: str
    oracle_sampler: str
    oracle_retention: float
    oracle_speedup: float
    best_fixed_sampler: str
    best_fixed_retention_here: float
    gap: float


def analyze(conditions: list[Condition], target: float) -> None:
    datasets = sorted({c.dataset for c in conditions})
    by_dataset: dict[str, list[Condition]] = defaultdict(list)
    for c in conditions:
        by_dataset[c.dataset].append(c)

    sampled = [c for c in conditions if c.method != "full"]
    samplers = sorted({c.sampler for c in sampled})

    # Per-dataset oracle: best admissible sampled policy, else full fallback.
    oracle: dict[str, Condition | None] = {}
    for dataset in datasets:
        admissible = [
            c
            for c in by_dataset[dataset]
            if c.method != "full" and c.predict_speedup >= target
        ]
        oracle[dataset] = (
            max(admissible, key=lambda c: c.retention) if admissible else None
        )

    # Fixed policies: coverage and mean retention across the whole portfolio.
    lookup = {(c.dataset, c.sampler): c for c in conditions}
    policy_rows = []
    for sampler in samplers:
        present = [lookup[(d, sampler)] for d in datasets if (d, sampler) in lookup]
        if len(present) != len(datasets):
            continue  # sampler not run on every dataset; cannot be a global policy
        coverage = sum(c.predict_speedup >= target for c in present) / len(datasets)
        mean_retention = statistics.fmean(c.retention for c in present)
        policy_rows.append((sampler, coverage, mean_retention))

    fully_admissible = [p for p in policy_rows if p[1] == 1.0]
    if fully_admissible:
        best_fixed = max(fully_admissible, key=lambda p: p[2])[0]
        best_fixed_clears = True
    else:
        # No cookie-cutter policy meets the latency target everywhere.
        best_fixed = (
            max(policy_rows, key=lambda p: (p[1], p[2]))[0] if policy_rows else ""
        )
        best_fixed_clears = False

    # Assemble the oracle-vs-fixed comparison per dataset.
    oracle_rows = []
    for dataset in datasets:
        oc = oracle[dataset]
        oracle_sampler = oc.sampler if oc else "full (no-sample)"
        oracle_ret = oc.retention if oc else 1.0
        oracle_spd = oc.predict_speedup if oc else 1.0
        fixed = lookup.get((dataset, best_fixed)) if best_fixed else None
        fixed_ret = fixed.retention if fixed else 1.0
        oracle_rows.append(
            OracleRow(
                dataset=dataset,
                oracle_sampler=oracle_sampler,
                oracle_retention=oracle_ret,
                oracle_speedup=oracle_spd,
                best_fixed_sampler=best_fixed or "(none)",
                best_fixed_retention_here=fixed_ret,
                gap=oracle_ret - fixed_ret,
            )
        )

    _report(
        target,
        datasets,
        conditions,
        policy_rows,
        best_fixed,
        best_fixed_clears,
        oracle_rows,
    )
    _write_outputs(conditions, oracle_rows)


def _report(
    target: float,
    datasets: list[str],
    conditions: list[Condition],
    policy_rows: list[tuple[str, float, float]],
    best_fixed: str,
    best_fixed_clears: bool,
    oracle_rows: list[OracleRow],
) -> None:
    print(f"\nLatency target: prediction speedup >= {target:g}x")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}\n")

    print("Per-(dataset, sampler) mean retention [speedup]  (* = meets target):")
    for dataset in datasets:
        cells = sorted(
            (c for c in conditions if c.dataset == dataset and c.method != "full"),
            key=lambda c: (c.method, c.fraction),
        )
        parts = [
            f"{c.sampler}={c.retention:.3f}[{c.predict_speedup:.2f}x]"
            f"{'*' if c.predict_speedup >= target else ' '}"
            for c in cells
        ]
        print(f"  {dataset}:")
        for part in parts:
            print(f"      {part}")

    print("\nFixed-policy portfolio scan (admissible on every dataset?):")
    for sampler, coverage, mean_retention in sorted(
        policy_rows, key=lambda p: (-p[1], -p[2])
    ):
        flag = "clears" if coverage == 1.0 else f"{coverage:.0%} of datasets"
        print(f"  {sampler:<14} mean retention {mean_retention:.3f}  ({flag})")

    if best_fixed_clears:
        print(f"\nBest fixed policy clearing the target everywhere: {best_fixed}")
    else:
        print(
            "\nNo single sampler meets the latency target on every dataset — "
            f"strongest partial policy is {best_fixed}."
        )

    print("\nOracle vs best fixed policy, per dataset:")
    for r in oracle_rows:
        print(
            f"  {r.dataset:<32} oracle {r.oracle_sampler:<16} "
            f"ret {r.oracle_retention:.3f} ({r.oracle_speedup:.2f}x) | "
            f"fixed {r.best_fixed_retention_here:.3f} | gap {r.gap:+.3f}"
        )

    mean_gap = statistics.fmean(r.gap for r in oracle_rows)
    mean_oracle = statistics.fmean(r.oracle_retention for r in oracle_rows)
    winners = {r.oracle_sampler for r in oracle_rows}
    forced_full = sum(r.oracle_sampler.startswith("full") for r in oracle_rows)

    print("\n" + "=" * 60)
    print(f"Mean oracle retention:        {mean_oracle:.4f}")
    print(f"Intelligence gap (mean):      {mean_gap:+.4f}")
    print(f"Distinct oracle winners:      {len(winners)}  -> {sorted(winners)}")
    print(f"Datasets forced to no-sample: {forced_full}/{len(oracle_rows)}")
    print("=" * 60)
    verdict = (
        "adaptivity pays: no fixed policy matches per-dataset choice"
        if len(winners) > 1 or mean_gap > 0.005
        else "cookie-cutter looks adequate: one policy ~= the oracle"
    )
    print(f"Read: {verdict}\n")


def _write_outputs(conditions: list[Condition], oracle_rows: list[OracleRow]) -> None:
    (HERE / "results").mkdir(exist_ok=True)
    summary = HERE / "results" / "condition_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["dataset", "sampler", "method", "fraction", "problem_type",
             "mean_retention", "mean_predict_speedup"]
        )
        for c in sorted(conditions, key=lambda c: (c.dataset, c.method, c.fraction)):
            writer.writerow(
                [c.dataset, c.sampler, c.method, c.fraction, c.problem_type,
                 f"{c.retention:.6f}", f"{c.predict_speedup:.6f}"]
            )

    gap_path = HERE / "results" / "oracle_gap.csv"
    with gap_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(OracleRow._fields)
        for r in oracle_rows:
            writer.writerow(
                [r.dataset, r.oracle_sampler, f"{r.oracle_retention:.6f}",
                 f"{r.oracle_speedup:.6f}", r.best_fixed_sampler,
                 f"{r.best_fixed_retention_here:.6f}", f"{r.gap:+.6f}"]
            )
    print(f"Wrote {summary.relative_to(HERE)} and {gap_path.relative_to(HERE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparisons",
        nargs="?",
        type=Path,
        default=DEFAULT_COMPARISONS,
        help="Paired comparisons CSV (default: results/raw_comparisons.csv)",
    )
    parser.add_argument("--speedup-target", type=float, default=1.5)
    args = parser.parse_args()

    if not args.comparisons.exists():
        raise SystemExit(
            f"No comparisons file at {args.comparisons}. Run the benchmark first:\n"
            "  uv run --no-sync ltm-sampling run "
            "--config experiments/002-oracle-gap-sweep/config.toml"
        )

    conditions = load_conditions(args.comparisons)
    if not conditions:
        raise SystemExit("No usable conditions found in the comparisons file.")
    analyze(conditions, args.speedup_target)


if __name__ == "__main__":
    main()
