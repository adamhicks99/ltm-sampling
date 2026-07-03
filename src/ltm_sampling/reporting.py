"""Create paired comparisons from raw JSONL benchmark records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PAIR_KEYS = ("dataset", "repeat", "fold", "sample", "seed")


def summarize_results(
    results_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(results_path)
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successful = [record for record in records if record["status"] == "success"]
    baselines = {
        _pair_key(record): record
        for record in successful
        if record["sampler_method"] == "full"
    }

    comparisons = []
    for record in successful:
        baseline = baselines.get(_pair_key(record))
        if baseline is None:
            continue
        comparison = {
            key: record[key]
            for key in (
                *PAIR_KEYS,
                "problem_type",
                "sampler",
                "sampler_method",
                "requested_fraction",
                "retained_fraction",
                "n_train_full",
                "n_train_sampled",
            )
        }
        comparison.update(
            {
                "accuracy": record.get("metric_accuracy"),
                "baseline_accuracy": baseline.get("metric_accuracy"),
                "accuracy_delta": _difference(
                    record.get("metric_accuracy"),
                    baseline.get("metric_accuracy"),
                ),
                "rmse": record.get("metric_rmse"),
                "baseline_rmse": baseline.get("metric_rmse"),
                "rmse_delta": _difference(
                    record.get("metric_rmse"),
                    baseline.get("metric_rmse"),
                ),
                "quality_delta": _quality_delta(record, baseline),
                "sampling_seconds": record["sampling_seconds"],
                "fit_seconds": record["fit_seconds"],
                "predict_seconds": record["predict_seconds"],
                "measured_total_seconds": record["measured_total_seconds"],
                "predict_speedup": _ratio(
                    baseline["predict_seconds"], record["predict_seconds"]
                ),
                "total_speedup": _ratio(
                    baseline["measured_total_seconds"],
                    record["measured_total_seconds"],
                ),
            }
        )
        comparisons.append(comparison)

    destination = (
        Path(output_path)
        if output_path
        else source.with_name(f"{source.stem}_comparisons.csv")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if comparisons:
        with destination.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    else:
        destination.write_text("", encoding="utf-8")
    return destination


def _pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[key] for key in PAIR_KEYS)


def _difference(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _quality_delta(
    record: dict[str, Any],
    baseline: dict[str, Any],
) -> float | None:
    if record["problem_type"] == "classification":
        return _difference(
            record.get("metric_accuracy"),
            baseline.get("metric_accuracy"),
        )
    # Positive always means the sampled condition improved predictive quality.
    return _difference(
        baseline.get("metric_rmse"),
        record.get("metric_rmse"),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None
