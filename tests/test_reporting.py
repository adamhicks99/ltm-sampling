import csv
import json

import numpy as np

from ltm_sampling.reporting import summarize_results


def test_summary_pairs_sampled_run_with_full_baseline(tmp_path):
    raw = tmp_path / "results.jsonl"
    common = {
        "status": "success",
        "dataset": "demo",
        "repeat": 0,
        "fold": 0,
        "sample": 0,
        "seed": 1,
        "problem_type": "classification",
        "requested_fraction": 1.0,
        "n_train_full": 100,
        "sampling_seconds": 1.0,
        "fit_seconds": 2.0,
    }
    baseline = {
        **common,
        "sampler": "full",
        "sampler_method": "full",
        "retained_fraction": 1.0,
        "n_train_sampled": 100,
        "metric_accuracy": 0.8,
        "predict_seconds": 8.0,
        "measured_total_seconds": 11.0,
    }
    sampled = {
        **common,
        "sampler": "random-50",
        "sampler_method": "random",
        "requested_fraction": 0.5,
        "retained_fraction": 0.5,
        "n_train_sampled": 50,
        "metric_accuracy": 0.79,
        "predict_seconds": 4.0,
        "measured_total_seconds": 7.0,
    }
    raw.write_text(
        "\n".join(json.dumps(record) for record in (baseline, sampled)) + "\n",
        encoding="utf-8",
    )

    summary = summarize_results(raw)
    with summary.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    sampled_row = next(row for row in rows if row["sampler"] == "random-50")
    assert np.isclose(float(sampled_row["accuracy_delta"]), -0.01)
    assert float(sampled_row["predict_speedup"]) == 2.0
