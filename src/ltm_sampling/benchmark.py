"""Paired full-context versus sampled-context benchmark runner."""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from typing import Any, Protocol

import numpy as np

from ltm_sampling.config import BenchmarkConfig, ProblemType
from ltm_sampling.datasets import DatasetFold, load_openml_dataset
from ltm_sampling.metrics import classification_metrics, regression_metrics
from ltm_sampling.models import TABFM_REVISION, PreparedModel, TabFMFactory
from ltm_sampling.samplers import build_sampler


class ModelFactory(Protocol):
    def prepare(self, problem_type: ProblemType) -> PreparedModel: ...

    def create_estimator(self, problem_type: ProblemType, random_state: int) -> Any: ...

    def synchronize(self, value: Any | None = None) -> None: ...


def run_benchmark(
    config: BenchmarkConfig,
    *,
    dataset_loader: Callable[[Any], DatasetFold] = load_openml_dataset,
    model_factory: ModelFactory | None = None,
) -> list[dict[str, Any]]:
    """Run the configured experiment matrix and write one JSON object per run."""
    factory = model_factory or TabFMFactory(config.model)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()
    runtime_metadata = _runtime_metadata(config)
    output_path = config.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for dataset_spec in config.datasets:
            dataset = dataset_loader(dataset_spec)
            prepared = factory.prepare(dataset.problem_type)

            for seed in config.seeds:
                order_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, dataset.openml_task_id])
                )
                execution_order = order_rng.permutation(len(config.samplers))

                for order_index, sampler_index in enumerate(execution_order):
                    sampler_spec = config.samplers[int(sampler_index)]
                    base_result = {
                        "run_id": run_id,
                        "started_at_utc": started_at,
                        "status": "success",
                        "error": None,
                        "dataset": dataset.name,
                        "openml_task_id": dataset.openml_task_id,
                        "problem_type": dataset.problem_type,
                        "repeat": dataset.repeat,
                        "fold": dataset.fold,
                        "sample": dataset.sample,
                        "seed": seed,
                        "sampler": sampler_spec.name,
                        "sampler_method": sampler_spec.method,
                        "requested_fraction": sampler_spec.fraction,
                        "execution_order": int(order_index),
                        "n_train_full": len(dataset.X_train),
                        "n_test": len(dataset.X_test),
                        "n_features": dataset.n_features,
                        "backend": config.model.backend,
                        "device": prepared.device,
                        "model_preset": config.model.preset,
                        "n_estimators": config.model.n_estimators,
                        "tabfm_revision": TABFM_REVISION,
                        "model_load_seconds": prepared.load_seconds,
                        **runtime_metadata,
                    }
                    try:
                        result = _run_condition(
                            base_result=base_result,
                            config=config,
                            dataset=dataset,
                            sampler_spec=sampler_spec,
                            seed=seed,
                            factory=factory,
                        )
                    except Exception as exc:  # keep a long benchmark auditable
                        result = {
                            **base_result,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        if config.fail_fast:
                            _write_result(output_file, result)
                            raise

                    results.append(result)
                    _write_result(output_file, result)

    return results


def _run_condition(
    *,
    base_result: dict[str, Any],
    config: BenchmarkConfig,
    dataset: DatasetFold,
    sampler_spec: Any,
    seed: int,
    factory: ModelFactory,
) -> dict[str, Any]:
    sampler = build_sampler(sampler_spec)
    def sample_context() -> tuple[np.ndarray, Any, Any]:
        indices = sampler.select(
            dataset.X_train,
            dataset.y_train,
            problem_type=dataset.problem_type,
            random_state=seed,
        )
        _validate_selection(indices, len(dataset.X_train))
        return (
            indices,
            _take_rows(dataset.X_train, indices),
            _take_rows(dataset.y_train, indices),
        )

    (
        (selected, X_sampled, y_sampled),
        sampling_seconds,
    ) = _timed(
        sample_context
    )

    estimator = factory.create_estimator(dataset.problem_type, seed)
    _, fit_seconds = _timed(
        lambda: estimator.fit(X_sampled, y_sampled),
        synchronize=factory.synchronize,
    )

    def predict_call() -> Any:
        if dataset.problem_type == "classification":
            return estimator.predict_proba(dataset.X_test)
        return estimator.predict(dataset.X_test)
    warmup_seconds = 0.0
    for _ in range(config.warmup_runs):
        _, elapsed = _timed(predict_call, synchronize=factory.synchronize)
        warmup_seconds += elapsed
    predictions, predict_seconds = _timed(
        predict_call,
        synchronize=factory.synchronize,
    )

    if dataset.problem_type == "classification":
        metrics = classification_metrics(
            dataset.y_test,
            predictions,
            estimator.classes_,
        )
        primary_metric = "accuracy"
    else:
        metrics = regression_metrics(dataset.y_test, predictions)
        primary_metric = "rmse"

    measured_seconds = sampling_seconds + fit_seconds + predict_seconds
    result = {
        **base_result,
        "n_train_sampled": len(selected),
        "retained_fraction": len(selected) / len(dataset.X_train),
        "sampling_seconds": sampling_seconds,
        "fit_seconds": fit_seconds,
        "warmup_runs": config.warmup_runs,
        "warmup_seconds": warmup_seconds,
        "predict_seconds": predict_seconds,
        "measured_total_seconds": measured_seconds,
        "workflow_total_seconds": measured_seconds + warmup_seconds,
        "primary_metric": primary_metric,
    }
    result.update(
        {
            f"metric_{name}": None if math.isnan(value) else value
            for name, value in metrics.items()
        }
    )
    return result


def _timed(
    function: Callable[[], Any],
    *,
    synchronize: Callable[[Any | None], None] | None = None,
) -> tuple[Any, float]:
    if synchronize:
        synchronize(None)
    start = time.perf_counter()
    value = function()
    if synchronize:
        synchronize(value)
    return value, time.perf_counter() - start


def _take_rows(value: Any, indices: np.ndarray) -> Any:
    if hasattr(value, "iloc"):
        return value.iloc[indices].reset_index(drop=True)
    return np.asarray(value)[indices]


def _validate_selection(indices: Any, n_rows: int) -> None:
    selected = np.asarray(indices)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("Sampler must return a nonempty one-dimensional index array")
    if not np.issubdtype(selected.dtype, np.integer):
        raise TypeError("Sampler indices must be integers")
    if selected.min() < 0 or selected.max() >= n_rows:
        raise IndexError("Sampler returned an out-of-range row index")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Sampler returned duplicate row indices")


def _write_result(output_file: Any, result: dict[str, Any]) -> None:
    output_file.write(json.dumps(result, sort_keys=True) + "\n")
    output_file.flush()


def _runtime_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "openml", "pandas", "scikit-learn", "tabfm"):
        try:
            versions[f"version_{package.replace('-', '_')}"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[f"version_{package.replace('-', '_')}"] = None

    accelerator = "unknown"
    backend_version = None
    try:
        if config.model.backend == "pytorch":
            import torch

            backend_version = torch.__version__
            accelerator = (
                torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            import jax

            backend_version = jax.__version__
            accelerator = ", ".join(str(device) for device in jax.devices())
    except ImportError:
        pass

    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_implementation": sys.implementation.name,
        "backend_version": backend_version,
        "accelerator": accelerator,
        **versions,
    }
