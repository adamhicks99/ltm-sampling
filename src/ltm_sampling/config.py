"""Configuration parsing and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ProblemType = Literal["classification", "regression"]
Backend = Literal["pytorch", "jax"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_task_id: int
    problem_type: ProblemType
    repeat: int = 0
    fold: int = 0
    sample: int = 0


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    method: str
    fraction: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    backend: Backend = "pytorch"
    preset: Literal["default", "ensemble"] = "default"
    n_estimators: int = 8
    batch_size: int | None = 1
    device: str = "auto"
    checkpoint_path: str | None = None
    estimator_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    datasets: tuple[DatasetSpec, ...]
    samplers: tuple[SamplerSpec, ...]
    model: ModelConfig = field(default_factory=ModelConfig)
    seeds: tuple[int, ...] = (0,)
    warmup_runs: int = 0
    output: Path = Path("outputs/results.jsonl")
    fail_fast: bool = False

    @property
    def run_count(self) -> int:
        return len(self.datasets) * len(self.samplers) * len(self.seeds)


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load a benchmark configuration from TOML."""
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    benchmark_raw = raw.get("benchmark", {})
    model_raw = raw.get("model", {})

    datasets = tuple(
        DatasetSpec(
            name=str(_required(item, "name", "datasets")),
            openml_task_id=int(_required(item, "openml_task_id", "datasets")),
            problem_type=str(
                _required(item, "problem_type", "datasets")
            ).lower(),  # type: ignore[arg-type]
            repeat=int(item.get("repeat", 0)),
            fold=int(item.get("fold", 0)),
            sample=int(item.get("sample", 0)),
        )
        for item in raw.get("datasets", [])
    )

    samplers = tuple(
        SamplerSpec(
            name=str(_required(item, "name", "samplers")),
            method=str(item.get("method", item["name"])).lower(),
            fraction=float(item.get("fraction", 1.0)),
            params=dict(item.get("params", {})),
        )
        for item in raw.get("samplers", [])
    )

    batch_size_raw = model_raw.get("batch_size", 1)
    model = ModelConfig(
        backend=str(model_raw.get("backend", "pytorch")).lower(),  # type: ignore[arg-type]
        preset=str(model_raw.get("preset", "default")).lower(),  # type: ignore[arg-type]
        n_estimators=int(model_raw.get("n_estimators", 8)),
        batch_size=None if batch_size_raw is None else int(batch_size_raw),
        device=str(model_raw.get("device", "auto")),
        checkpoint_path=model_raw.get("checkpoint_path"),
        estimator_kwargs=dict(model_raw.get("estimator_kwargs", {})),
    )

    config = BenchmarkConfig(
        datasets=datasets,
        samplers=samplers,
        model=model,
        seeds=tuple(int(seed) for seed in benchmark_raw.get("seeds", [0])),
        warmup_runs=int(benchmark_raw.get("warmup_runs", 0)),
        output=Path(benchmark_raw.get("output", "outputs/results.jsonl")),
        fail_fast=bool(benchmark_raw.get("fail_fast", False)),
    )
    _validate(config)
    return config


def _validate(config: BenchmarkConfig) -> None:
    if not config.datasets:
        raise ValueError("At least one [[datasets]] entry is required")
    if not config.samplers:
        raise ValueError("At least one [[samplers]] entry is required")
    if not config.seeds:
        raise ValueError("benchmark.seeds cannot be empty")
    if config.warmup_runs < 0:
        raise ValueError("benchmark.warmup_runs must be non-negative")
    if config.model.backend not in {"pytorch", "jax"}:
        raise ValueError("model.backend must be 'pytorch' or 'jax'")
    if config.model.preset not in {"default", "ensemble"}:
        raise ValueError("model.preset must be 'default' or 'ensemble'")
    if config.model.n_estimators < 1:
        raise ValueError("model.n_estimators must be positive")

    dataset_names = [dataset.name for dataset in config.datasets]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("Dataset names must be unique")
    for dataset in config.datasets:
        if dataset.problem_type not in {"classification", "regression"}:
            raise ValueError(
                f"{dataset.name}: problem_type must be classification or regression"
            )
        if min(dataset.repeat, dataset.fold, dataset.sample) < 0:
            raise ValueError(f"{dataset.name}: split coordinates must be non-negative")

    sampler_names = [sampler.name for sampler in config.samplers]
    if len(sampler_names) != len(set(sampler_names)):
        raise ValueError("Sampler names must be unique")
    full_samplers = [sampler for sampler in config.samplers if sampler.method == "full"]
    if len(full_samplers) != 1:
        raise ValueError("Exactly one full sampler is required as the paired baseline")
    for sampler in config.samplers:
        if sampler.method not in {"full", "knn", "random", "stratified"}:
            raise ValueError(
                f"{sampler.name}: unknown sampler method {sampler.method!r}"
            )
        if not 0 < sampler.fraction <= 1:
            raise ValueError(f"{sampler.name}: fraction must be in (0, 1]")
        if sampler.method == "full" and sampler.fraction != 1:
            raise ValueError(f"{sampler.name}: the full sampler must use fraction = 1")
