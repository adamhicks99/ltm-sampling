"""Training-row sampling strategies.

New techniques should implement the ``Sampler`` protocol and be registered in
``build_sampler``. Sampling returns positional row indices so the benchmark can
apply exactly the same selection to features and targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ltm_sampling.config import ProblemType, SamplerSpec

IndexArray = NDArray[np.int64]


class Sampler(Protocol):
    def select(
        self,
        X: Any,
        y: Any,
        *,
        problem_type: ProblemType,
        random_state: int,
    ) -> IndexArray:
        """Return unique positional indices into the training data."""


@dataclass(frozen=True)
class FullSampler:
    def select(
        self,
        X: Any,
        y: Any,
        *,
        problem_type: ProblemType,
        random_state: int,
    ) -> IndexArray:
        del y, problem_type, random_state
        return np.arange(len(X), dtype=np.int64)


@dataclass(frozen=True)
class RandomSampler:
    fraction: float

    def select(
        self,
        X: Any,
        y: Any,
        *,
        problem_type: ProblemType,
        random_state: int,
    ) -> IndexArray:
        del y, problem_type
        target_size = _target_size(len(X), self.fraction)
        rng = np.random.default_rng(random_state)
        return np.sort(
            rng.choice(len(X), size=target_size, replace=False).astype(np.int64)
        )


@dataclass(frozen=True)
class StratifiedSampler:
    fraction: float
    regression_bins: int = 10

    def select(
        self,
        X: Any,
        y: Any,
        *,
        problem_type: ProblemType,
        random_state: int,
    ) -> IndexArray:
        labels = _strata(y, problem_type, self.regression_bins)
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        requested_size = _target_size(len(X), self.fraction)
        target_size = max(requested_size, len(unique_labels))
        target_size = min(target_size, len(X))

        group_indices = [
            np.flatnonzero(inverse == group_id)
            for group_id in range(len(unique_labels))
        ]
        allocations = _allocate_strata(
            np.asarray([len(indices) for indices in group_indices]), target_size
        )

        rng = np.random.default_rng(random_state)
        selected = [
            rng.choice(indices, size=count, replace=False)
            for indices, count in zip(group_indices, allocations, strict=True)
        ]
        return np.sort(np.concatenate(selected).astype(np.int64))


def build_sampler(spec: SamplerSpec) -> Sampler:
    if spec.method == "full":
        return FullSampler()
    if spec.method == "random":
        return RandomSampler(fraction=spec.fraction)
    if spec.method == "stratified":
        return StratifiedSampler(
            fraction=spec.fraction,
            regression_bins=int(spec.params.get("regression_bins", 10)),
        )
    raise ValueError(f"Unknown sampler method: {spec.method!r}")


def _target_size(total: int, fraction: float) -> int:
    if total < 1:
        raise ValueError("Cannot sample an empty training set")
    return min(total, max(1, int(np.ceil(total * fraction))))


def _strata(y: Any, problem_type: ProblemType, regression_bins: int) -> NDArray[Any]:
    values = np.asarray(y)
    if values.ndim != 1:
        values = values.ravel()
    if problem_type == "classification":
        return values
    if regression_bins < 2:
        raise ValueError("regression_bins must be at least 2")
    if len(np.unique(values)) < 2:
        return np.zeros(len(values), dtype=np.int64)
    return np.asarray(
        pd.qcut(
            values,
            q=min(regression_bins, len(np.unique(values))),
            labels=False,
            duplicates="drop",
        )
    )


def _allocate_strata(counts: NDArray[np.int64], target: int) -> IndexArray:
    """Allocate an exact sample budget while retaining every nonempty stratum."""
    if np.any(counts < 1):
        raise ValueError("Strata must be nonempty")
    if target < len(counts) or target > int(counts.sum()):
        raise ValueError("Target is incompatible with stratum counts")

    allocation = np.ones(len(counts), dtype=np.int64)
    remaining = target - len(counts)
    capacities = counts - 1
    if remaining == 0 or capacities.sum() == 0:
        return allocation

    ideal_extra = capacities * (remaining / capacities.sum())
    extra = np.minimum(np.floor(ideal_extra).astype(np.int64), capacities)
    allocation += extra
    remaining -= int(extra.sum())

    remainders = ideal_extra - np.floor(ideal_extra)
    order = np.argsort(-remainders, kind="stable")
    while remaining:
        progressed = False
        for index in order:
            if allocation[index] < counts[index]:
                allocation[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the requested stratified sample")
    return allocation
