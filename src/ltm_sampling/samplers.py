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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


@dataclass(frozen=True)
class KNNRedundancySampler:
    """Prune locally redundant rows while protecting nearby representatives."""

    fraction: float
    n_neighbors: int = 5
    regression_bins: int = 10

    def select(
        self,
        X: Any,
        y: Any,
        *,
        problem_type: ProblemType,
        random_state: int,
    ) -> IndexArray:
        if self.fraction < 0.5:
            raise ValueError(
                "KNN redundancy sampling currently requires fraction >= 0.5"
            )
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")

        labels = _strata(y, problem_type, self.regression_bins)
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        requested_size = _target_size(len(X), self.fraction)
        target_size = max(requested_size, len(unique_labels))
        target_size = min(target_size, len(X))
        encoded_features = _encode_features(X)

        group_indices = [
            np.flatnonzero(inverse == group_id)
            for group_id in range(len(unique_labels))
        ]
        allocations = _allocate_strata(
            np.asarray([len(indices) for indices in group_indices]), target_size
        )
        rng = np.random.default_rng(random_state)
        selected = [
            self._select_group(
                encoded_features,
                indices,
                count,
                rng,
            )
            for indices, count in zip(group_indices, allocations, strict=True)
        ]
        return np.sort(np.concatenate(selected).astype(np.int64))

    def _select_group(
        self,
        features: Any,
        group_indices: IndexArray,
        target_size: int,
        rng: np.random.Generator,
    ) -> IndexArray:
        group_size = len(group_indices)
        if target_size == group_size:
            return group_indices.copy()

        group_features = features[group_indices]
        neighbor_count = min(self.n_neighbors + 1, group_size)
        model = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric="euclidean",
            n_jobs=1,
        )
        model.fit(group_features)
        distances, neighbors = model.kneighbors(group_features)

        neighbor_lists: list[NDArray[np.int64]] = []
        mean_distances = np.empty(group_size, dtype=float)
        for row_index in range(group_size):
            non_self = neighbors[row_index] != row_index
            row_neighbors = neighbors[row_index][non_self][: self.n_neighbors]
            row_distances = distances[row_index][non_self][: self.n_neighbors]
            neighbor_lists.append(row_neighbors.astype(np.int64))
            mean_distances[row_index] = (
                float(np.mean(row_distances)) if len(row_distances) else 0.0
            )

        tie_breakers = rng.random(group_size)
        removal_order = np.lexsort((tie_breakers, mean_distances))
        active = np.ones(group_size, dtype=bool)
        protected = np.zeros(group_size, dtype=bool)
        active_count = group_size

        for candidate in removal_order:
            if active_count == target_size:
                break
            if not active[candidate] or protected[candidate]:
                continue

            representatives = neighbor_lists[candidate][
                active[neighbor_lists[candidate]]
            ]
            if len(representatives):
                representative = int(representatives[0])
            else:
                other_active = np.flatnonzero(active)
                other_active = other_active[other_active != candidate]
                nearest, _ = pairwise_distances_argmin_min(
                    group_features[candidate : candidate + 1],
                    group_features[other_active],
                    metric="euclidean",
                )
                representative = int(other_active[int(nearest[0])])

            active[candidate] = False
            protected[representative] = True
            active_count -= 1

        if active_count != target_size:
            raise RuntimeError(
                "KNN pruning could not reach the requested retained-row count"
            )
        return group_indices[np.flatnonzero(active)]


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
    if spec.method == "knn":
        return KNNRedundancySampler(
            fraction=spec.fraction,
            n_neighbors=int(spec.params.get("n_neighbors", 5)),
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


def _encode_features(X: Any) -> Any:
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            )
        )
    transformer = ColumnTransformer(transformers, sparse_threshold=1.0)
    return transformer.fit_transform(frame)


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
