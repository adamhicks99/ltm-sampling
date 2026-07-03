"""OpenML dataset loading with official task splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltm_sampling.config import DatasetSpec, ProblemType


@dataclass(frozen=True)
class DatasetFold:
    name: str
    openml_task_id: int
    problem_type: ProblemType
    repeat: int
    fold: int
    sample: int
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def load_openml_dataset(spec: DatasetSpec) -> DatasetFold:
    """Download one OpenML task and preserve its published train/test split."""
    try:
        import openml
    except ImportError as exc:
        raise RuntimeError(
            "OpenML is not installed. Run `uv sync --extra pytorch` first."
        ) from exc

    task = openml.tasks.get_task(spec.openml_task_id)
    dataset = task.get_dataset()
    target = getattr(task, "target_name", None) or dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(target=target)
    train_indices, test_indices = task.get_train_test_split_indices(
        repeat=spec.repeat,
        fold=spec.fold,
        sample=spec.sample,
    )

    return DatasetFold(
        name=spec.name,
        openml_task_id=spec.openml_task_id,
        problem_type=spec.problem_type,
        repeat=spec.repeat,
        fold=spec.fold,
        sample=spec.sample,
        X_train=X.iloc[train_indices].reset_index(drop=True),
        y_train=y.iloc[train_indices].reset_index(drop=True),
        X_test=X.iloc[test_indices].reset_index(drop=True),
        y_test=y.iloc[test_indices].reset_index(drop=True),
    )
