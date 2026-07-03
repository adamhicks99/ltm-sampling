"""TabFM model loading and estimator construction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ltm_sampling.config import ModelConfig, ProblemType

TABFM_REVISION = "5ee6cd7829b5a4fdfd7e2a266259df733d40d036"


@dataclass(frozen=True)
class PreparedModel:
    backbone: Any
    load_seconds: float
    device: str


class TabFMFactory:
    """Load one backbone per problem type and reuse it across paired runs."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._models: dict[ProblemType, PreparedModel] = {}

    def prepare(self, problem_type: ProblemType) -> PreparedModel:
        if problem_type in self._models:
            return self._models[problem_type]

        try:
            import tabfm
        except ImportError as exc:
            raise RuntimeError(
                f"TabFM is not installed. Run `uv sync --extra {self.config.backend}`."
            ) from exc

        start = time.perf_counter()
        if self.config.backend == "pytorch":
            device = self._resolve_pytorch_device()
            backbone = tabfm.tabfm_v1_0_0_pytorch.load(
                model_type=problem_type,
                checkpoint_path=self.config.checkpoint_path,
                device=device,
            )
        else:
            if self.config.device != "auto":
                raise ValueError(
                    "Explicit JAX device selection is not supported; use device='auto'"
                )
            device = "jax-default"
            backbone = tabfm.tabfm_v1_0_0_jax.load(
                model_type=problem_type,
                checkpoint_path=self.config.checkpoint_path,
            )
        prepared = PreparedModel(
            backbone=backbone,
            load_seconds=time.perf_counter() - start,
            device=device,
        )
        self._models[problem_type] = prepared
        return prepared

    def create_estimator(self, problem_type: ProblemType, random_state: int) -> Any:
        import tabfm

        prepared = self.prepare(problem_type)
        estimator_class = (
            tabfm.TabFMClassifier
            if problem_type == "classification"
            else tabfm.TabFMRegressor
        )
        constructor = (
            estimator_class.ensemble
            if self.config.preset == "ensemble"
            else estimator_class
        )
        kwargs = {
            "n_estimators": self.config.n_estimators,
            "batch_size": self.config.batch_size,
            "random_state": random_state,
            **self.config.estimator_kwargs,
        }
        if kwargs.get("max_num_rows") is not None:
            raise ValueError(
                "TabFM max_num_rows must remain unset: internal row sampling would "
                "confound the external sampling benchmark"
            )
        kwargs["max_num_rows"] = None
        return constructor(model=prepared.backbone, **kwargs)

    def synchronize(self, value: Any | None = None) -> None:
        """Wait for asynchronous accelerator work before stopping a timer."""
        if self.config.backend == "pytorch":
            import torch

            uses_cuda = any(
                prepared.device.startswith("cuda")
                for prepared in self._models.values()
            )
            if uses_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            return

        if value is not None and hasattr(value, "block_until_ready"):
            value.block_until_ready()

    def _resolve_pytorch_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
