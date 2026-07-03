import numpy as np
import pandas as pd

from ltm_sampling.benchmark import run_benchmark
from ltm_sampling.config import (
    BenchmarkConfig,
    DatasetSpec,
    ModelConfig,
    SamplerSpec,
)
from ltm_sampling.datasets import DatasetFold
from ltm_sampling.models import PreparedModel


class FakeClassifier:
    def fit(self, X, y):
        del X
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        positive = np.clip(np.asarray(X["signal"], dtype=float), 0.05, 0.95)
        return np.column_stack([1 - positive, positive])


class FakeFactory:
    def prepare(self, problem_type):
        del problem_type
        return PreparedModel(backbone=None, load_seconds=0.01, device="cpu")

    def create_estimator(self, problem_type, random_state):
        del problem_type, random_state
        return FakeClassifier()

    def synchronize(self, value=None):
        del value


def test_benchmark_runs_paired_conditions_without_tabfm(tmp_path):
    dataset = DatasetFold(
        name="synthetic",
        openml_task_id=1,
        problem_type="classification",
        repeat=0,
        fold=0,
        sample=0,
        X_train=pd.DataFrame({"signal": np.linspace(0, 1, 20)}),
        y_train=pd.Series([0] * 10 + [1] * 10),
        X_test=pd.DataFrame({"signal": [0.1, 0.9]}),
        y_test=pd.Series([0, 1]),
    )
    config = BenchmarkConfig(
        datasets=(DatasetSpec("synthetic", 1, "classification"),),
        samplers=(
            SamplerSpec(name="full", method="full"),
            SamplerSpec(name="random-50", method="random", fraction=0.5),
        ),
        model=ModelConfig(n_estimators=1),
        seeds=(4,),
        output=tmp_path / "results.jsonl",
    )

    results = run_benchmark(
        config,
        dataset_loader=lambda _: dataset,
        model_factory=FakeFactory(),
    )

    assert len(results) == 2
    assert {result["status"] for result in results} == {"success"}
    assert {result["n_train_sampled"] for result in results} == {10, 20}
    assert all(result["metric_accuracy"] == 1.0 for result in results)
