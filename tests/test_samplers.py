import numpy as np
import pandas as pd

from ltm_sampling.samplers import RandomSampler, StratifiedSampler


def test_random_sampler_is_exact_reproducible_and_order_preserving():
    X = pd.DataFrame({"x": range(10)})
    sampler = RandomSampler(fraction=0.5)

    first = sampler.select(
        X, None, problem_type="classification", random_state=7
    )
    second = sampler.select(
        X, None, problem_type="classification", random_state=7
    )

    assert len(first) == 5
    assert np.array_equal(first, second)
    assert np.all(first[:-1] < first[1:])


def test_classification_stratification_retains_every_class():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series([0] * 15 + [1] * 4 + [2])

    selected = StratifiedSampler(fraction=0.25).select(
        X, y, problem_type="classification", random_state=3
    )

    assert len(selected) == 5
    assert set(y.iloc[selected]) == {0, 1, 2}


def test_regression_stratification_spans_target_distribution():
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series(np.arange(100, dtype=float))

    selected = StratifiedSampler(fraction=0.2, regression_bins=5).select(
        X, y, problem_type="regression", random_state=11
    )

    selected_bins = pd.qcut(y.iloc[selected], q=5, labels=False)
    assert len(selected) == 20
    assert selected_bins.nunique() == 5
