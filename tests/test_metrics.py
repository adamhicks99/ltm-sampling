import numpy as np

from ltm_sampling.metrics import classification_metrics, regression_metrics


def test_classification_metrics_use_probability_class_order():
    metrics = classification_metrics(
        y_true=np.array(["no", "yes", "yes"]),
        probabilities=np.array(
            [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]]
        ),
        classes=np.array(["no", "yes"]),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] > 0


def test_regression_metrics():
    metrics = regression_metrics(
        y_true=np.array([1.0, 2.0, 3.0]),
        predictions=np.array([1.0, 2.0, 4.0]),
    )

    assert np.isclose(metrics["rmse"], np.sqrt(1 / 3))
    assert np.isclose(metrics["mae"], 1 / 3)
