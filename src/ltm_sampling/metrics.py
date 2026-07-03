"""Predictive-quality metrics for paired benchmark runs."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    classes: Any,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities, dtype=float)
    class_array = np.asarray(classes)
    predictions = class_array[np.argmax(probability_array, axis=1)]

    metrics = {
        "accuracy": float(accuracy_score(y_true_array, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, predictions)
        ),
        "f1_macro": float(
            f1_score(y_true_array, predictions, average="macro", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true_array, probability_array, labels=class_array)
        ),
    }

    try:
        if len(class_array) == 2:
            binary_target = (y_true_array == class_array[1]).astype(int)
            metrics["roc_auc"] = float(
                roc_auc_score(binary_target, probability_array[:, 1])
            )
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true_array,
                    probability_array,
                    labels=class_array,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        # Some published test folds may omit a rare class.
        metrics["roc_auc"] = float("nan")
    return metrics


def regression_metrics(y_true: Any, predictions: Any) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float).ravel()
    return {
        "rmse": float(mean_squared_error(y_true_array, prediction_array) ** 0.5),
        "mae": float(mean_absolute_error(y_true_array, prediction_array)),
        "r2": float(r2_score(y_true_array, prediction_array)),
    }
