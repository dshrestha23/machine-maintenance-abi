from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def get_feature_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Measure the importance of original inputs using permutation.

    Each input column is shuffled repeatedly to measure how much
    the model's F1-score decreases.

    Args:
        model: Fitted classification pipeline.
        features: Original six-feature observations.
        target: Actual binary failure labels.
        n_repeats: Number of shuffles per feature.

    Returns:
        Feature importance sorted from highest to lowest.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")

    importance = permutation_importance(
        estimator=model,
        X=features,
        y=target,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    dataframe = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )

    return dataframe.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def inspect_prediction(
    model: Pipeline,
    observation: pd.DataFrame,
) -> dict[str, Any]:
    """Predict failure from one original six-feature observation."""
    if len(observation) != 1:
        raise ValueError("Provide exactly one observation.")

    probability = float(model.predict_proba(observation)[0, 1])
    prediction = int(model.predict(observation)[0])

    return {
        "prediction": prediction,
        "failure_probability": probability,
    }
