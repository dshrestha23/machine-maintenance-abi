from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.pdm.data import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from src.pdm.features import MaintenanceFeatureEngineer

RANDOM_STATE = 42

ENGINEERED_COLUMNS: tuple[str, ...] = (
    "Temperature difference [K]",
    "Mechanical power [W]",
    "Wear-load interaction",
)


def create_preprocessor(
    use_feature_engineering: bool = True,
) -> ColumnTransformer:
    """Create preprocessing for the selected feature configuration."""
    numeric_columns = list(NUMERICAL_COLUMNS)

    if use_feature_engineering:
        numeric_columns.extend(ENGINEERED_COLUMNS)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
    )


def create_model(
    use_feature_engineering: bool = True,
    **forest_parameters: Any,
) -> Pipeline:
    """Create an unfitted Random Forest classification pipeline.

    Args:
        use_feature_engineering: Add three derived measurements if True.
        **forest_parameters: Overrides for Random Forest parameters.

    Returns:
        A pipeline that can be trained or used in cross-validation.
    """
    default_parameters: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 12,
        "max_features": 0.8,
        "min_samples_leaf": 4,
        "min_samples_split": 8,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }

    default_parameters.update(forest_parameters)

    feature_step = (
        MaintenanceFeatureEngineer() if use_feature_engineering else "passthrough"
    )

    return Pipeline(
        steps=[
            ("feature_engineering", feature_step),
            (
                "preprocessor",
                create_preprocessor(use_feature_engineering),
            ),
            (
                "classifier",
                RandomForestClassifier(**default_parameters),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a trained model using classification metrics.

    Args:
        model: Fitted Random Forest pipeline.
        features: Observations to evaluate.
        target: Actual binary failure labels.
        threshold: Probability cutoff for predicting failure.

    Returns:
        Classification metrics and the confusion matrix.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1_score": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }
