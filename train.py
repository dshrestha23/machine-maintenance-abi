import json
from pathlib import Path
from typing import Any

import joblib
from scipy.stats import randint
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from pdm.data import load_dataset, split_features_target
from pdm.model import (
    RANDOM_STATE,
    create_model,
    evaluate_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SEARCH_ITERATIONS = 20


def create_parameter_distributions() -> dict[str, Any]:
    """Define Random Forest hyperparameters to search."""
    return {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, 12, 20, None],
        "classifier__min_samples_split": randint(2, 11),
        "classifier__min_samples_leaf": randint(1, 5),
        "classifier__max_features": ["sqrt", 0.8],
        "classifier__class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],
    }


def tune_model(
    x_train: Any,
    y_train: Any,
    use_feature_engineering: bool,
) -> RandomizedSearchCV:
    """Find strong Random Forest settings using training data only."""
    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=create_model(use_feature_engineering=use_feature_engineering),
        param_distributions=create_parameter_distributions(),
        n_iter=SEARCH_ITERATIONS,
        scoring="f1",
        cv=cross_validation,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )

    search.fit(x_train, y_train)
    return search


def print_metrics(
    title: str,
    metrics: dict[str, Any],
) -> None:
    """Print numerical metrics and the confusion matrix."""
    print(f"\n{title}:")

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")

    print("  confusion_matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")


def main() -> None:
    """Run model selection, evaluation, and artifact saving."""
    dataframe = load_dataset(DATASET_PATH)
    features, target = split_features_target(dataframe)

    # 70% training, 30% remaining.
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        target,
        train_size=0.70,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # Divide the remaining 30% equally:
    # 15% validation and 15% testing.
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    print("Dataset split:")
    print(f"  Training: {len(x_train):,} observations (70%)")
    print(f"  Validation: {len(x_validation):,} observations (15%)")
    print(f"  Testing: {len(x_test):,} observations (15%)")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: ORIGINAL SIX FEATURES")
    print("=" * 60)

    original_search = tune_model(
        x_train=x_train,
        y_train=y_train,
        use_feature_engineering=False,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ENGINEERED NINE FEATURES")
    print("=" * 60)

    engineered_search = tune_model(
        x_train=x_train,
        y_train=y_train,
        use_feature_engineering=True,
    )

    searches = {
        "original_features": original_search,
        "engineered_features": engineered_search,
    }

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 60)

    for name, search in searches.items():
        print(f"  {name}: {search.best_score_:.4f}")

    # Choose the feature configuration with the highest
    # mean cross-validation F1-score.
    selected_name = max(
        searches,
        key=lambda name: searches[name].best_score_,
    )

    selected_search = searches[selected_name]
    model = selected_search.best_estimator_

    print(f"\nSelected feature configuration: {selected_name}")
    print("Best cross-validation F1-score: " f"{selected_search.best_score_:.4f}")

    print("\nBest hyperparameters:")
    for name, value in selected_search.best_params_.items():
        print(f"  {name}: {value}")

    # Evaluate the selected model without changing its settings.
    validation_metrics = evaluate_model(
        model,
        x_validation,
        y_validation,
    )

    test_metrics = evaluate_model(
        model,
        x_test,
        y_test,
    )

    print_metrics("Validation metrics", validation_metrics)
    print_metrics("Final test metrics", test_metrics)

    # Store a comparison of both feature configurations.
    feature_comparison = {
        name: {
            "cross_validation_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
        }
        for name, search in searches.items()
    }

    results = {
        "model_name": "Tuned Random Forest",
        "random_state": RANDOM_STATE,
        "selected_feature_set": selected_name,
        "search_iterations_per_feature_set": SEARCH_ITERATIONS,
        "cross_validation_folds": 3,
        "cross_validation_scoring": "f1",
        "selected_cross_validation_f1": float(selected_search.best_score_),
        "best_parameters": selected_search.best_params_,
        "feature_comparison": feature_comparison,
        "dataset_split": {
            "training": len(x_train),
            "validation": len(x_validation),
            "testing": len(x_test),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "random_forest.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nTraining completed successfully.")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
