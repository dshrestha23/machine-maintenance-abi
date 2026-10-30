import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pdm.data import load_dataset, split_features_target
from src.pdm.inspection import get_feature_importance, inspect_prediction
from src.pdm.model import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main() -> None:
    """Verify the saved model and inspect feature importance."""
    model_path = ARTIFACTS_DIR / "random_forest.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    if not model_path.is_file():
        raise FileNotFoundError("Trained model not found. Run python train.py first.")

    model = joblib.load(model_path)

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    print("Saved model loaded successfully.")
    print(
        "Selected feature set:",
        metrics["selected_feature_set"],
    )

    dataframe = load_dataset(DATASET_PATH)
    features, target = split_features_target(dataframe)

    # Reproduce the training script's 70/15/15 split.
    _, x_remaining, _, y_remaining = train_test_split(
        features,
        target,
        train_size=0.70,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    x_validation, _, y_validation, _ = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # Verify the saved model accepts only the six original inputs.
    observation = x_validation.iloc[[0]].copy()
    result = inspect_prediction(model, observation)

    print("\nExample prediction:")
    print(f"  Prediction: {result['prediction']}")
    print("  Failure probability: " f"{result['failure_probability']:.4f}")

    # Verify the model automatically generates derived features.
    feature_step = model.named_steps["feature_engineering"]

    if feature_step == "passthrough":
        print("\nFeature engineering is disabled.")
    else:
        transformed = feature_step.transform(observation)

        print("\nAutomatically generated features:")
        for column in (
            "Temperature difference [K]",
            "Mechanical power [W]",
            "Wear-load interaction",
        ):
            print(f"  {column}: {transformed[column].iloc[0]:.4f}")

    print("\nCalculating permutation feature importance...")

    importance = get_feature_importance(
        model=model,
        features=x_validation,
        target=y_validation,
        n_repeats=5,
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    output_path = ARTIFACTS_DIR / "feature_importance.csv"
    importance.to_csv(output_path, index=False)

    print(f"\nSaved feature importance: {output_path}")


if __name__ == "__main__":
    main()
