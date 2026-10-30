from pathlib import Path

import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)

TARGET_COLUMN = "Machine failure"

CATEGORICAL_COLUMNS: tuple[str, ...] = ("Type",)

NUMERICAL_COLUMNS: tuple[str, ...] = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load and validate the AI4I predictive maintenance dataset.

    Args:
        file_path: Path to the dataset CSV file.

    Returns:
        A validated DataFrame containing the original dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
        ValueError: If required columns or valid target values are missing.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Download ai4i2020.csv and place it in the data folder."
        )

    dataframe = pd.read_csv(path)

    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: " f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    if not dataframe[TARGET_COLUMN].isin((0, 1)).all():
        raise ValueError("Machine failure must contain only binary values: 0 or 1.")

    return dataframe


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor features from the machine failure target.

    Args:
        dataframe: Validated AI4I dataset.

    Returns:
        A tuple containing the feature DataFrame and target Series.
    """
    features = dataframe.loc[:, list(FEATURE_COLUMNS)].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target
