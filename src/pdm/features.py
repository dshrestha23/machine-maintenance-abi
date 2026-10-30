import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MaintenanceFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create domain-informed features from machine measurements."""

    REQUIRED_COLUMNS = (
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    )

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "MaintenanceFeatureEngineer":
        """Validate input columns without learning any parameters."""
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Feature engineering requires a DataFrame.")

        missing = set(self.REQUIRED_COLUMNS).difference(x.columns)

        if missing:
            raise ValueError(f"Missing feature engineering columns: {sorted(missing)}")

        self.feature_names_in_ = list(x.columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the data with three engineered features."""
        check_is_fitted(self, "feature_names_in_")

        missing = set(self.REQUIRED_COLUMNS).difference(x.columns)

        if missing:
            raise ValueError(f"Missing feature engineering columns: {sorted(missing)}")

        dataframe = x.copy()

        dataframe["Temperature difference [K]"] = (
            dataframe["Process temperature [K]"] - dataframe["Air temperature [K]"]
        )

        dataframe["Mechanical power [W]"] = (
            dataframe["Torque [Nm]"]
            * dataframe["Rotational speed [rpm]"]
            * (2.0 * np.pi / 60.0)
        )

        dataframe["Wear-load interaction"] = (
            dataframe["Tool wear [min]"] * dataframe["Torque [Nm]"]
        )

        return dataframe
