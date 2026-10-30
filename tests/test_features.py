import numpy as np
import pandas as pd
import pytest

from pdm.features import MaintenanceFeatureEngineer
from pdm.model import create_model


@pytest.fixture
def sample_observations() -> pd.DataFrame:
    """Provide example machine measurements."""
    return pd.DataFrame(
        {
            "Type": ["M", "L"],
            "Air temperature [K]": [300.0, 298.0],
            "Process temperature [K]": [310.0, 307.0],
            "Rotational speed [rpm]": [1500.0, 1600.0],
            "Torque [Nm]": [40.0, 35.0],
            "Tool wear [min]": [100.0, 80.0],
        }
    )


def test_temperature_difference(
    sample_observations: pd.DataFrame,
) -> None:
    """Check temperature difference calculations."""
    transformer = MaintenanceFeatureEngineer()

    transformed = transformer.fit_transform(sample_observations)

    assert transformed["Temperature difference [K]"].tolist() == [
        10.0,
        9.0,
    ]


def test_mechanical_power(
    sample_observations: pd.DataFrame,
) -> None:
    """Check mechanical power calculations."""
    transformer = MaintenanceFeatureEngineer()

    transformed = transformer.fit_transform(sample_observations)

    expected_power = 40.0 * 1500.0 * 2.0 * np.pi / 60.0

    assert transformed["Mechanical power [W]"].iloc[0] == pytest.approx(expected_power)


def test_wear_load_interaction(
    sample_observations: pd.DataFrame,
) -> None:
    """Check wear-load interaction calculations."""
    transformer = MaintenanceFeatureEngineer()

    transformed = transformer.fit_transform(sample_observations)

    assert transformed["Wear-load interaction"].tolist() == [
        4000.0,
        2800.0,
    ]


def test_original_data_is_not_modified(
    sample_observations: pd.DataFrame,
) -> None:
    """Ensure feature engineering does not modify its input."""
    original = sample_observations.copy(deep=True)

    transformer = MaintenanceFeatureEngineer()
    transformed = transformer.fit_transform(sample_observations)

    pd.testing.assert_frame_equal(sample_observations, original)

    assert len(sample_observations.columns) == 6
    assert len(transformed.columns) == 9


def test_original_pipeline_has_no_feature_engineering() -> None:
    """Check that the original pipeline bypasses feature engineering."""
    model = create_model(use_feature_engineering=False)

    assert model.named_steps["feature_engineering"] == "passthrough"


def test_engineered_pipeline_includes_transformer() -> None:
    """Check that the engineered pipeline includes its transformer."""
    model = create_model(use_feature_engineering=True)

    assert isinstance(
        model.named_steps["feature_engineering"],
        MaintenanceFeatureEngineer,
    )
