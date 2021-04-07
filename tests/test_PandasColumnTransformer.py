import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.PandasColumnTransformer import PandasColumnTransformer


@pytest.fixture
def synthetic_data():
    df_train = pd.DataFrame({
        "num1": np.arange(8, dtype=np.float64),
        "num2": [-1, -2, -3, -4, -5, -6, -7, -8],
        "cat1": ["Heaven"]*3 + ["Earth"]*2 + ["Purgatory"]*3,
        "cat2": ["Male"]*4 + ["Female"]*4
    })

    df_test = pd.DataFrame({
        "num1": np.arange(8, dtype=np.float64),
        "num2": [-1, -2, -3, -4, -5, -6, -7, -8],
        "cat1": ["Heaven"]*3 + ["Earth"]*2 + ["Mars"]*3,
        "cat2": ["Male"]*4 + ["Female"]*4
    })

    return df_train, df_test


def test_column_transformer_drop(synthetic_data):
    df_train, df_test = synthetic_data
    pcf = PandasColumnTransformer([
        ("numerical_standard", StandardScaler(), ["num1"]),
        ("cat", OneHotEncoder(handle_unknown="error"), ["cat1", "cat2"]),
    ], remainder="drop").fit(df_train)

    # Confirmed with manual print-outs
    assert set(pcf.transform(df_train).columns) == set([
        "num1", "cat1_Earth", "cat1_Heaven", "cat1_Purgatory",
        "cat2_Female", "cat2_Male"
    ])

    with pytest.raises(ValueError):
        _ = pcf.transform(df_test)


def test_column_transformer_passthrough(synthetic_data):
    df_train, df_test = synthetic_data
    pcf = PandasColumnTransformer([
        ("numerical_standard", StandardScaler(), ["num1"]),
        ("cat", OneHotEncoder(handle_unknown="error"), ["cat1", "cat2"]),
    ], remainder="passthrough").fit(df_train)

    # Confirmed with manual print-outs
    assert pcf.transform(df_train).columns.tolist() == [
        "num1", "cat1_Earth", "cat1_Heaven", "cat1_Purgatory",
        "cat2_Female", "cat2_Male", "num2"
    ]

    with pytest.raises(ValueError):
        _ = pcf.transform(df_test)
