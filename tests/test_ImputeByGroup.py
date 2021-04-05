""" 
Run tests from parent directory, which
    - is one level up from tests/;
    - contains src/
"""

import numpy as np
import pandas as pd
import pytest
from src.ImputeByGroup import ImputeNumericalByGroup, ImputeCategoricalByGroup


@pytest.fixture
def titanic_data():
    titanic_train = pd.read_csv("tests/titanic_train.csv")
    titanic_test = pd.read_csv("tests/titanic_test.csv")
    return titanic_train, titanic_test


@pytest.fixture
def mismatch_group_numerical_data():
    df_train = pd.DataFrame({
        "key_a": ["Adam"]*4 + ["Eve"]*4,
        "value": [1,2,3,4] + [5,6,7,8],
    })

    df_test = pd.DataFrame({
        "key_a": ["Adam"]*4 + ["Peter"]*4,
        "value": [1,2,np.nan,4] + [np.nan]*4
    })
    return df_train, df_test


def test_numerical(titanic_data):
    """ Fitting and transforming without needing to consider unseen groups (e.g. in test data) """
    titanic_train, _ = titanic_data
    na_mask = titanic_train["Age"].isna()

    for gb_col in (None, "Pclass", ["Pclass", "Embarked"]):
        imputer = ImputeNumericalByGroup(
            target_col="Age", groupby_col=gb_col, copy=True, return_df=True
        ).fit(titanic_train)

        df = imputer.transform(titanic_train)
        assert df["Age"].isna().sum() == 0
        assert id(df) != id(titanic_train)

        if gb_col:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Age"]
                    == imputer.imputation_values[tuple(row[imputer.groupby_col])],
                    axis=1,
                )
            )
        else:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Age"] == imputer.imputation_values,
                    axis=1,
                )
            )


def test_categorical(titanic_data):
    """ Fitting and transforming without needing to consider unseen groups (e.g. in test data) """
    titanic_train, _ = titanic_data
    na_mask = titanic_train["Embarked"].isna()

    for gb_col in (None, "Pclass"):
        imputer = ImputeCategoricalByGroup(
            target_col="Embarked", groupby_col=gb_col, copy=True, return_df=True
        ).fit(titanic_train)

        df = imputer.transform(titanic_train)
        assert df["Embarked"].isna().sum() == 0
        assert id(df) != id(titanic_train)

        if gb_col:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Embarked"]
                    == imputer.imputation_values[tuple(row[imputer.groupby_col])],
                    axis=1,
                )
            )
        else:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Embarked"] == imputer.imputation_values,
                    axis=1,
                )
            )


def test_mismatch_numerical(mismatch_group_numerical_data):
    """ 
        Examine behavior of numerical imputer when encountering unseen groups 

        With groups seen during fitting, impute from dict imputation_values.
        In contrast, impute rows of unseen groups using dataframe-wide statistics 
        OR throw a KeyError. Choose between the two behaviors via keyword argument.
    """
    train, test = mismatch_group_numerical_data

    # Impute missing groups with df-wide stats
    imputer = ImputeNumericalByGroup(
        target_col="value", 
        groupby_col="key_a",
        return_df=True,
        copy=True,
        key_error_on_unseen=False
    ).fit(train)

    imputed_test = imputer.transform(test)
    assert np.all(imputed_test.query("`key_a` == 'Peter'")["value"] == train.value.median())
    assert imputed_test.iloc[2, 1] == train.query("`key_a` == 'Adam'")["value"].median()

    # Raise KeyError
    imputer = ImputeNumericalByGroup(
        target_col="value", 
        groupby_col="key_a",
        return_df=True,
        copy=True,
        key_error_on_unseen=True
    ).fit(train)

    with pytest.raises(KeyError):
        _ = imputer.transform(test)