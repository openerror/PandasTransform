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


# def test_numerical_df_wide(titanic_data):
#     """ no grouping by; just impute using dataframe-wide median """
#     titanic_train, _ = titanic_data
#     na_mask = titanic_train["Age"].isna()

#     imputer = ImputeNumericalByGroup(target_col="Age", copy=True, return_df=True).fit(
#         titanic_train
#     )
#     df = imputer.transform(titanic_train)
#     assert df["Age"].isna().sum() == 0
#     assert id(df) != id(titanic_train)
#     assert np.all(df.loc[na_mask, "Age"] == 28.0)


# def test_numerical_gb1(titanic_data):
#     """ gb1 stands for grouping by 1 columns """
#     titanic_train, _ = titanic_data
#     na_mask = titanic_train["Age"].isna()

#     imputer = ImputeNumericalByGroup(
#         target_col="Age", groupby_col=["Pclass"], copy=True, return_df=True
#     ).fit(titanic_train)
#     df = imputer.transform(titanic_train)
#     assert df["Age"].isna().sum() == 0
#     assert id(df) != id(titanic_train)
#     assert np.all(
#         df.loc[na_mask].apply(
#             lambda row: row["Age"] == imputer.imputation_values[(row["Pclass"],)],
#             axis=1,
#         )
#     )


# def test_numerical_gb2(titanic_data):
#     """ gb2 stands for grouping by 2 columns """
#     titanic_train, _ = titanic_data
#     na_mask = titanic_train["Age"].isna()

#     imputer = ImputeNumericalByGroup(
#         target_col="Age", groupby_col=["Pclass", "Embarked"], copy=True, return_df=True
#     ).fit(titanic_train)
#     df = imputer.transform(titanic_train)
#     assert df["Age"].isna().sum() == 0
#     assert id(df) != id(titanic_train)
#     assert np.all(
#         df.loc[na_mask].apply(
#             lambda row: row["Age"] == imputer.imputation_values[tuple(row[imputer.groupby_col])],
#             axis=1,
#         )
#     )

def test_numerical(titanic_data):
    titanic_train, _ = titanic_data  
    na_mask = titanic_train["Age"].isna()

    for gb_col in (None, 'Pclass', ['Pclass', 'Embarked']):
        imputer = ImputeNumericalByGroup(
            target_col="Age", groupby_col=gb_col, copy=True, return_df=True
        ).fit(titanic_train)

        df = imputer.transform(titanic_train)
        assert df["Age"].isna().sum() == 0
        assert id(df) != id(titanic_train)

        if gb_col:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Age"] == imputer.imputation_values[tuple(row[imputer.groupby_col])],
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
    titanic_train, _ = titanic_data  
    na_mask = titanic_train["Embarked"].isna()

    for gb_col in (None, 'Pclass'):
        imputer = ImputeCategoricalByGroup(
            target_col="Embarked", groupby_col=gb_col, copy=True, return_df=True
        ).fit(titanic_train)

        df = imputer.transform(titanic_train)
        assert df["Embarked"].isna().sum() == 0
        assert id(df) != id(titanic_train)

        if gb_col:
            assert np.all(
                df.loc[na_mask].apply(
                    lambda row: row["Embarked"] == imputer.imputation_values[tuple(row[imputer.groupby_col])],
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