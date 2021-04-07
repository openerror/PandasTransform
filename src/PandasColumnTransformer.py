from itertools import chain
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class PandasColumnTransformer(BaseEstimator, TransformerMixin):
    """A wrapper around sklearn.column.ColumnTransformer to facilitate
    recovery of column (feature) names"""

    def __init__(self, transformers, **kwargs):
        """Initialize by creating ColumnTransformer object
        https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html 

        Args:
            transformers (list of length-3 tuples): (name, Transformer, target columns); see docs
            kwargs: keyword arguments for sklearn.compose.ColumnTransformer
        """
        self.col_transformer = ColumnTransformer(transformers, **kwargs)
        self.transformed_col_names: List[str] = []

    def _get_col_names(self, X: pd.DataFrame):
        """Get names of transformed columns from a fitted self.col_transformer

        Args:
            X (pd.DataFrame): [description]

        Yields:
            Iterator[Iterable[str]]: [description]
        """
        for name, transformer, cols in self.col_transformer.transformers_:
            if hasattr(transformer, "get_feature_names"):
                yield transformer.get_feature_names(cols)
                # print(transformer.get_feature_names(cols))
            elif name == "remainder" and self.col_transformer.remainder=="passthrough":
                yield X.columns[cols].tolist()
                # print(X.columns[cols].tolist())
            elif name == "remainder" and self.col_transformer.remainder=="drop":
                continue
            else:
                yield cols

    def fit(self, X: pd.DataFrame, y: Any=None):
        """[summary]

        Args:
            X (pd.DataFrame): DataFrame to be fitted on
            y (Any, optional): . Defaults to None.
        """
        assert isinstance(X, pd.DataFrame)
        self.col_transformer = self.col_transformer.fit(X)
        self.transformed_col_names = list(chain.from_iterable(self._get_col_names(X)))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a new DataFrame using fitted self.col_transformer

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: DataFrame transformed by self.col_transformer
        """
        assert isinstance(X, pd.DataFrame)
        transformed_X = self.col_transformer.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(transformed_X, columns=self.transformed_col_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                transformed_X, columns=self.transformed_col_names
            )
