from dataclasses import dataclass
from typing import *

import pandas as pd
from numpy.lib.arraysetops import isin
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class ImputeByGroups:
    """ Collect common attributes used by group-by-imputers """
    target_col: str  # the ONE column to be imputed 
    groupby_col: Iterable[str] = None  # the columns on which to pd.DataFrame.groupby, if any
    imputation_values: Any = None  # dict(-like) obj mapping from grouped by
    # return_df: bool = True  # whether to return a DataFrame or just the imputed array
    copy: bool = True  # return a copy, or modify in-place? Only relevant when return_df == True
    
    def __post_init__(self):
        assert isinstance(self.target_col, str)
        self.groupby_col = [self.groupby_col] if isinstance(self.groupby_col, str) else self.groupby_col


class ImputeNumericalByGroups(ImputeByGroups, BaseEstimator, TransformerMixin):
    """ Impute ONE continuous numerical column, optionally after grouping by other (discrete) columns """
    def __init__(self, target_col, **kwargs):
        super().__init__(target_col=target_col, **kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        """ Implement the imputation logic here """
        assert isinstance(X, pd.DataFrame)
        if self.groupby_col:
            self.imputation_values = X.groupby(self.groupby_col)[self.target_col].median().to_dict()
        else:
            self.imputation_values = X[self.target_col].median()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)
        imputed_df = X.copy() if self.copy else X

        na_mask = imputed_df[self.target_col].isna()
        if self.groupby_col:
            imputed_df.loc[na_mask, self.target_col] = imputed_df.loc[na_mask].apply(
                lambda row: self.imputation_values[ tuple(row.loc[self.groupby_col]) ],
                axis=1
            )
        else:
            imputed_df.loc[:, self.target_col].fillna(self.imputation_values, inplace=True)
    
        return imputed_df
        

