from collections import defaultdict 
from dataclasses import dataclass
from typing import *

import pandas as pd
from numpy.lib.arraysetops import isin
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class ImputeByGroup:
    """ 
        Collect common attributes used by group-by-imputers 
        TODO:
            - Encountering new, unrecorded group in test data
    """
    target_col: str  # the ONE column to be imputed 
    copy: bool = True  # return a copy, or modify in-place? Only relevant when return_df == True
    groupby_col: Iterable[str] = None  # the columns on which to pd.DataFrame.groupby, if any
    key_error_on_unseen: bool = True  # throw KeyError at unseen groupby_col values in .transform?
    imputation_values: Any = None  # dict(-like) obj mapping from grouped by
    return_df: bool = False  # return the whole DataFrame, or just the imputed column?
    
    def __post_init__(self):
        assert isinstance(self.target_col, str)
        self.groupby_col = [self.groupby_col] if isinstance(self.groupby_col, str) else self.groupby_col
    
    def _prepare_imputation_values(self, standin: Any):
        """ 
            Refactored common steps from subclasses
            standin_value: if self.imputation_values dict does not contain inquired group, dict
                should return this value
        """
        self.imputation_values = {
            key if isinstance(key, tuple) else (key,): val 
            for key, val in self.imputation_values.iteritems()
        }
        if not self.key_error_on_unseen:
            self.imputation_values = defaultdict(lambda: standin, self.imputation_values) 

    def transform(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """ Same .transform() applies for numerical and categorical data """
        assert isinstance(X, pd.DataFrame)
        assert self.imputation_values, "Imputer is not fitted yet"

        # Only consider self.copy if returning a dataframe
        # If returning just the imputed column, we will copy anyway
        if self.copy and self.return_df:
            imputed_result = X.copy()
        elif self.return_df:
            imputed_result = X
        else:
            imputed_result = X.loc[:, self.target_col].copy()

        na_mask = X[self.target_col].isna()
        if self.groupby_col:
            # Note that imputed_target_col is NOT the whole column
            # It contains only the rows that had missing values, but are now imputed
            imputed_target_col = X.loc[na_mask].apply(
                lambda row: self.imputation_values[ tuple(row.loc[self.groupby_col]) ],
                axis=1
            )
        else:
            imputed_target_col = X.loc[na_mask, self.target_col].fillna(self.imputation_values)

        if self.return_df:
            # imputed_result is DataFrame
            imputed_result.loc[na_mask, self.target_col] = imputed_target_col
        else:
            # imputed_result is Series
            imputed_result.loc[na_mask] = imputed_target_col.values
        
        return imputed_result


class ImputeNumericalByGroup(ImputeByGroup, BaseEstimator, TransformerMixin):
    """ Impute ONE continuous numerical column, optionally after grouping by other (discrete) columns """
    def __init__(self, target_col, **kwargs):
        super().__init__(target_col=target_col, **kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        """ Implement the imputation logic here """
        assert isinstance(X, pd.DataFrame)
        if self.groupby_col:
            self.imputation_values = X.groupby(self.groupby_col)[self.target_col].median()
            self._prepare_imputation_values(standin=X[self.target_col].median())
        else:
            self.imputation_values = X[self.target_col].median()
        return self
    

class ImputeCategoricalByGroup(ImputeByGroup, BaseEstimator, TransformerMixin):
    """ 
        Impute ONE discrete categorical column, optionally after grouping by other (discrete) columns 

        If multiple modes encountered in value_counts()
        select the min value like sklearn.preprocessing.SimpleImputer would 

        Potential TODO: Faster implementation 
        https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
    """

    def __init__(self, target_col, **kwargs):
        super().__init__(target_col=target_col, **kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        """ Implement the imputation logic here """
        assert isinstance(X, pd.DataFrame)
        if self.groupby_col:
            self.imputation_values = (
                X.groupby(self.groupby_col)[self.target_col]
                .value_counts()
                .unstack(level=list(range(len(self.groupby_col))))
                .idxmax()
            )
            # May encounter pandas bugs with the line below            
            self._prepare_imputation_values(standin=X[self.target_col].mode().min())
        else:
            self.imputation_values = X[self.target_col].mode().min()
        return self