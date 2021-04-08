# Useful `scikit-learn` transformers for working with Pandas DataFrames
[Description](https://towardsdatascience.com/making-scikit-learn-work-better-with-pandas-13d197e60dc9) on Towards Data Science.

## Quick --- What is Here?
1. Code for processing `pandas.DataFrame` that is compatible with `sklearn.pipeline`
2. High code coverage by unit tests (via [pytest](https://docs.pytest.org/en/stable/#))
3. Continuous integration via [GitHub Actions](https://docs.github.com/en/actions) as I add more functionality

## Motivation
Make scikit-learn pipelines retain and remember metadata, e.g. column names, from pandas DataFrames. Facilitates model debugging and interpretation! For details, see my article on [Towards Data Science](https://towardsdatascience.com/making-scikit-learn-work-better-with-pandas-13d197e60dc9).

### What about `ColumnTransformer` in scikit-learn?
To be fair, there is one way that scikit-learn utilizes metadata in DataFrames: [ColumnTransformer]((https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)) can identify DataFrame columns by their string names, and directs your desired transformers to each column. Here is an [example](https://towardsdatascience.com/using-columntransformer-to-combine-data-processing-steps-af383f7d5260) by Allison Honold on Towards Data Science.

Unfortunately, `ColumnTransformer` produces numpy arrays or scipy sparse matrices. My code extends `ColumnTransformer` such that it produces pandas.DataFrame as well.

## Functionality Implemented
All files are located within directory `src`.
- `ImputeByGroup.py`
    - `ImputeNumericalByGroup`; [`groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html), calculate per-group statistics (e.g. median) of a **numerical** column with missing values, and then impute each group using said statistics.
    - `ImputeCategoricalByGroup`; like the above, but for imputing discrete, categorical columns. Fills up missing values using the most frequent unique value.

- `PandasColumnTransformer.py`
    - Wrapper around `sklearn.compose.ColumnTransformer` for automatic bookkeeping of column names, even when the number of columns changed after a transformation (e.g. one-hot encoding)

## Usage
Copy the source file(s) of interest from the `src` directory into your own project, and then import as necessary. Please see `playground.ipynb` for a usage demonstration.