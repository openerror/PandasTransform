# Useful `scikit-learn` transformers for working with Pandas DataFrames

## Motivation
In addition to providing common machine learning algorithms, scikit-learn allows users to build
reusable [*pipelines*](https://scikit-learn.org/stable/modules/compose.html#pipeline) that integrate data processing and model building steps into one object. Each step in the pipeline object consists of a [Transformer](https://scikit-learn.org/stable/data_transforms.html) instance, which exposes the easy-to-use `fit`/`transform` API.

Unfortunately, [scikit-learn works directly with numpy arrays or scipy sparse arrays, but not `pandas.DataFrame`](https://scikit-learn.org/stable/faq.html#why-does-scikit-learn-not-directly-work-with-for-example-pandas-dataframe) which is widespread in the data science community. We can get around the problem by subclassing the `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin` and writing our own transformers, such that they properly utilize or produce `pandas.DataFrame`. As a result, for example, it becomes easier to bookkeep the names of each feature (column) after each step in the pipeline --- because `pandas.DataFrame` contains the relevant metadata. Also, for a `pandas.DataFrame` we can easily compute conditional statistics --- e.g. when we impute missing values in a numerical column using medians computed from *distinct* segments (rows) of the data.

### What about `ColumnTransformer` in scikit-learn?


## Functionality Implemented


## Usage
Copy the source file(s) of interest from the `src` directory into your own project, and then import as necessary. Please see `playground.ipynb` for a usage demonstration.