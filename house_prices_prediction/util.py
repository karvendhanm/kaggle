from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd


class DropMissingData(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X:pd.DataFrame):
        return X.dropna()

    