import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# internal modules
from house_prices_prediction import config
from house_prices_prediction import util

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

X_train = pd.read_csv(config.datapath + 'train.csv')
categorical_vars = X_train.select_dtypes(include='O').columns.to_list()
print(f'length of the categorical variables is: {len(categorical_vars)}\n')
print(categorical_vars)

numerical_vars = X_train.select_dtypes(include='number').columns.to_list()
print(f'length of the numerical variables is: {len(numerical_vars)}\n')
print(numerical_vars)

before = X_train[numerical_vars].isna().mean().sort_values(ascending=False)


# missing value imputation
num_var_pipeline = Pipeline(
    [
        ('num_selector', util.DataFrameSelector(numerical_vars)),
        ('mean_imputer', SimpleImputer(strategy='mean'))
    ]
)

num_X_train = num_var_pipeline.fit_transform(X_train)
num_X_train = pd.DataFrame(num_X_train, columns=numerical_vars)
after = num_X_train.isna().mean().sort_values(ascending=False)
print('this is just for debugging')
