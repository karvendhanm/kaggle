from scipy.stats import boxcox, kurtosis, skew
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os

for dirname, _, filenames in os.walk('/home/karvsmech/Projects/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.options.display.max_columns = 100
pd.options.display.max_rows = 200
pd.options.display.max_info_rows = 1690785
pd.options.display.max_info_columns = 200
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.date_dayfirst = True
pd.options.mode.chained_assignment = None


def na(df, percent=True, verbose=True):
    na_series = df.isna().sum()
    na_series = na_series.where(na_series > 0).dropna().sort_values(ascending=False)
    if percent:
        if verbose:
            print("% of NaNs in each column")
        return (na_series / df.shape[0]) * 100
    else:
        if verbose:
            print("# of NaNs in each column:")
        return na_series


def get_corr_list(df, num_vars, threshold=0.6):
    df_corr = df[num_vars].corr().unstack().to_frame().reset_index()
    df_corr.columns = ['var1', 'var2', 'corr']

    # removing rows where var1 is equal to var2
    df_corr = df_corr.loc[df_corr['var1'] != df_corr['var2'], :]
    df_corr['abs_corr'] = df_corr['corr'].abs()
    df_corr = df_corr.loc[df_corr['abs_corr'] > threshold, :]
    df_corr.sort_values(by=['abs_corr'], ascending=False, inplace=True)

    # remove mirrored items
    df_corr = df_corr.iloc[::2]

    return df_corr.reset_index()


data_path = '/home/karvsmech/Projects/kaggle/input/house_prices_prediction/'

df = pd.read_csv(data_path + 'train.csv')
dtypes = df.dtypes.to_frame().reset_index()
dtypes.columns = ['col', 'dtype']
dtypes.groupby('dtype').size()

na(df, False)
num_vars = df.select_dtypes(include='number').columns.tolist()
df_corr = get_corr_list(df, num_vars)

cat_cols = df.select_dtypes('O').columns.to_list() + ['MSSubClass']
cat_cols = [col for col in cat_cols if col not in ['Neighborhood', 'Condition1', 'Condition2', 'FireplaceQu']]

# OneHotEncoder
ohencoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
ohencoder.set_output(transform='pandas')

# one hot encoder treats nan as a separate category.
df_cats_encoded = ohencoder.fit_transform(df[cat_cols])
# dropping all the categorical variable that has been "one hot encoded"
df = pd.concat([df.drop(cat_cols, axis=1), df_cats_encoded], axis=1)

# Handling missing values.
na(df, False)

df['LotFrontage'] = df['LotFrontage'].fillna(0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

# Handling missing values.
na(df, False)

# to impute missing values we use the column that has the most correlation with the
# column with missing values
df.select_dtypes(exclude='O').corr()['GarageYrBlt'].sort_values(ascending=False)
df.loc[df['GarageYrBlt'].isna(), 'GarageYrBlt'] = df.loc[df['GarageYrBlt'].isna(), 'YearBuilt']

# Handling Outliers
df.select_dtypes(exclude='O').corr()['SalePrice'].sort_values(ascending=False)
# all outliers

outliers = pd.concat([
    df[(df['OverallQual'] == 4) & (df['SalePrice'] > 2e5)],
    df[(df['OverallQual'] == 8) & (df['SalePrice'] > 5e5)],
    df[(df['OverallQual'] == 10) & (df['SalePrice'] > 7e5)],
    df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 3e5)],
    df[(df['OverallCond'] == 2) & (df['SalePrice'] > 3e5)],
    df[(df['OverallCond'] == 6) & (df['SalePrice'] > 7e5)]
]).sort_index().drop_duplicates()
df = df.drop(outliers.index)
na(df, False)

# Power transformation - skew, kurtosis, Box Cox

skew_kurtosis_data = []
for col in df.select_dtypes(include='number').columns:
    srs = df[col].dropna()
    ln_srs = np.log1p(srs)
    skew_kurtosis_data.append({
        'name': col,
        'normal_skew': skew(srs),
        'ln_skew': skew(ln_srs),
        'normal_kurtosis': kurtosis(srs),
        'ln_kurtosis': kurtosis(ln_srs)
    })

skew_kurtosis_data = pd.DataFrame(skew_kurtosis_data)
skew_kurtosis_data.iloc[:, 1:] = skew_kurtosis_data.iloc[:, 1:].abs()

boxcox_lambdas = []
for col in df.select_dtypes(exclude='O').columns:
    coef = boxcox(df[col] + 1)[1]
    boxcox_lambdas.append({'name': col,
                           'boxcox_coef': coef})
boxcox_lambdas = pd.DataFrame(boxcox_lambdas).set_index('name')
uniques = df.nunique()
boxcox_lambdas.loc[uniques.loc[uniques >= 100].index]

# def lambda_comparison(col):
#     figs, axes = plt.subplots(1, 2, figsize=(8, 4))
#     df[col].hist(bins=50, ax=axes[0])
#     axes[0].title.set_text('no transformation')
#
#     _lambda = boxcox_lambdas.loc[col].values[0]
#     pd.Series(boxcox(df[col] + 1, _lambda)).hist(bins=50, ax=axes[1])
#     axes[1].title.set_text(f'with transformation, a={_lambda:.2f}')
#     figs.suptitle(col)
#     plt.show()
#     return None


# for col in boxcox_lambdas.loc[uniques.loc[uniques >= 100].index].index:
#     lambda_comparison(col)

df['LotArea'].describe()
decided_lambdas_dict = {
    'LotArea': 0, 'MasVnrArea': 0,
    'BsmtFinSF1': 0, 'BsmtUnfSF': .5,
    '1stFlrSF': 0, 'GrLivArea': 0,
    'WoodDeckSF': 0, 'OpenPorchSF': 0,
    'SalePrice': 0
}

for col, lmbda in decided_lambdas_dict.items():
    df[col] = boxcox(df[col] + 1, lmbda)

df['LotArea'].describe()
na(df, False)

# Feature engineering
dt_df = pd.to_datetime(df['MoSold'].astype('str') + '.' + df['YrSold'].astype('str'),
                       format="%m.%Y").rename('date')
dt_df = pd.concat([dt_df, df['SalePrice']], axis=1).set_index('date')
average_price_at_date = dt_df.groupby(pd.Grouper(freq='MS'))['SalePrice'].mean().to_frame().reset_index().rename({
    'SalePrice': 'average_price_at_date'}, axis=1)
average_price_at_date['date'] = average_price_at_date['date'].dt.strftime('%m.%Y')

# figs, axes = plt.subplots(1, 2, figsize=(12, 4))
# average_price_at_date.plot(x='date', y='average_price_at_date', ax=axes[0])

average_price_at_date_smoothed = pd.concat([average_price_at_date['date'],
                                            average_price_at_date['average_price_at_date'].ewm(span=4,
                                                                                               adjust=False).mean()],
                                           axis=1)
# average_price_at_date_smoothed.plot(x='date', y='average_price_at_date', ax=axes[1])
# plt.show()

# average price at time
df['date'] = pd.to_datetime(df['MoSold'].astype('str') + '.' + df['YrSold'].astype('str'),
                            format="%m.%Y").dt.strftime('%m.%Y')
df = df.merge(average_price_at_date_smoothed, on=['date'], how='left')
df.drop('date', axis=1, inplace=True)

# average price at Neighborhood
avg_price_at_neighborhood = df.groupby('Neighborhood',
                                       as_index=False)['SalePrice'].mean().rename(
    {'SalePrice': 'avg_price_at_neighborhood'},
    axis=1)
df = df.merge(avg_price_at_neighborhood, on=['Neighborhood'], how='left')
df.drop('Neighborhood', axis=1, inplace=True)

conditions = list(set(df['Condition1'].unique().tolist() + df['Condition2'].unique().tolist()))
conditions.remove('Norm')

for condition in conditions:
    df[f'Condition_{condition}'] = ((df['Condition1'] == condition) | (df['Condition2'] == condition)).astype('int')
df.drop(['Condition1', 'Condition2'], axis=1)


def process_data(df: pd.DataFrame, return_saleprice: bool = True,
                 mode: str = 'train') -> pd.DataFrame:
    """
    process data
    :param df:
    :param return_saleprice:
    :param mode:
    :return:
    """
    # OHE
    df_cats_encoded = ohencoder.transform(df[cat_cols])
    df = pd.concat([df.drop(labels=cat_cols, axis=1), df_cats_encoded], axis=1)
    # /OHE

    # handling outliers
    if mode == 'train':
        df = df.drop(outliers.index)
    # /handling outliers

    # BoxCox transformation
    for col, _lmbda in decided_lambdas_dict.items():
        if mode == 'test' and col == 'SalePrice':
            continue
        else:
            df[col] = boxcox(df[col] + 1, _lmbda)
    # /BoxCox transformation

    # feature engineering:
    df['date'] = pd.to_datetime(df['MoSold'].astype('str') + '.' + df['YrSold'].astype('str'),
                                format="%m.%Y").dt.strftime("%m.%Y")
    df = df.merge(average_price_at_date_smoothed, how='left', on='date')
    df = df.drop('date', axis=1)

    df = df.merge(avg_price_at_neighborhood, how='left', on='Neighborhood')
    df = df.drop('Neighborhood', axis=1)

    for condition in conditions:
        df[f'Condition_{condition}'] = ((df['Condition1'] == condition) | (df['Condition2'] == condition)).astype('int')
    df = df.drop(['Condition1', 'Condition2'], axis=1)

    df['FireplaceQu'] = df['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0}).astype('int')
    df['FireplaceQu*Fireplaces'] = df['Fireplaces'] * df['FireplaceQu']
    df = df.drop('FireplaceQu', axis=1)

    df['ageAtRemod'] = df['YrSold'] - df['YearRemodAdd']
    df['ageAtSold'] = df['YrSold'] - df['YearBuilt']

    df['SFPerRoomAboveGround'] = df['GrLivArea']/df['TotRmsAbvGrd']
    df['TotalBathrooms'] = df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['SF'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['TotalPorchSF'] + df['GarageArea']
    # /feature engineering

    # standard scaling
    for col in df.columns:
        if col == 'Id':
            continue
        elif col == 'SalePrice':
            saleprice_mean = df[col].mean()
            saleprice_std = df[col].std()

        # this type of scaling is wrong. mean and median should come from the training dataset
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    # /standard scaling

    # missing value imputation
    df.loc[df['GarageYrBlt'].isna(), 'GarageYrBlt'] = df.loc[df['GarageYrBlt'].isna(), 'YrBuilt']

    for col in ['average_price_at_date', 'avg_price_at_neighborhood']:
        df[col] = df[col].fillna(df[col].mean())

    df = df.fillna(0)
    # /missing value imputation

    if return_saleprice:
        return df, saleprice_mean, saleprice_std
    else:
        return df


df = pd.read_csv(data_path + 'train.csv', dtype={'MoSold': 'int', 'YrSold': 'int'})
# df1 = pd.read_csv(data_path + 'AmesHousing.csv', dtype={'MoSold': 'int', 'YrSold': 'int'}).drop('PID', axis=1)
# df1 = df1.rename({'Order': 'Id'}, axis=1)
# df = pd.concat([df, df1], axis=0)
process_data(df)
