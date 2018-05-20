import sys
import pandas as pd
import numpy as np
import pickle
import time

from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

_ = sns.set(style="whitegrid", rc={"figure.figsize": (12, 6),
                                   #                                "legend.fontsize": "large",
                                   "axes.titlesize": "large",
                                   "xtick.labelsize": "large",
                                   "ytick.labelsize": "large",
                                   })


##############################################################
#               HELPER FUNCTIONS
##############################################################


def print_runtime(start_time):
    end_time = time.time()
    run_time = end_time - start_time
    print('Run time: {}h:{}m:{}s'.format(int(run_time / 3600), int(run_time / 60) % 60, int(run_time % 60)))


def save_pickle(object, path_and_name):
    # Saves pickle
    with open(path_and_name, 'wb') as fp:
        pickle.dump(object, fp)
    pass


def open_pickle(path_and_name):
    # Opens pickle - note required 'rb' inside open given above 'wb'
    with open(path_and_name, 'rb') as fp:
        object = pickle.load(fp)
    return object


def prepare_and_save_pickle(size=None, name='df.p'):
    print('Preparing pickle ... took:')
    start_time = time.time()

    # Read data
    df = pd.read_csv('data/ds_test_000000000000')

    # Slim data to make it faster
    if size:
        df = df.iloc[:size]

    # Convert datetime col to datetime obj (takes long! :()
    df['received_time'] = pd.to_datetime(df['received_time'])

    save_pickle(df, 'data/' + name)

    print_runtime(start_time)

    return df


def get_continent_dict():
    df = pd.read_csv('data/country_continent.csv', keep_default_na=False)
    continent_dict = df.set_index('iso 3166 country').to_dict()['continent code']
    return continent_dict


def map_country(x):
    try:
        return continent_dict[x]
    except KeyError:
        return 'other'


def map_currency(x):
    # Major currencies
    if x in ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD']:
        return x
    else:
        return 'other'


def remove_price_outliers(df, column_name):
    # # Remove cases when prices are 0
    # df = df[(df['ota_price'] != 0) & (df['direct_price'] != 0)]
    #
    # # Create currency agnostic measure of disparity (already handled cases of div by zero earlier)
    # df['price_ratio'] = df['ota_price'] / df['direct_price']
    #
    # # Remove abnormally large/small ratios (assumed outliers) - this is about 1.7% of the data
    # df = df[(df['price_ratio'] >= 0.01) & (df['price_ratio'] <= 2)]

    column = df[column_name]
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    filtered = df.query('(@Q1 - 1.5 * @IQR) <= ' + column_name + ' <= (@Q3 + 1.5 * @IQR)')

    mask = df.isin(filtered)[column_name]

    return ~mask


def run_simple(X, y, model):
    print('Running model... took:')

    start_time = time.time()

    # split the data with 80% in train
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    # fit the model on one set of data
    model.fit(X_train, y_train)

    # evaluate the model on the second set of data
    y_test_model = model.predict(X_test)

    print_runtime(start_time)
    print(accuracy_score(y_test, y_test_model))
    pass


##############################################################
#               MAIN
##############################################################

# df = prepare_and_save_pickle()

start_time = time.time()

df = open_pickle('data/df.p')
print('Before anything: ', df.shape)

# #################################
#       DATA CLEANING
# #################################
# Note: using full dataset for this for now, but in practice should only do on train)

# We have missing data, so drop for simplicity drop them now (seems to be in ota_price and user_country)
# NOTE: With more time one could try to impute these
df = df.dropna()

# # Lots of outliers in the prices on a per currency basis, so will apply a remove procedure based on IQR
# grouped = df[['currency', 'direct_price', 'ota_price']].groupby('currency')
# for name, group in grouped:
#     print(name)
#     outlier = (remove_price_outliers(group, 'direct_price')) | (remove_price_outliers(group, 'ota_price'))
#     df.loc[group.index, 'price_outlier'] = np.where(outlier == 1, 1, 0)
# df = df[df['price_outlier'] != 1]

print('After cleaning: ', df.shape)

# #################################
#       DOWNSAMPLING
# #################################
# This is required as we have serious class imbalance for the undercut class

# Create currency agnostic measure of disparity whican can be used for binary target
# (already handled cases of div by zero earlier?)
df['price_ratio'] = df['ota_price'] / df['direct_price']
# This will be the target in our classifier
df['undercut'] = np.where(df['price_ratio'] < 1, 1, 0)

# Separate majority and minority classes
df_majority = df[df['undercut'] == 0]
df_minority = df[df['undercut'] == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,  # sample without replacement
                                   n_samples=df_minority.shape[0],  # to match minority class
                                   random_state=0)  # reproducible results

# Combine minority class with downsampled majority class
df = pd.concat([df_majority_downsampled, df_minority])

# Reset index since we removed lots of rows
df = df.reset_index()

# Display new class counts
print('After downsampling: ', df.shape)
print('New class balance:', df['undercut'].value_counts())

# #################################
#       PREPROC & FEATURE ENG
# #################################

# Get all currencies to upper case so they are not duplicated as cat
df['currency'] = df['currency'].apply(lambda x: x.upper())

# Cast relevant cols as category type
cats = ['client', 'hotel', 'currency', 'ota', 'user_country']
for cat in cats:
    df[cat] = df[cat].astype('category')

# Rename long client and hotel cats for just numbers
for col in ['client', 'hotel']:
    n_col = np.unique(df[col])
    df[col] = df[col].cat.rename_categories(np.arange(1, n_col.shape[0] + 1))


# Rescale the direct price (Note again we are doing this on the whole set instead of train, which is cheating)
min_max_scaler = preprocessing.MinMaxScaler()
grouped = df[['currency', 'direct_price']].groupby('currency')
for name, group in grouped:
    print(name)
    df.loc[group.index, 'direct_price_scaled'] = min_max_scaler.fit_transform(group[['direct_price']])


# ------- TIME -------

# Re-index to received_time so can groupby time and create time features quickly from this
# Note: this is faster than applying lambda func over all rows to get hour, weekday, month and year
temp = df[['received_time']].copy()
temp.index = temp['received_time']

# Cyclic time - get sine and cosine
df['h'] = temp.index.hour
df['w'] = temp.index.weekday
df['m'] = temp.index.month
# Hours numbered 0-23
df['h_sin'] = np.sin(df['h'] * (2. * np.pi / 24))
df['h_cos'] = np.cos(df['h'] * (2. * np.pi / 24))
# Weeks numbered 0-6
df['w_sin'] = np.sin(df['w'] * (2. * np.pi / 7))
df['w_cos'] = np.cos(df['w'] * (2. * np.pi / 7))
# Months numbered 1-12, hence we subtract 1
df['m_sin'] = np.sin((df['m'] - 1) * (2. * np.pi / 12))
df['m_cos'] = np.cos((df['m'] - 1) * (2. * np.pi / 12))

# Non-cyclic time

# Years
df['yr'] = temp.index.year
# Rebase years
df['yr_r'] = np.max(df['yr']) - df['yr']

# ------- AGGREGATIONS -------
# Note: this should be done on train set only
# so technically we are cheating a bit; assumption is dist doesn't change much

# Is the hotel part of a chain?
hotels_per_client = df[['client', 'hotel']].groupby('client').nunique().sort_values('hotel', ascending=False)
hotels_per_client = hotels_per_client.rename(columns={'hotel': 'n_hotels'})
hotels_per_client = hotels_per_client[['n_hotels']]
df = df.join(hotels_per_client, on='client')
df['chain'] = np.where(df['n_hotels'] > 1, 1, 0)

# nunique number of adults per hotel equates to n different rooms:
rooms_per_hotel = df[['hotel', 'adults']].groupby('hotel').nunique().sort_values('adults', ascending=False)
rooms_per_hotel = rooms_per_hotel.rename(columns={'adults': 'n_rooms'})
rooms_per_hotel = rooms_per_hotel[['n_rooms']]
df = df.join(rooms_per_hotel, on='hotel')

searches_by_hotel = df[['hotel', 'received_time']].groupby('hotel').nunique().sort_values('received_time',
                                                                                          ascending=False)
searches_by_hotel = searches_by_hotel.rename(columns={'received_time': 'n_searches'})
searches_by_hotel = searches_by_hotel[['n_searches']]
df = df.join(searches_by_hotel, on='hotel')

# ------- CATEGORICAL -------

# Reduce dimensionality of cat features

continent_dict = open_pickle('data/continent_dict.p')
df['user_continent'] = df['user_country'].apply(lambda x: map_country(x))

df['currency_v2'] = df['currency'].apply(lambda x: map_currency(x))
df['major_currency'] = np.where(df['currency_v2'] != 'other', 1, 0)

print(df.shape)
print_runtime(start_time)

# #################################
#       DEFINE X and y
# #################################

features_dict = {'orig': {
    'cat': {'enc': ['ota'], 'bin': []},
    'num': ['adults', 'children', 'direct_price_scaled']
},
    'eng': {
        'cat': {'enc': ['user_continent', 'currency_v2'], 'bin': ['chain', 'major_currency']},
        'num': ['h_sin', 'h_cos', 'w_sin', 'w_cos', 'm_sin', 'm_cos', 'yr_r',
                'n_hotels', 'n_rooms', 'n_searches']
    }
}

f_num = features_dict['orig']['num'] + features_dict['eng']['num']
X = df[f_num].copy()

f_bin = features_dict['orig']['cat']['bin'] + features_dict['eng']['cat']['bin']
X[f_bin] = df[f_bin]

print(X.shape)

f_enc = features_dict['orig']['cat']['enc'] + features_dict['eng']['cat']['enc']
for col in f_enc:
    print(col)
    prefix = col[:3] if 'continent' not in col else 'cont'
    X = X.join(pd.get_dummies(df[col], prefix=prefix))
    print(X.shape)

# Binary target
y = df['undercut']

print(X.shape, y.shape)

# #################################
#       MODEL
# #################################

model = LogisticRegression(random_state=0)
run_simple(X, y, model)
