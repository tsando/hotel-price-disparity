import pandas as pd
import numpy as np
import pickle
import time

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

    # Drop missing values for simplicity now (seem to be in ota_price and user_country)
    df = df.dropna()

    # Slim data to make it faster
    if size:
        df = df.iloc[:size]

    # Convert datetime col to datetime obj (takes long! :()
    df['received_time'] = pd.to_datetime(df['received_time'])

    # Cast relevant cols as category type
    cats = ['client', 'hotel', 'currency', 'ota', 'user_country']
    for cat in cats:
        df[cat] = df[cat].astype('category')

    # Rename long client and hotel cats for just numbers
    for col in ['client', 'hotel']:
        n_col = np.unique(df[col])
        df[col] = df[col].cat.rename_categories(np.arange(1, n_col.shape[0] + 1))

    save_pickle(df, 'data/' + name)

    print_runtime(start_time)

    return df


##############################################################
#               MAIN
##############################################################

# df = prepare_and_save_pickle()

df = open_pickle('data/df.p')
