import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def to_interim(dataframe, file_name, interim_filepath='data/interim'):
    """
    Interim process to transform dataset.

    """
    ### SAVE TO INTERIM PATH ###
    dataframe.to_csv(interim_filepath + file_name)


def to_processed(train_data, submission_data,
                 train_suffixe, submission_suffixe, processed_filepath='data/processed'):
    """
    Final transformation applied to the dataset.

    """

    X = train_data.drop(['species'], axis=1)  # ['species', 'general_species']
    y = train_data.species

    ### SAVE TO PROCESSED PATH ###
    X.to_csv(processed_filepath + '/x_' + train_suffixe + '.csv')
    y.to_csv(processed_filepath + '/y_' + train_suffixe + '.csv')
    submission_data.to_csv(processed_filepath + '/x_' + submission_suffixe + '.csv')


def to_train_dataset(data_process, processed_filepath='data/processed', test_size=0.25):
    """

    """
    X = pd.read_csv(processed_filepath + '/x_train.csv', index_col='id')
    y = pd.read_csv(processed_filepath + '/y_train.csv', index_col='id')

    # encode label first
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.ravel(y.values))

    #columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='drop', sparse_threshold=0)
    #y_one_hot = np.array(columnTransformer.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(X.values, y_encoded, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test

