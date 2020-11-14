import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))

import pandas as pd
from sklearn.model_selection import train_test_split

def to_interim(train_data, submission_data, interim_filepath):
    """
    Interim process to transform dataset.

    """
    ### SAVE TO INTERIM PATH ###
    train_data.to_csv(interim_filepath + '/train.csv')
    submission_data.to_csv(interim_filepath + '/x_submission.csv')


def to_processed(train_data, submission_data, processed_filepath):
    """
    Final transformation applied to the dataset.

    """

    X = train_data.drop(['species'], axis=1)
    y = train_data.species

    ### SAVE TO PROCESSED PATH ###
    X.to_csv(processed_filepath + '/x_train.csv')
    y.to_csv(processed_filepath + '/y_train.csv')
    submission_data.to_csv(processed_filepath + '/x_submission.csv')


def to_train_dataset(path='data/processed', test_size=0.25):
    """

    """
    X = pd.read_csv(path + '/x_train.csv', index_col='id')
    y = pd.read_csv(path + '/y_train.csv', index_col='id')

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test

