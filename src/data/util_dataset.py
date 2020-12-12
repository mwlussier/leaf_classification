import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def to_interim(dataframe, file_name, seperate_label=False, interim_filepath='data/interim'):
    """
    Interim process to transform dataset.
    Perform an initial separation of features and labels prior to processing.
    """
    ### SAVE TO INTERIM PATH ###
    dataframe.to_csv(interim_filepath + file_name)
    if seperate_label:
        X = dataframe.drop(['species'], axis=1)
        y = dataframe.species

        ### SAVE TO INTERIM PATH ###
        X.to_csv(interim_filepath + '/x_train.csv')
        y.to_csv(interim_filepath + '/y_train.csv')


def to_processed(train_data, submission_data,
                 train_suffixe, submission_suffixe, processed_filepath='data/processed'):
    """
    Final transformation applied to the dataset to separate features and labels into two distinct files.
    """

    X = train_data.drop(['species'], axis=1)  # ['species', 'general_species']
    y = train_data.species

    ### SAVE TO PROCESSED PATH ###
    X.to_csv(processed_filepath + '/x_' + train_suffixe + '.csv')
    y.to_csv(processed_filepath + '/y_' + train_suffixe + '.csv')
    submission_data.to_csv(processed_filepath + '/x_' + submission_suffixe + '.csv')


def to_train_dataset(data_process, test_size=0.20):
    """
        We are using 'StratifiedShuffleSplit' to keep a better distribution of the different target we have.
        By separating into equally distribution and than aggregating to perform our training and testing set,
        we raise our chance to have a good representation into both set.
    """
    if data_process:
        X = pd.read_csv('data/processed/x_train.csv', index_col='id')
        y = pd.read_csv('data/processed/y_train.csv', index_col='id')
    else:
        X = pd.read_csv('data/interim/x_train.csv', index_col='id')
        y = pd.read_csv('data/interim/y_train.csv', index_col='id')

    # Encode Label into numeric value
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(np.ravel(y.values))

    # Using Stratified Split to get a good representation of every classes
    ss_split = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=42)
    ss_split.get_n_splits(X, y_encoded)

    # Aggregate the stratified split data into a training and testing set
    for train_index, test_index in ss_split.split(X, y_encoded):
        X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    return X_train, y_train, X_test, y_test

