import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def leaf_class_reduction(train_data):
    """
    Reduce the numbers of class to predict by grouping each leaf into their more general classification.
    """
    train_data['general_species'] = train_data['species'].str.split('_').str[0]


def complete_preprocessing(train_data, submission_data):
    """
    Apply full preprocessing operation to the training and to submit data.

    Include
    -------
    Standardization (Z-score)
    PCA(n_components=50) - value found from exploration (notebook)

    """
    train = train_data.copy()
    train_target = train.species
    train.drop(['species'], axis=1, inplace=True)

    submission = submission_data[train.columns].copy()

    standard_scaler = StandardScaler().fit(train)
    train_scaled = pd.DataFrame(standard_scaler.transform(train),
                                columns=train.columns, index=train.index)
    train_scaled['species'] = train_target
    submission_scaled = pd.DataFrame(standard_scaler.transform(submission),
                                     columns=submission.columns, index=submission.index)

    return train_scaled, submission_scaled
