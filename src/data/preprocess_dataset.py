import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def complete_preprocessing(train_data, submission_data, pca=None, features_selection=False):
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
    if features_selection:
        train, submission = remove_useless_features(train, submission)

    standard_scaler = StandardScaler().fit(train)
    train_scaled = standard_scaler.transform(train)
    submission_scaled = standard_scaler.transform(submission)

    if pca is not None:
        train_pca, submission_pca = pca_decomposition(train_scaled, submission_scaled, pca)
        train_pca.index = train.index
        submission_pca.index = submission.index
        train_pca['species'] = train_target
        return train_pca, submission_pca

    train_scaled = pd.DataFrame(train_scaled,
                                columns=train.columns, index=train.index)
    submission_scaled = pd.DataFrame(submission_scaled,
                                     columns=submission.columns, index=submission.index)
    train_scaled['species'] = train_target
    return train_scaled, submission_scaled


def pca_decomposition(train, submission, n_components):
    """
        Apply PCA decomposition to reduce dimensionality.
        For the sake of this study, we will be using a decomposition to 50 and 100 features.
        PCA(30)  : ~ 84.5% explained
        PCA(50)  : ~ 91.6% explained
        PCA(100) : ~ 98.1% explained
        PCA(150) : ~ 99.9% explained
    """
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca.fit(train)
    train_data = pca.transform(train)
    submission_data = pca.transform(submission)

    train_pca = pd.DataFrame(train_data, columns=['pc' + str(_) for _ in np.arange(1, n_components + 1)])
    submission_pca = pd.DataFrame(submission_data, columns=['pc' + str(_) for _ in np.arange(1, n_components + 1)])

    print(f'With {n_components} principal components you explained is:{pca.explained_variance_ratio_.sum()}')
    return train_pca, submission_pca


def remove_useless_features(train, submission, pourc_of_db=0.30):
    """
        Apply manual dimension reduction on features having more than 50% (default) equal to zero(0).
    """

    mask = ((train == 0).sum() / train.shape[0]) > pourc_of_db
    nb_useless = mask.sum()
    print("Number of features having more than {0}% of their values equal to  zero(0): {1}".format(pourc_of_db*100,
                                                                                                   nb_useless))
    return train.loc[:, ~mask], submission.loc[:, ~mask]
