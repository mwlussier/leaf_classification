import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    train_scaled = standard_scaler.transform(train)
    submission_scaled = standard_scaler.transform(submission)
    train_pca, submission_pca = pca_decomposition(train_scaled, submission_scaled, 50)
    train_pca.index = train.index
    submission_pca.index = submission.index
    # train_scaled = pd.DataFrame(train_pca,
    #                             columns=train.columns, index=train.index)
    train_pca['species'] = train_target

    # submission_scaled = pd.DataFrame(standard_scaler.transform(submission),
    #                                  columns=submission.columns, index=submission.index)

    return train_pca, submission_pca


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
