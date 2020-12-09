import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from src.data.util_dataset import to_train_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline


class VanillaClassifier:
    """
    Vanilla Classifier
    ==================
        Base class being inherited in every child classifiers.

    Attributes
    ==========
        model   - sklearn model used for training and predictions
        x_train - training variables
        y_train - training labels

    """

    def __init__(self, model):
        self.model = model
        self.X_train, self.y_train, self.X_test, self.y_test = to_train_dataset()
        self.best_parameters = {}

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self, X):
        return self.model.predict(X)

    def evaluate(self):
        pred_train = self.prediction(self.X_train)
        err_train = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(self.y_train, pred_train)])

        pred_test = self.prediction(self.X_test)
        err_test = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(self.y_test, pred_test)])

        print('Erreur train = ', str(round(err_train.mean() * 100, 5)), '%')
        print('Erreur test = ', str(round(err_test.mean() * 100, 5)), '%')
        analyse_erreur(err_train.mean(), err_test.mean())

    def erreur(self, y, prediction, type='classification'):
        """
        y: true label
        prediction: model prediction
        type: model type
        
        Classification
        --------------
            Return True or False depending if y == prediction

        Regression
        ----------
            Return the squared difference of y and the prediction
        """
        if type == 'classification':
            return y != prediction

        if type == 'regression':
            return (y - prediction) ** 2

    def pipeline(self, param_grid, kfold=5, score_metrics='accuracy', refit=True):
        """
        Stratified KFold (default=5): Generate test sets such that all contain the
            same distribution of classes, or as close as possible.
        Grid Search Cross-Validation
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        """
        cross_validation = StratifiedKFold(n_splits=kfold, shuffle=True)
        gs = GridSearchCV(estimator=self.model,
                          param_grid=param_grid,
                          scoring=score_metrics,
                          refit=refit,
                          cv=cross_validation)

        gs = gs.fit(self.X_train, self.y_train)
        print("Best Score: ", gs.best_score_)
        print("Best Parameters: ", gs.best_params_)
        self.best_parameters = gs.best_params_


def analyse_erreur(err_train, err_test, threshold=0.3):
    """
    Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
    apprentissage
    """
    #AJOUTER CODE ICI
    # Utilisation d'un threshold arbitraire de 30% qui signale lorsqu'il y a sur ou sous apprentissage
    if (err_test - err_train) > threshold:
        print("WARNING: SUR-apprentissage")

    if (err_train > threshold):
        print("WARNING: SOUS-apprentissage")