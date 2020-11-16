import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from src.data.util_dataset import to_train_dataset
import numpy as np
from sklearn.model_selection import GridSearchCV
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

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self, X):
        return self.model.predict(X)

    def evaluate(self):
        pred_train = self.prediction(self.X_train)
        err_train = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(self.y_train, pred_train)])

        pred_test = self.prediction(self.X_test)
        err_test = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(self.y_test, pred_test)])

        print('Erreur train = ', err_train.mean(), '%')
        print('Erreur test = ', err_test.mean(), '%')
        analyse_erreur(err_train.mean(), err_test.mean())

    def erreur(self, y, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        #return (y - prediction) ** 2
        return (y != prediction)

    def pipeline(self, param_grid, score_metrics='accuracy', refit=True):
        """
        Grid Search Cross-Validation (cv=5)
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        """
        #kfold = KFold(n_splits=10, random_state=22)
        gs = GridSearchCV(estimator=self.model,
                          param_grid=param_grid,
                          scoring=score_metrics,
                          refit=refit,
                          cv=5)

        gs = gs.fit(self.X_train, self.y_train)
        print("Best Score: ", gs.best_score_)
        print("Best Parameters: ", gs.best_params_)


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