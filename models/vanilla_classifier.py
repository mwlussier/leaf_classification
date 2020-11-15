import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from src.data.util_dataset import to_train_dataset
import numpy as np


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

    def pipeline(self):
        pass


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