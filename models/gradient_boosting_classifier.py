import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from models.vanilla_classifier import VanillaClassifier


class gradient_boosting(VanillaClassifier):
    """
    GBoost Classifier
    ==================
        Child class implementing Gradient Boosting (GBoost) classifying model.
    Attributes
    ==========
        _loss           -
        _learning_rate  -
        _n_estimators   -
        _criterion      -
    """
    def __init__(self, _loss='deviance', _learning_rate=0.01, _n_estimators=100, _criterion='friedman_mse',
                 data_process=False):
        super().__init__(GradientBoostingClassifier(loss=_loss, learning_rate=_learning_rate,
                                                    n_estimators=_n_estimators, criterion=_criterion),
                         data_process=data_process)
        self.parameters = {'loss': _loss, 'learning_rate': _learning_rate,
                           'n_estimators': _n_estimators, 'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'loss': ['deviance', 'exponential'],
                 'learning_rate': [1e-5, 0.001, 0.01],
                 'n_estimators': [50, 100, 200, 400, 600],
                 'criterion': ['friedman_mse', 'mse', 'mae'],
                 'max_depth': [4, 8, 16, 24],
                 'min_samples_split': [2, 3, 5],
                 'min_samples_leaf': [1, 3]
                 }

