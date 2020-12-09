import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.vanilla_classifier import VanillaClassifier


class random_forest(VanillaClassifier):
    """
    Random Forest Classifier
    ==================
        Child class implementing Random Forest classifying model.
    Attributes
    ==========
        _n_estimators -
        _criterion      -
    """
    def __init__(self, _n_estimators=100, _criterion='gini'):
        super().__init__(RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion))
        self.parameters = {'n_estimators': _n_estimators, 'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return [{'n_estimators': [50, 100, 200, 400, 600],
                 'criterion': ['gini', 'entropy'],
                 'max_depth': [4, 8, 16, 24],
                 'min_samples_split': [2, 3, 5],
                 'min_samples_leaf': [1, 3],
                 'warm-start': [True, False]
                 }]

