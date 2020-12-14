import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from models.vanilla_classifier import VanillaClassifier


class GradientBoosting(VanillaClassifier):
    """
    GBoost Classifier
    ==================
        Child class implementing Gradient Boosting (GBoost) classifying model.
    Attributes
    ==========
        _loss            - Loss function to be optimized
        _learning_rate   - Learning rate shrinkage
        _n_estimators    - Number of boosting stages to perform
        _criterion       - Function to measure the quality of a split
        _data_processing - Type of processed data to use in the training est testing process
    """
    def __init__(self, _loss='deviance', _learning_rate=0.1, _n_estimators=100, _criterion='friedman_mse',
                 data_process=None):
        super().__init__(GradientBoostingClassifier(loss=_loss, learning_rate=_learning_rate,
                                                    n_estimators=_n_estimators, criterion=_criterion),
                         data_process=data_process)
        self.parameters = {'loss': _loss, 'learning_rate': _learning_rate,
                           'n_estimators': _n_estimators, 'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'loss': ['deviance'],
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [100, 200],
                'max_depth': [4, 8, 16],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 3]
                }

