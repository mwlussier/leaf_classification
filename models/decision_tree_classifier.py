import numpy as np
from sklearn.tree import DecisionTreeClassifier
from models.vanilla_classifier import VanillaClassifier


class decision_tree(VanillaClassifier):
    """
    Decision Tree Classifier
    ==================
        Child class implementing Decision Tree classifying model.
    Attributes
    ==========
        _criterion      -
    """
    def __init__(self, _criterion='gini', data_process=None):
        super().__init__(DecisionTreeClassifier(criterion=_criterion), data_process=data_process)
        self.parameters = {'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'criterion': ['gini', 'entropy'],
                'max_depth': [6, 8, 10]
                }

