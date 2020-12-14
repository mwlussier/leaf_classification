import numpy as np
from sklearn.tree import DecisionTreeClassifier
from models.vanilla_classifier import VanillaClassifier


class DecisionTree(VanillaClassifier):
    """
    Decision Tree Classifier
    ==================
        Child class implementing Decision Tree classifying model.
    Attributes
    ==========
        _criterion       - Function to measure quality of a split
        _data_processing - Type of processed data to use in the training est testing process
    """
    def __init__(self, _criterion='gini', data_process=None):
        super().__init__(DecisionTreeClassifier(criterion=_criterion), data_process=data_process)
        self.parameters = {'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'criterion': ['gini', 'entropy'],
                'max_depth': [50, 100],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [3, 5]
                }

