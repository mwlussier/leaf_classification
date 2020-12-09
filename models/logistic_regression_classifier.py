import numpy as np
from sklearn.linear_model import LogisticRegression
from models.vanilla_classifier import VanillaClassifier

class logistic_regression(VanillaClassifier):
    """
    LOGIT Classifier
    ==================
        Child class implementing Logistic Regression (LOGIT) classifying model.
    Attributes
    ==========
        _penalty -
        _solver      -
        _c  -
    """
    def __init__(self, _penalty='l2', _solver='newton-cg', _c=100, _max_iter=100, data_process=False):
        super().__init__(LogisticRegression(penalty=_penalty, solver=_solver, C=_c, max_iter=_max_iter),
                         data_process=data_process)
        self.parameters = {'penalty': _penalty, 'solver': _solver, 'C': _c, 'max_iter': _max_iter}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'penalty': ['l2'],
                 'solver': ['newton-cg', 'lbfgs', 'saga'],
                 'C': np.logspace(0, 4, 6),
                 'max_iter': [200, 400]
                 }

