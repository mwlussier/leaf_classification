import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))

from sklearn.svm import SVC
from models.vanilla_classifier import VanillaClassifier

class svm_classifier(VanillaClassifier):
    """
    SVM Classifier
    ==================
        Child class implementing support vector machine (SVM) classifying model.
    Attributes
    ==========
        _kernel -
        _c      -
        _gamma  -
    """
    def __init__(self, _kernel='rbf', _c=1, _gamma=1, _degree=3):
        super().__init__(SVC(probability=True, kernel=_kernel, gamma=_gamma, C=_c, degree=_degree))
        self.parameters = {'kernel': _kernel, 'gamma': _gamma, 'C': _c, 'degree': _degree}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        # svc__C --> refer to a pipeline label 'svc' (need to create a pipeline)
        # self.param_grid = [{'svc__C': param_C,
        #                     'svc__gamma': param_gamma,
        #                     'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']}]
        return [{'C': [0.0001, 0.001, 0.01, 0.1, 1.0],
                 'gamma': [0.1, 1.0, 10.0, 100.0],
                 'degree': [1, 3, 5, 7, 9],
                 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                 }]

