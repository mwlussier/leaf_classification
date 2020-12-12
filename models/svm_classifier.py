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
        _data_processing
    """
    def __init__(self, _kernel='rbf', _c=1, _gamma=1, _degree=3, data_process=None):
        super().__init__(SVC(probability=True, kernel=_kernel, gamma=_gamma, C=_c, degree=_degree),
                         data_process=data_process)
        self.parameters = {'kernel': _kernel, 'gamma': _gamma, 'C': _c, 'degree': _degree}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'degree': [1, 3, 5, 7, 9],
                'kernel': ['rbf', 'poly', 'sigmoid']
                }

