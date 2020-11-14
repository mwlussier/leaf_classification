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
    def __init__(self, _kernel='rbf', _c=1, _gamma=1):
        super().__init__(SVC(probability=True, kernel=_kernel, gamma=_gamma, C=_c))
