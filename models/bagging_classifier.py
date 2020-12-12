from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from models.vanilla_classifier import VanillaClassifier


class bagging(VanillaClassifier):
    """
    Bagging Classifier
    ==================
        Child class implementing Bagging classifying model.
    Attributes
    ==========
        _estimators -
        _voting      -
    """
    def __init__(self, _estimators=[('SVM', SVC(probability=True)), ('GBoost', GradientBoostingClassifier()),
                                    ('fc_relu', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu'))],
                 _voting='soft', data_process=None):
        super().__init__(VotingClassifier(estimators=_estimators, voting=_voting), data_process=data_process)
        self.parameters = {'estimators': _estimators, 'voting': _voting}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {}

