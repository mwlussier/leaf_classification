from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.models.vanilla_classifier import VanillaClassifier


class Bagging(VanillaClassifier):
    """
    Bagging Classifier
    ==================
        Child class implementing Bagging classifying model.
    Attributes
    ==========
        _estimators      - List of classifier to use in the bagging
        _voting          - Soft: Predicts the class label based on the argmax of the sums of the predicted probabilities
        _data_processing - Type of processed data to use in the training est testing process
    """
    def __init__(self, _estimators=[('SVM', SVC(probability=True)), ('GBoost', GradientBoostingClassifier()),
                                    ('fc_relu', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu'))],
                 _voting='soft', data_process=None):
        super().__init__(VotingClassifier(estimators=_estimators, voting=_voting), data_process=data_process)
        self.parameters = {'estimators': _estimators, 'voting': _voting}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'weights': [[1, 1, 1], [1, 0, 1]]}

