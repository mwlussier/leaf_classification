from sklearn.ensemble import RandomForestClassifier
from src.models.vanilla_classifier import VanillaClassifier


class RandomForest(VanillaClassifier):
    """
    Random Forest Classifier
    ==================
        Child class implementing Random Forest classifying model.
    Attributes
    ==========
        _n_estimators    - Number of trees in the forest
        _criterion       - Function to measure the quality of a split
        _data_processing - Type of processed data to use in the training est testing process
    """
    def __init__(self, _n_estimators=100, _criterion='gini', data_process=None):
        super().__init__(RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion),
                         data_process=data_process)
        self.parameters = {'n_estimators': _n_estimators, 'criterion': _criterion}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'n_estimators': [100, 200, 400, 600],
                'criterion': ['gini', 'entropy'],
                'max_depth': [4, 8, 16, 24],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 3]
                }

