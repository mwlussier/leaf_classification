from sklearn.neural_network import MLPClassifier
from models.vanilla_classifier import VanillaClassifier


class FullyConnectedClassifier(VanillaClassifier):
    """
    MLP Classifier
    ==================
        Child class implementing Multi-Layer Perceptron classifying model.
    Attributes
    ==========
        _hidden_layer_sizes -
        _activation      -
        _solver
        _alpha
        _learning_rate
        _learning_rate_init
        _max_iter
    """

    def __init__(self, _hidden_layer_sizes=100, _activation='relu', _solver='adam', _alpha=0.0001,
                 _learning_rate='adaptive', _learning_rate_init=0.001, _max_iter=200):
        super().__init__(MLPClassifier(hidden_layer_sizes=_hidden_layer_sizes, activation=_activation, solver=_solver,
                                       alpha=_alpha, learning_rate=_learning_rate,
                                       learning_rate_init=_learning_rate_init))
        self.parameters = {'hidden_layer_sizes': _hidden_layer_sizes, 'activation': _activation, 'solver': _solver,
                           'alpha': _alpha, 'learning_rate': _learning_rate, 'learning_rate_init': _learning_rate_init,
                           'max_iter': _max_iter}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return [{'hidden_layer_sizes': [(5,), (5, 5)],
                 'activation': ['logistic', 'relu'],
                 'solver': ['sgd', 'adam'],
                 'alpha': [1e-5, 0.001, 0.01],
                 'learning_rate': ['constant', 'adaptive'],
                 'learning_rate_init': [1e-5, 0.001, 0.01],
                 'max_iter': [200, 400],
                 'early_stopping': [True, False]
                 }]
