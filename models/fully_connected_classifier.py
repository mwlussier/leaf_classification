from sklearn.neural_network import MLPClassifier
from models.vanilla_classifier import VanillaClassifier


class fully_connected(VanillaClassifier):
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



The number of hidden neurons should be between the size of the input layer and the size of the output layer.
The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
The number of hidden neurons should be less than twice the size of the input layer.
see this https://www.heatonresearch.com/2017/06/01/hidden-layers.html
    """

    def __init__(self, _hidden_layer_sizes=(100,), _activation='relu', _solver='adam', _alpha=0.0001,
                 _learning_rate='constant', _learning_rate_init=0.001, _max_iter=200, data_process=None):
        super().__init__(MLPClassifier(hidden_layer_sizes=_hidden_layer_sizes, activation=_activation, solver=_solver,
                                       alpha=_alpha, learning_rate=_learning_rate,
                                       learning_rate_init=_learning_rate_init), data_process=data_process)
        self.parameters = {'hidden_layer_sizes': _hidden_layer_sizes, 'activation': _activation, 'solver': _solver,
                           'alpha': _alpha, 'learning_rate': _learning_rate, 'learning_rate_init': _learning_rate_init,
                           'max_iter': _max_iter}
        self.param_grid = self.get_param_grid()

    def get_param_grid(self):
        return {'hidden_layer_sizes': [(128,), (128, 64)],
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [1e-5, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [1e-5, 0.001, 0.01],
                'max_iter': [200, 400],
                'early_stopping': [True, False]
                }
