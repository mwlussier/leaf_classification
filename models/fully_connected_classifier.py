from sklearn.neural_network import MLPClassifier
from models.vanilla_classifier import VanillaClassifier


class FullyConnected(VanillaClassifier):
    """
    MLP Classifier
    ==================
        Child class implementing Multi-Layer Perceptron classifying model.
    Attributes
    ==========
        _hidden_layer_sizes - The ith element represents the number of neurons in the ith hidden layer
        _activation         - Activation function for the hidden layer
        _solver             - Solver for weight optimization
        _alpha              - L2 regularization term
        _learning_rate      - Learning rate schedule (only use when solver='sgd')
        _learning_rate_init - Initial learning rate used
        _max_iter           - Maximum number of iteration
        _data_processing    - Type of processed data to use in the training est testing process
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
