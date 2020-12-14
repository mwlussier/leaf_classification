import sys
import pandas as pd
from src.models.bagging_classifier import Bagging
from src.models.decision_tree_classifier import DecisionTree
from src.models.fully_connected_classifier import FullyConnected
from src.models.gradient_boosting_classifier import GradientBoosting
from src.models.logistic_regression_classifier import Logit
from src.models.random_forest_classifier import RandomForest
from src.models.svm_classifier import SvmClassifier


def main():
    """
    python3 cross_valuation.py <pipeline> <data_processing>

    pipeline: simple | cross_validation
    data_processing: <Empty> | simple | fselection | pca_50 | pca_100 | pca_150

    'gboost': GradientBoosting(data_process=data_process)
    """
    pipeline = "cross_validation" #sys.argv[1]
    data_process = "pca_100" #sys.argv[2]

    # models = {
    #            'decision_tree': decision_tree(data_process=data_process)
    #           }

    models = {'bagging': Bagging(data_process=data_process),
              'decision_tree': DecisionTree(data_process=data_process),
              'fconnected': FullyConnected(data_process=data_process),
              'logit': Logit(data_process=data_process),
              'random_forest': RandomForest(data_process=data_process),
              'svm': SvmClassifier(data_process=data_process)
              }

    cross_valuation = []
    for _name, _model in models.items():
        print("Model valuation: [" + _name.upper() + "]")
        if pipeline == 'simple':
            _model.training()
            accuracy_train, accuracy_test, train_loss, test_loss = _model.evaluate()
            parameters = _model.parameters
            cross_valuation.append({'model': _name, 'accuracy_train': accuracy_train,
                                    'train_loss': train_loss, 'test_loss': test_loss,
                                    'accuracy_test': accuracy_test, 'parameters': parameters})
        else:
            gs_best_score, gs_best_parameters, accuracy_train, accuracy_test, train_loss, test_loss = (
                _model.pipeline(_model.param_grid, score_metrics='neg_log_loss'))

            cross_valuation.append({'model': _name, 'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test,
                                    'train_loss': train_loss, 'test_loss': test_loss,
                                    'gs_parameters': gs_best_parameters, 'gs_score': gs_best_score})

    pd.DataFrame(cross_valuation).to_csv('reports/cross_valuation/cross_valuation_'+data_process + '.csv')


if __name__ == "__main__":
    main()
