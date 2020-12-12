import sys
import pandas as pd
from models.bagging_classifier import bagging
from models.decision_tree_classifier import decision_tree
from models.fully_connected_classifier import fully_connected
from models.gradient_boosting_classifier import gradient_boosting
from models.logistic_regression_classifier import logistic_regression
from models.random_forest_classifier import random_forest
from models.svm_classifier import svm_classifier


def main():
    """
    python3 cross_valuation.py <pipeline> <data_processing>

    pipeline: simple | cross_validation
    data_processing: <Empty> | simple | fselection | pca50 | pca100 | pca150
    """
    pipeline = "cross_validation" #sys.argv[1]
    data_process = "simple" #sys.argv[2]

    models = {'bagging': bagging(data_process=data_process),
              'decision_tree': decision_tree(data_process=data_process),
              'fconnected': fully_connected(data_process=data_process),
              'gboost': gradient_boosting(data_process=data_process),
              'logit': logistic_regression(data_process=data_process),
              'random_forest': random_forest(data_process=data_process),
              'svm': svm_classifier(data_process=data_process)
              }

    cross_valuation = []
    for _name, _model in models.items():
        print("Model valuation: [" + _name.upper() + "]")
        if pipeline == 'simple':
            _model.training()
            best_score = _model.evaluate()
            parameters = _model.parameters
            cross_valuation.append({'model': _name, 'score': best_score, 'parameters': parameters})
        else:
            gs_best_score, gs_best_parameters, best_score = _model.pipeline(_model.param_grid)
            cross_valuation.append({'model': _name, 'score': best_score,
                                    'gs_parameters': gs_best_parameters, 'gs_score': gs_best_score})

    pd.DataFrame(cross_valuation).to_csv('reports/cross_valuation/cross_valuation_'+data_process + '.csv')


if __name__ == "__main__":
    main()
