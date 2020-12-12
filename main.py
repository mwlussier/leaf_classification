# import os
import sys
# sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import numpy as np
from models.bagging_classifier import bagging
from models.decision_tree_classifier import decision_tree
from models.fully_connected_classifier import fully_connected
from models.gradient_boosting_classifier import gradient_boosting
from models.logistic_regression_classifier import logistic_regression
from models.random_forest_classifier import random_forest
from models.svm_classifier import svm_classifier



def main():
    """
    python3 main.py <model> <pipeline> <metrics> <data_processing> ...<visualisation>

    model : bagging | decision_tree | fconnected | gboost | logit | random_forest | svm
    pipeline: simple | cross_validation
    cv_metrics: accuracy | roc_auc
    evaluate: simple | confusion_matrix | report | roc_auc
    data_processing: 0 | 1
    """

    model = fully_connected(data_process=True)
    #model = bagging(data_process=True)
    model.pipeline(model.param_grid, k_fold=5)
    model.training()
    model.evaluate()

    if sys.argv[5] == '1':
        data_process = True

    #  ~~~ Bagging ~~~
    if sys.argv[1] == 'bagging':
        model = bagging(data_process=data_process)
    #  ~~~ Decision Tree ~~~
    if sys.argv[1] == 'decision_tree':
        model = decision_tree(data_process=data_process)
    #  ~~~ Fully Connected ~~~
    if sys.argv[1] == 'fconnected':
        model = fully_connected(data_process=data_process)
    #  ~~~ GBoost ~~~
    if sys.argv[1] == 'gboost':
        model = gradient_boosting(data_process=data_process)
    #  ~~~ LOGIT ~~~
    if sys.argv[1] == 'logit':
        model = logistic_regression(data_process=data_process)
    #  ~~~ Random Forest ~~~
    if sys.argv[1] == 'random_forest':
        model = random_forest(data_process=data_process)
    #  ~~~ SVM ~~~
    if sys.argv[1] == 'svm':
        model = svm_classifier(data_process=data_process)


    if sys.argv[2] == 'simple':
        model.training()
        model.evaluate(visualisation=sys.argv[4])

    if sys.argv[2] == 'cross_validation':
        model.pipeline(model.param_grid, k_fold=5, score_metrics=sys.argv[3], evaluation=sys.argv[4])



if __name__ == "__main__":
    main()
