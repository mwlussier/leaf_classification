import os
import sys
from src.models.bagging_classifier import Bagging
from src.models.decision_tree_classifier import DecisionTree
from src.models.fully_connected_classifier import FullyConnected
from src.models.gradient_boosting_classifier import GradientBoosting
from src.models.logistic_regression_classifier import Logit
from src.models.random_forest_classifier import RandomForest
from src.models.svm_classifier import SvmClassifier
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))


def main():
    """
    python main.py <model> <pipeline> <cv_metrics> <evaluate> <data_processing>
    $ python main.py svm simple accuracy report fselection

    model : bagging | decision_tree | fconnected | gboost | logit | random_forest | svm
    pipeline: simple | cross_validation
    cv_metrics: accuracy | roc_auc_ovr
    evaluate: <report> | confusion_matrix
    data_processing: <Empty> | simple | fselection | pca_50 | pca_100 | pca_150
    """

    try:
        data_process = sys.argv[5]
    except:
        data_process = None

    #  ~~~ Bagging ~~~
    if sys.argv[1] == 'bagging':
        model = Bagging(data_process=data_process)
    #  ~~~ Decision Tree ~~~
    if sys.argv[1] == 'decision_tree':
        model = DecisionTree(data_process=data_process)
    #  ~~~ Fully Connected ~~~
    if sys.argv[1] == 'fconnected':
        model = FullyConnected(data_process=data_process)
    #  ~~~ GBoost ~~~
    if sys.argv[1] == 'gboost':
        model = GradientBoosting(data_process=data_process)
    #  ~~~ LOGIT ~~~
    if sys.argv[1] == 'logit':
        model = Logit(data_process=data_process)
    #  ~~~ Random Forest ~~~
    if sys.argv[1] == 'random_forest':
        model = RandomForest(data_process=data_process)
    #  ~~~ SVM ~~~
    if sys.argv[1] == 'svm':
        model = SvmClassifier(data_process=data_process)

    if sys.argv[2] == 'simple':
        model.training()
        model.evaluate(visualisation=sys.argv[4])

    if sys.argv[2] == 'cross_validation':
        model.pipeline(model.param_grid, k_fold=5, score_metrics=sys.argv[3], evaluation=sys.argv[4])


if __name__ == "__main__":
    main()
