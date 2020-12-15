import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
from pickle import dump, load
from src.data.util_dataset import to_train_dataset, to_submit


class VanillaClassifier:
    """
    Vanilla Classifier
    ==================
        Base class being inherited in every child classifiers.

    Attributes
    ==========
        model            - sklearn model used for training and predictions
        data_processing  - Type of processed data to use in the training est testing process
        x_train, y_train - training variables and labels
        x_test, y_test   - testing variables and labels
    """

    def __init__(self, model, data_process=None):
        self.model = model
        self.data_process = data_process
        self.X_train, self.y_train, self.X_test, self.y_test, self.label_map = to_train_dataset(data_process)
        self.X_submission = to_submit(data_process)

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self, X):
        return self.model.predict(X)

    def prediction_probabilities_submission(self):
        return pd.DataFrame(self.model.predict_proba(self.X_submission),
                            index=self.X_submission.index, columns=self.label_map)

    def evaluate(self, evaluation='report', visualisation=False):
        """
        Evaluate the model reporting different metrics including:
            accuracy
            log_loss
            confusion matrix
            classification report
        """
        accuracy_train = self.model.score(self.X_train, self.y_train)
        accuracy_test = self.model.score(self.X_test, self.y_test)
        err_train = 1 - accuracy_train
        err_test = 1 - accuracy_test

        print('Training error: ', str(round(err_train * 100, 3)), '%')
        print('Testing error: ', str(round(err_test * 100, 3)), '%')
        error_flag = error_analysis(err_train, err_test)
        train_log_loss = 9999
        test_log_loss = 9999
        try:
            probability_train = self.model.predict_proba(self.X_train)
            probability = self.model.predict_proba(self.X_test)
            train_log_loss = log_loss(self.y_train, probability_train)
            test_log_loss = log_loss(self.y_test, probability)
        except:
            probability=None

        if error_flag:
            prediction = self.model.predict(self.X_test)
            metrics(self.y_test, self.label_map, prediction=prediction,
                    evaluation=evaluation, probability=probability)
        return accuracy_train, accuracy_test, train_log_loss, test_log_loss

    def pipeline(self, param_grid, k_fold=5, score_metrics='accuracy', evaluation='report'):
        """
        Stratified KFold (default=5): Generate test sets such that all contain the
            same distribution of classes, or as close as possible.

        GridSearchCV: Execute cross-validation using a parameter grid included by default in each model file.
            scoring = 'accuracy' | 'roc_auc_ovr'
        """
        k_fold_cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring=score_metrics,
                                   cv=k_fold_cv,
                                   verbose=1,
                                   n_jobs=-1)

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_
        self.training()
        accuracy_train, accuracy_test, train_loss, test_loss = self.evaluate(evaluation=evaluation)
        print("Best Score: ", grid_search.best_score_)
        print("Best Parameters: ", grid_search.best_params_)
        if self.data_process is not None:
            self.save_model()
        return grid_search.best_score_, grid_search.best_params_, accuracy_train, accuracy_test, train_loss, test_loss

    def save_model(self, filename=""):
        if filename == "":
            if self.model.__str__()[:6] == 'Voting':
                filename = "Voting_best_estimator_" + self.data_process
            else:
                filename = str(self.model.__class__())[:-2] + "_best_estimator_" + self.data_process
        dump(self.model, open('models/' + filename + '.pkl', 'wb'))


def metrics(y, label_map, prediction, evaluation='report', probability=None):
    """
        Evaluate Classification Report, AUC (default), and Confusion Matrix.
    """
    if evaluation == 'confusion_matrix':
        print(f"CONFUSION MATRIX:\n================================================"
              f"\n{confusion_matrix(y, prediction)}")
        print("================================================")

    print("Metrics Result:\n================================================")
    print(f"CLASSIFICATION REPORT:\n{classification_report(y, prediction, target_names=label_map.values)}")
    if probability is not None:
        auc = roc_auc_score(y, probability, multi_class='ovr')
        print('ROC AUC: {:.2f}'.format(auc))
        print("================================================")


def error_analysis(err_train, err_test, threshold=0.2):
    """
        Using a threshold to flag when there is over/under-fitting.
        Default: 0.20
    """
    if (err_test - err_train) > threshold:
        print("WARNING: OVER-fitting")
        return False
    if err_train > threshold:
        print("WARNING: UNDER-fitting")
        return False
    return True
