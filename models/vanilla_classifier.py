import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from src.data.util_dataset import to_train_dataset
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pickle import dump, load


class VanillaClassifier:
    """
    Vanilla Classifier
    ==================
        Base class being inherited in every child classifiers.

    Attributes
    ==========
        model   - sklearn model used for training and predictions
        x_train - training variables
        y_train - training labels

    """

    def __init__(self, model, data_process=False):
        self.model = model
        self.data_process = data_process
        self.X_train, self.y_train, self.X_test, self.y_test = to_train_dataset(data_process)

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def prediction(self, X):
        return self.model.predict(X)

    def evaluate(self, evaluation='simple', visualisation=False):
        accuracy_train = self.model.score(self.X_train, self.y_train)
        accuracy_test = self.model.score(self.X_test, self.y_test)
        err_train = 1 - accuracy_train
        err_test = 1 - accuracy_test

        print('Training error: ', str(round(err_train * 100, 3)), '%')
        print('Testing error: ', str(round(err_test * 100, 3)), '%')
        error_flag = error_analysis(err_train, err_test)
        if error_flag:
            prediction = self.model.predict(self.X_test)
            #probability = self.model.predict_proba(self.X_test)
            metrics(self.y_test, prediction=prediction, evaluation=evaluation, probability=None)




    def pipeline(self, param_grid, k_fold=5, score_metrics='accuracy', evaluation='simple'):
        """
        Stratified KFold (default=5): Generate test sets such that all contain the
            same distribution of classes, or as close as possible.
        Grid Search Cross-Validation
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        """
        k_fold_cv = StratifiedKFold(n_splits=k_fold, shuffle=True)
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring=score_metrics,
                                   refit=True,
                                   cv=k_fold_cv,
                                   verbose=1,
                                   n_jobs=-1)

        grid_search.fit(self.X_train, self.y_train)
        self.model = grid_search.best_estimator_
        self.training()
        self.evaluate(evaluation=evaluation)
        print("Best Score: ", grid_search.best_score_)
        print("Best Parameters: ", grid_search.best_params_)
        if self.data_process:
            self.save_model()

    def save_model(self, filename=""):
        if filename == "":
            filename = str(self.model.__class__())[:-2] + "_best_estimator"
        dump(self.model, open('models/best_estimators/' + filename + '.pkl', 'wb'))


def metrics(y, prediction, evaluation='simple', probability=None):
    """
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

    """
    if evaluation == 'confusion_matrix':
        confusion = confusion_matrix(y, prediction)
        print('Confusion Matrix\n')
        print(confusion)
    if evaluation == 'report':
        print('\nClassification Report\n')
        print(classification_report(y, prediction))
    if evaluation == 'roc_auc':
        auc = roc_auc_score(y, probability)
        print('\nROC AUC: {:.2f}\n'.format(auc))

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y, prediction)))
    print('Micro Precision: {:.2f}'.format(precision_score(y, prediction, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y, prediction, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y, prediction, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y, prediction, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y, prediction, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y, prediction, average='macro')))


def error_analysis(err_train, err_test, threshold=0.2):
    """
        Using a threshold to flag when there is over/under-fitting.
        Default: 0.20
    """
    if (err_test - err_train) > threshold:
        print("WARNING: OVER-fitting")
        return True
    if (err_train > threshold):
        print("WARNING: UNDER-fitting")
        return True
    return False
