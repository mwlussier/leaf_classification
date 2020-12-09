# import os
import sys
# sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import numpy as np
import models


def main():

    model = models.svm_classifier.svm_classifier()
    model.training()
    model.evaluate()

    if sys.argv[1] == 'smv':
        model = models.svm_classifier.svm_classifier()
        if sys.argv[2] == 'pipeline':
            model.pipeline(model.param_grid)

    if sys.argv[1] == 'logit':
        model = models.logistic_regression_classifier.logistic_regression()
        if sys.argv[2] == 'pipeline':
            model.pipeline(model.param_grid)

    if sys.argv[1] == 'random_forest':
        model = models.random_forest_classifier.random_forest()
        if sys.argv[2] == 'pipeline':
            model.pipeline(model.param_grid)


if __name__ == "__main__":
    main()
