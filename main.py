# import os
import sys
# sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import numpy as np
import models


def main():
    if sys.argv[1] == 'smv':
        model = models.svm_classifier.svm_classifier()

        if sys.argv[1] == 'pipeline':
            model.pipeline(model.param_grid)

    # model.training()
    # model.evaluate()

