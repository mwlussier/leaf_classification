import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from src.data

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


    def __init__(self, model):
        self.model = model
        self.x_train = pd.read_csv('data/processed/train_processed.csv')
        self.y_train = pd.read_csv('data/processed/train_processed.csv')


    def training(self, x_train, t_train):
        self.model.fit(x_train, t_train)

    def prediction(self, x_train):
        return self.model.predict(x_train)

    def evaluate(self):
        pass

    def pipeline(self):
        pass