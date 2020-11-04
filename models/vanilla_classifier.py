import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))

class VanillaClassifier:
    """


    """


    def __inti__(self, model):
        self.model = model

    def training(self, x_train, t_train):
        self.model.fit(x_train, t_train)

    def prediction(self, x_train):
        return self.model.predict(x_train)

    def evaluate(self):
        pass

    def pipeline(self):
        pass