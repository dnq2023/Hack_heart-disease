from .model import Model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class SVMModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC()

    def train(self, train_data, train_label, test_data, test_label):
        self.model.fit(train_data, train_label)
        predict = self.model.predict(test_data)
        print('SVM Accuracy:', accuracy_score(test_label, predict))
        return self.model, predict