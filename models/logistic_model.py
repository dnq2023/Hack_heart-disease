from .model import Model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class LogisticModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = LogisticRegression(max_iter=1000)

    def train(self, train_data, train_label, test_data, test_label):
        self.model.fit(train_data, train_label)
        predict = self.model.predict(test_data)
        print('Logistic Regression Accuracy:', accuracy_score(test_label, predict))
        return self.model, predict