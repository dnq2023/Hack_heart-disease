from .model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class CLSModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100)


    def train(self, train_data, train_label, test_data, test_label):
        self.model.fit(train_data, train_label)
        predict = self.model.predict(test_data)
        print('Random Forest Accuracy:', accuracy_score(test_label, predict))
        return self.model, predict
