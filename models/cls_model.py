from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_cls_model(train_data, train_label, test_data, test_label):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train_data, train_label)
    rf_pred = rf.predict(test_data)
    print('Random Forest Accuracy:', accuracy_score(test_label, rf_pred))
    return rf, rf_pred