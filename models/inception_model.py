from .model import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.applications import InceptionV3


class InceptionModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    def build_model(self, input_shape):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_label, test_data, test_label):
        input_shape = (224, 224, 3)
        self.build_model(input_shape)
        self.model.fit(train_data, train_label, epochs=32, batch_size=32, validation_split=0.2)

        inception_acc = self.model.evaluate(test_data, test_label)
        labels_pred_deep = (self.model.predict(test_data) > 0.5).astype("int32")
        print('InceptionV3 Accuracy:', inception_acc)
        return self.model, labels_pred_deep

    def save(self, file_name: str):
        self.model.save(file_name)
    