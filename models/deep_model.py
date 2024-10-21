from .model import Model
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.optimizers import Adam
from keras import Input


class DeepModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential()

    def build_model(self, input_shape):
        self.model.add(Input(shape=input_shape))  # Explicit input layer
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_label, test_data, test_label):
        input_shape = (train_data.shape[1], 1)  # Ensure input shape has three dimensions
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
        test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
        self.build_model(input_shape)
        
        self.model.fit(train_data, train_label, epochs=32, batch_size=32, validation_split=0.2)

        inception_acc = self.model.evaluate(test_data, test_label)
        labels_pred_deep = (self.model.predict(test_data) > 0.5).astype("int32")
        print('Deep Model Accuracy:', inception_acc)
        return self.model, labels_pred_deep