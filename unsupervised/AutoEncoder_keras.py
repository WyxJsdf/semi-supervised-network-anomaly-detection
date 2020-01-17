import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

class AutoEncoderKeras(object):

    def _build_model(self):
        input_shape=(self._num_features,)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dense(self._num_features, activation='sigmoid'))
        return model

    def deep_autoencoder(self):
        input_shape=(self._num_features,)
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=input_shape))
        # model.add(Dropout(0.1))
        model.add(Dense(80, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(10))
        # model.add(Dropout(0.1))
        model.add(Dense(80, activation='relu'))
        # model.add(Dropout(0.1))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self._num_features, activation='tanh'))
        return model

    def __init__(self, num_features, hidden_units=64, optimizer='adam', loss='mean_squared_error'):
        self._num_features = num_features
        self._model = self.deep_autoencoder()
        self._optimizer = optimizer
        self._loss = loss


    def train_model(self, X, epochs=10, batch_size=64):
        self._model.compile(optimizer=self._optimizer, loss=self._loss)
        hist = self._model.fit(
            x=X,
            y=X,
            epochs=epochs,
            batch_size=batch_size,
        )
        print(hist.history)

    def invert_order(self, scores):
        return (-score.ravel())

    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, X):
        predicted_score = self._model.predict(X)
        return self.get_distance(X, predicted_score)
