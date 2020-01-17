import os
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras import Input, metrics
from keras.utils import plot_model, to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
class AutoEncoder(object):

    def _build_model(self):
        input_shape = (self._num_features,)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=input_shape))
        model.add(Dense(self._num_features, activation='sigmoid'))
        return model

    def deep_autoencoder(self):
        input_tensor = Input(shape=(self._num_features,))
        encoder = Dense(100, activation='relu')(input_tensor)
        encoder = Dense(80, activation='relu')(encoder)
        encoder_output = Dense(10, activation='relu')(encoder)


        model_encoder = Model(input_tensor, encoder_output)

        input_labeled = Input(shape=(self._num_features,), name='input_labeled')
        # input_unlabeled = Input(shape=(self._num_features,), name='input_unlabeled')
        model_labeled = model_encoder(input_labeled)
        model_unlabeled = model_encoder(input_labeled)

        decoder = Dense(80, activation='relu')(model_unlabeled)
        decoder = Dense(100, activation='relu')(decoder)
        decoder_output = Dense(self._num_features, activation='tanh', name='decoder_output')(decoder)

        classify_output = Dense(2, activation='softmax', name='classify_output')(model_labeled)
        model = Model([input_labeled], [classify_output, decoder_output])
        return model

    def __init__(self, num_features, hidden_units=64, optimizer='adam'):
        self._num_features = num_features
        self._model = self.deep_autoencoder()
        self._optimizer = optimizer
        plot_model(self._model, show_shapes=True)
        print(self._model.summary())
        self._model.compile(optimizer=self._optimizer, loss={'decoder_output': 'mean_squared_error', 
                            'classify_output': 'binary_crossentropy'}, loss_weights={'decoder_output': 1, 
                            'classify_output': 1}, metrics={'classify_output': ['accuracy', metrics.categorical_accuracy]})


    def train_model(self, feature_labeled, feature_unlabeled, label, test_feature, test_label, 
                    epochs=20, batch_size=128):
        feature_labeled = np.vstack((feature_labeled, feature_unlabeled))
        label = np.vstack((to_categorical(label), np.zeros((feature_unlabeled.shape[0], 2))))
        hist = self._model.fit(
            {'input_labeled': feature_labeled},
            {'decoder_output': feature_labeled,
            'classify_output': label},
            validation_data=(test_feature, {'classify_output': to_categorical(test_label),
                                            'decoder_output': test_feature}),
            epochs=epochs,
            batch_size=batch_size
        )
        print(hist.history)

    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, feature):
        predicted_class, predicted_score = self._model.predict(
            {'input_labeled': feature})
        return np.argmax(predicted_class, axis=1), self.get_distance(feature, predicted_score)
