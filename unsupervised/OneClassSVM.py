import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM as ocSVM

class OneClassSVM(object):

    def __init__(self, kernel='rbf', degree=3, gamma=0.1, nu=0.05, max_iter=-1):
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._nu = nu
        self.max_iter = max_iter

    def train_model(self, X):
        self._model = ocSVM(degree=self._degree, kernel=self._kernel,
            gamma=self._gamma, nu=self._nu, max_iter=self._max_iter).fit(X)

    def evaluate_model(self, X):
        predicted_label = self._model.predict(X)
        maxVal = {-1:1, 1:0}
        predicted_label[:] = [maxVal[item] for item in predicted_label[:]]
        return predicted_label
