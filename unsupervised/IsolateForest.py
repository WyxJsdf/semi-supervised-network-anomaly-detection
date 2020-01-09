import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as IsolationF

class IsolationForest(object):

    def __init__(self, contamination=0.25, max_samples="auto", n_jobs=1):
        self._contam = contamination
        self._max_samples = max_samples
        self._n_jobs = n_jobs

    def train_model(self, X):
        self._model = IsolationF(max_samples=self._max_samples, n_jobs=self._n_jobs,
            contamination=self._contam, behaviour='new').fit(X)

    def evaluate_model(self, X):
        predicted_label = self._model.predict(X)
        maxVal = {-1:1, 1:0}
        predicted_label[:] = [maxVal[item] for item in predicted_label[:]]
        return predicted_label
