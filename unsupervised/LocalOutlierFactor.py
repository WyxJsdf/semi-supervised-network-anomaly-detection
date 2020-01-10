import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as lof

class LocalOutlierFactor(object):

    def __init__(self, contamination=0.2, algorithm="auto", n_jobs=1):
        self._contam = contamination
        self._algorithm = algorithm
        self._n_jobs = n_jobs



    def train_model(self, X):
        self._label = lof(algorithm=self._algorithm, n_jobs=self._n_jobs,
            contamination=self._contam).fit_predict(X)

    def evaluate_model(self, X):
        predicted_label = self._label
        maxVal = {-1:1, 1:0}
        predicted_label[:] = [maxVal[item] for item in predicted_label[:]]
        return predicted_label
