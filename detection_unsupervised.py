import sys
import argparse
import numpy as np
import pandas as pd

from unsupervised.IsolateForest import IsolationForest
from unsupervised.LocalOutlierFactor import LocalOutlierFactor
from unsupervised.OneClassSVM import OneClassSVM
from unsupervised.AutoEncoder import AutoEncoder

from numpy import percentile
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    parser.add_argument('--contam', type=float,
        help='the percent of the outiers.', default=0.2)
    return parser.parse_args(argv)

def split_dataset(dataset):
    data = dataset[:, :-2]
    label = np.array(dataset[:, -2], dtype='int')
    raw_label = np.array(dataset[:, -1])
    return data, label, raw_label

def scale_data(data):
    data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    return data
    
def shuffle_data(data):
    data = shuffle(data)
    return data

def read_data(path):
    dataset = pd.read_csv(path,low_memory=False)
    return dataset.values
    # dataset = np.genfromtxt(path, dtype=None, delimiter=',', names=True)
    # return dataset

def eval_data(label, predicted_label, raw_label):
    ifR = pd.crosstab(label, predicted_label)
    ifR = pd.DataFrame(ifR)
    print(ifR)

    rawifR = pd.crosstab(raw_label, predicted_label)
    print(pd.DataFrame(rawifR))

    f1 = f1_score(label, predicted_label, average='binary', pos_label=1)
    precision = precision_score(label, predicted_label, average='binary', pos_label=1)
    recall = recall_score(label, predicted_label, average='binary', pos_label=1)
    return precision, recall, f1

def get_label_n(predicted_score, contam):
    threshold = percentile(y_pred, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label

def exec_isolate_forest(data, label, raw_label):
    isolate_forest = IsolationForest(contamination=contam, max_samples='auto', n_jobs=2)
    isolate_forest.train_model(data)
    predicted_label = isolate_forest.evaluate_model(data)
    precision, recall, f1_score = eval_data(label, predicted_label, raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def exec_one_class_svm(data, label, raw_label, contam):
    one_class_svm = OneClassSVM()
    one_class_svm.train_model(data)
    predicted_label = one_class_svm.evaluate_model(data)
    precision, recall, f1_score = eval_data(label, predicted_label, raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def exec_local_outlier_factor(data, label, raw_label, contam):
    local_outlier_factor = LocalOutlierFactor(contamination=contam, n_jobs=2)
    local_outlier_factor.train_model(data)
    predicted_label = local_outlier_factor.evaluate_model(data)
    precision, recall, f1_score = eval_data(label, predicted_label, raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def exec_autoencoder(data, label, raw_label, contam):
    autoencoder = AutoEncoder(data.shape[-1])
    autoencoder.train_model(data)
    predicted_score = autoencoder.evaluate_model(data)
    roc=roc_auc_score(label, predicted_score)
    print("roc= %.6lf" %(roc))
    predicted_label = get_label_n(predicted_score, contam)
    precision, recall, f1_score = eval_data(label, predicted_label, raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle(dataset)
    data, label, raw_label = split_dataset(dataset)
    data = scale_data(data)
    print("Preprocessing Data done......")

    # exec_isolate_forest(data, label, raw_label, args.contam)
    # exec_local_outlier_factor(data, label, raw_label, args.contam)
    # exec_one_class_svm(data, label, raw_label, args.contam)
    exec_autoencoder(data, label, raw_label, args.contam)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
