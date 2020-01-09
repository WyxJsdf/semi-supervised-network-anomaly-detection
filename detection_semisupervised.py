import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from unsupervised.IsolateForest import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    # parser.add_argument('--epoch', type=int,
    #     help='number of images in this folder.', default=300)
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

def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle(dataset)
    data, label, raw_label = split_dataset(dataset)
    data = scale_data(data)
    print("Preprocessing Data done......")

    isolate_forest = IsolationForest(contamination=0.2, max_samples='auto', n_jobs=2)
    isolate_forest.train_model(data)
    predicted_label = isolate_forest.evaluate_model(data)
    precision, recall, f1_score = eval_data(label, predicted_label, raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))
    

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
