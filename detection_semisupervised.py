import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from semi_supervised.AutoEncoder import AutoEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    parser.add_argument('--contam', type=float,
        help='the percent of the outiers.', default=0.2)
    return parser.parse_args(argv)

def split_dataset_horizontal(dataset, rate=0.2):
    num_train = int(dataset.shape[0] * rate)
    return dataset[:num_train], dataset[num_train:]

def split_dataset_vertical(dataset):
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

def exec_autoencoder(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, contam):
    autoencoder = AutoEncoder(train_labeled_data.shape[-1])
    autoencoder.train_model(train_labeled_feature, train_unlabeled_feature, train_label)
    predicted_label, predicted_score = autoencoder.evaluate_model(test_feature)
    # roc=roc_auc_score(test_label, predicted_score)
    # print("roc= %.6lf" %(roc))
    # predicted_label = get_label_n(predicted_score, contam)
    precision, recall, f1_score = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle(dataset)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6)
    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(dataset, 0.2)
    train_labeled_feature, train_label, train_raw_label = split_dataset_vertical(train_labeled_data)
    train_unlabeled_feature, _, _ = split_dataset_vertical(train_unlabeled_data)
    test_feature, test_label, test_raw_label = split_dataset_vertical(test_data)
    data = scale_data(data)
    print("Preprocessing Data done......")
    exec_autoencoder(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, args.contam)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
