import sys
import argparse
import numpy as np
import pandas as pd
import csv

from sequence_model.LSTM_classifier import LSTMClassifier

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

WINDOW_SIZE = 20
EMBEDDING_SIZE=97

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    parser.add_argument('--contam', type=float,
        help='the percent of the outiers.', default=0.2)
    return parser.parse_args(argv)

def split_dataset_horizontal(dataset, rate=0.2, is_split=True):
    num_train = int(len(dataset) * rate)
    if is_split:
        return dataset[:num_train], dataset[num_train:]
    else:
        return np.copy(dataset[:num_train]), dataset

def split_dataset_vertical(dataset):
    data = dataset[:, :-2]
    label = np.array(dataset[:, -2], dtype='int')
    raw_label = np.array(dataset[:, -1])
    return data, label, raw_label

def scale_data(data, scalar=None):
    # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    if (scalar == None):
        scalar = MinMaxScaler().fit(data)
    data = scalar.transform(data)
    return data, scalar

def shuffle_data(data):
    data = shuffle(data, random_state=1310)
    return data

def transform(data_list):
    features = []
    labels = []
    for data in data_list:
        feature = []
        for i in range(min(len(data) - 1, 20)):
            src_port, dst_port, pllength, delta_ts, TCP_winsize = [int(ele) for ele in data[i].split('|')]
            emb_src_port = big_unpackbits(src_port, 2)
            emb_dst_port = big_unpackbits(dst_port, 2)
            emb_pllength = big_unpackbits(pllength, 4)
            emb_delta_ts = np.array([delta_ts])
            emb_TCP_winsize = big_unpackbits(TCP_winsize, 4)
            h = np.concatenate((emb_src_port, emb_dst_port, emb_pllength, emb_delta_ts, emb_TCP_winsize))
            feature.append(h)
        if len(data) <= WINDOW_SIZE:
            for i in range(WINDOW_SIZE + 1 - len(data)):
                feature.append(np.array([0] * 97))
        features.append(np.stack(feature))
        labels.append(int(data[-1]))
    return np.stack(features).astype(np.double), np.array(labels, dtype=int)

def big_unpackbits(mynum, max_block=1):
    cutted_num = np.array([(mynum>>i*8)&0xff for i in range(max_block)], dtype=np.uint8)
    return np.unpackbits(cutted_num)

def read_data(path):
    with open (path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_list = list(reader)
    return data_list
    

def eval_data(label, predicted_label, raw_label):
    ifR = pd.crosstab(label, predicted_label)
    ifR = pd.DataFrame(ifR)
    print(ifR)

    # rawifR = pd.crosstab(raw_label, predicted_label)
    # print(pd.DataFrame(rawifR))

    f1 = f1_score(label, predicted_label, average='binary', pos_label=1)
    precision = precision_score(label, predicted_label, average='binary', pos_label=1)
    recall = recall_score(label, predicted_label, average='binary', pos_label=1)
    accuracy = accuracy_score(label, predicted_label)
    return precision, recall, f1, accuracy

def get_label_n(predicted_score, contam):
    threshold = percentile(y_pred, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label
def exec_lstm_classify(train_labeled_feature, train_label, test_feature, test_label, contam):
    lstmClassifier = LSTMClassifier(train_labeled_feature.shape[2])
    lstmClassifier.train_model(train_labeled_feature, train_label, test_feature, test_label)
    predicted_label = lstmClassifier.evaluate_model(test_feature, test_label)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, [])
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))



def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)
    train_labeled_feature, train_label = transform(train_data)
    test_feature, test_label = transform(test_data)


    # train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, 0.01, False)
    # train_labeled_feature, train_label, train_raw_label = split_dataset_vertical(train_labeled_data)
    # train_unlabeled_feature, _, _ = split_dataset_vertical(train_unlabeled_data)
    # test_feature, test_label, test_raw_label = split_dataset_vertical(test_data)
    # train_labeled_feature, scalar = scale_data(train_labeled_feature)
    # train_unlabeled_feature, _ = scale_data(train_unlabeled_feature, scalar)
    # test_feature, _ = scale_data(test_feature, scalar)
    print("Preprocessing Data done......")
    exec_lstm_classify(train_labeled_feature, train_label, test_feature, test_label, args.contam)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
