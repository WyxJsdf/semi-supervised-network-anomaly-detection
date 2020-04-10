import sys
import argparse
import numpy as np
import pandas as pd
import csv

from numpy import percentile
from sequence_model.LSTM_classifier import LSTMClassifier
from sequence_model.LSTM_AutoEncoder import LSTMAutoEncoder
from sequence_model.LSTM_AutoEncoder_chain import LSTMAutoEncoderChain

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from unsupervised.AutoEncoder_torch import AutoEncoder

WINDOW_SIZE = 20
EMBEDDING_SIZE=97

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    parser.add_argument('--cuda', type=str,
        help='type of gpu device', default='cpu')
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

def clear_specific_class(data, class_name):
    index = data[:, -1] != class_name
    return (data[index])

def output_score(scores, raw_label, name):
    f = open(name,"w",newline='')
    writer=csv.writer(f,dialect='excel')
    for i in range(len(raw_label)):
        p = []
        for h in scores:
            p.append(h[i])
        p.append(raw_label[i])
        writer.writerow(p)
    f.close()

def scale_data(data, scalar=None):
    # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    if (scalar == None):
        scalar = MinMaxScaler(feature_range=(0, 1)).fit(data)
    data = scalar.transform(data)
    return data, scalar

def shuffle_data(data):
    data = shuffle(data, random_state=1310)
    return data

def transform(data_list):
    features = []
    labels = []
    seq_length = []
    for data in data_list:
        feature = []
        for i in range(min(len(data) - 2, WINDOW_SIZE)):
            dst_port, hlength, pllength, delta_ts, TCP_winsize = [int(ele) for ele in data[i].split('|')]
            # emb_src_port = big_unpackbits(src_port, 2)
            emb_dst_port = big_unpackbits(dst_port, 2)
            emb_hlength = big_unpackbits(hlength, 4)
            emb_pllength = big_unpackbits(pllength, 4)
            emb_TCP_winsize = big_unpackbits(TCP_winsize, 4)
            emb_delta_ts = np.array([delta_ts])
            # emb_dst_port = np.array([dst_port])
            # emb_hlength = np.array([hlength])
            emb_pllength = np.array([pllength])
            # emb_TCP_winsize = np.array([TCP_winsize])
            h = np.concatenate((emb_dst_port, emb_TCP_winsize,emb_hlength, emb_pllength, emb_delta_ts))
            feature.append(h)
        if len(data) - 2 < WINDOW_SIZE:
            for i in range(WINDOW_SIZE + 2 - len(data)):
                feature.append(np.array([0] * len(h)))
        features.append(np.stack(feature))
        labels.append(int(data[-2]))
        seq_length.append(min(len(data) - 2, WINDOW_SIZE))
    return (np.stack(features).astype(np.double).reshape(len(features), -1)
           , np.array(labels, dtype=int), np.array(seq_length, dtype=int))

def big_unpackbits(mynum, max_block=1):
    # return np.array([mynum])
    cutted_num = np.array([(mynum>>i*8)&0xff for i in range(max_block)], dtype=np.uint8)
    return np.unpackbits(cutted_num)

def read_data(path):
    with open (path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data_list = list(reader)
    return data_list
    

def eval_data(label, predicted_label):
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
    threshold = percentile(predicted_score, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label

def exec_lstm_classify(train_labeled_data, test_data, contam, cuda):
    lstmClassifier = LSTMClassifier(train_labeled_data[0].shape[2], cuda)
    lstmClassifier.train_model(train_labeled_data, test_data, epoch=3000)
    predicted_label, classify_score = lstmClassifier.evaluate_model(test_data)

    roc=roc_auc_score(test_data[1], classify_score)
    print("roc auc classify= %.6lf" %(roc))

    precision, recall, f1_score, accuracy = eval_data(test_data[1], predicted_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))


def exec_lstm_autoencoder(train_labeled_data, train_unlabeled_data, test_data, contam, cuda):
    print("now execute the model LSTM AutoEncoder by Pytorch!")
    lstmAutoencoder = LSTMAutoEncoder(train_unlabeled_data[0].shape[2], train_unlabeled_data[0].shape[1], cuda)
    lstmAutoencoder.train_model(train_labeled_data, train_unlabeled_data, test_data)
    predicted_label, predicted_score, classify_score = lstmAutoencoder.evaluate_model(test_data)
    # predicted_label = get_label_n(predicted_score, contam)

    roc=roc_auc_score(test_data[1], predicted_score)
    print("roc auc= %.6lf" %(roc))

    roc=roc_auc_score(test_data[1], classify_score)
    print("roc auc classify= %.6lf" %(roc))

    output_score((classify_score, predicted_score), test_data[1], 'lstmAutoencoder.csv')
    precision, recall, f1_score, accuracy = eval_data(test_data[1], predicted_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_lstm_autoencoder_chain(train_labeled_data, train_unlabeled_data, test_data, contam, cuda):
    lstmAutoencoder = LSTMAutoEncoderChain(train_unlabeled_data[0].shape[2], train_unlabeled_data[0].shape[1], cuda)
    lstmAutoencoder.train_model(train_labeled_data, train_unlabeled_data, test_data)
    predicted_label, predicted_score, classify_score = lstmAutoencoder.evaluate_model(test_data)
    # predicted_label = get_label_n(predicted_score, contam)

    roc=roc_auc_score(test_data[1], predicted_score)
    print("roc auc= %.6lf" %(roc))

    roc=roc_auc_score(test_data[1], classify_score)
    print("roc auc classify= %.6lf" %(roc))

    precision, recall, f1_score, accuracy = eval_data(test_data[1], predicted_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_autoencoder(train_feature, test_feature, test_label, contam):
    print("now execute the model AutoEncoder by Pytorch!")
    autoencoder = AutoEncoder(train_feature.shape[-1])
    autoencoder.train_model(train_feature, test_feature, test_label)
    predicted_score = autoencoder.evaluate_model(test_feature)
    predicted_label = get_label_n(predicted_score, contam)

    roc=roc_auc_score(test_label, predicted_score)
    print("roc auc= %.6lf" %(roc))

    precision, recall, f1_score = eval_data(test_label, predicted_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf" %(precision, recall, f1_score))

def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)
    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, 0.01, False)
    train_labeled_feature, train_label, train_labeled_seqlen = transform(train_labeled_data)
    train_unlabeled_feature, _, train_unlabeled_seqlen = transform(train_unlabeled_data)
    test_feature, test_label, test_seqlen = transform(test_data)

    # p, scalar = scale_data(train_unlabeled_feature[:,:,-4:].reshape(-1, 4))
    # train_unlabeled_feature[:,:,-4:] = p.reshape(-1, WINDOW_SIZE, 4)
    # p, _ = scale_data(train_labeled_feature[:, :, -4:].reshape(-1, 4), scalar)
    # train_labeled_feature[:,:,-4:] = p.reshape(-1, WINDOW_SIZE, 4)
    # p, _ = scale_data(test_feature[:, :, -4:].reshape(-1, 4), scalar)
    # test_feature[:,:,-4:] = p.reshape(-1, WINDOW_SIZE, 4)

    train_unlabeled_feature, scalar = scale_data(train_unlabeled_feature)
    train_labeled_feature, _ = scale_data(train_labeled_feature, scalar)
    test_feature, _ = scale_data(test_feature, scalar)
    train_unlabeled_feature = train_unlabeled_feature.reshape(len(train_unlabeled_feature), WINDOW_SIZE, -1)
    train_labeled_feature = train_labeled_feature.reshape(len(train_labeled_feature), WINDOW_SIZE, -1)
    test_feature = test_feature.reshape(len(test_feature), WINDOW_SIZE, -1)
    print("Preprocessing Data done......")
    exec_lstm_classify((train_labeled_feature, train_label, train_labeled_seqlen), (test_feature, test_label, test_seqlen), args.contam, args.cuda)
    # exec_lstm_autoencoder((train_labeled_feature, train_label, train_labeled_seqlen),
    #                       (train_unlabeled_feature, train_unlabeled_seqlen),
    #                       (test_feature, test_label, test_seqlen), args.contam, args.cuda)
    # exec_lstm_autoencoder_chain((train_labeled_feature, train_label, train_labeled_seqlen),
    #                       (train_unlabeled_feature, train_unlabeled_seqlen),
    #                       (test_feature, test_label, test_seqlen), args.contam, args.cuda)

    # exec_autoencoder(train_unlabeled_feature, test_feature, test_label, args.contam)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
