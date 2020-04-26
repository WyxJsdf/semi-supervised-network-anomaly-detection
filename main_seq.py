import sys
import argparse
import numpy as np
import pandas as pd
import csv
import json

from numpy import percentile
from sequence_model.estimate_gru_ae import LSTMAutoEncoder

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from unsupervised.AutoEncoder_torch import AutoEncoder

WINDOW_SIZE = 20
EMBEDDING_SIZE=97

FILTER_CLASS_NAME=['', 'DoS Hulk', 'DDoS', 'PortScan', 'DoS GoldenEye', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'SSH-Patator']

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
    parser.add_argument('output_path', type=str, help='Path of the result to save')
    parser.add_argument('--filter_class', type=int,
        help='id of filter class', default=0)
    parser.add_argument('--epoch', type=int,
        help='number of the training epochs', default=10)
    parser.add_argument('--batch_size', type=int,
        help='number of the batch_size', default=128)
    parser.add_argument('--theta', type=int,
        help='theta for confidence estimate', default=50)
    parser.add_argument('--seed', type=int,
        help='random seed', default=1310)
    parser.add_argument('--ratio_label', type=float,
        help='ratio of labeled training data', default=0.01)
    parser.add_argument('--device', type=str,
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

def clear_specific_class(data, class_name):
    data = list(filter(lambda p: p[-1] != class_name, data))
    return data

def output_score(scores, name):
    f = open(name,"w",newline='')
    writer=csv.writer(f,dialect='excel')
    for i in range(len(scores[0])):
        p = []
        for h in scores:
            p.append(h[i])
        writer.writerow(p)
    f.close()

def scale_data(data, scalar=None):
    # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    if (scalar == None):
        scalar = MinMaxScaler(feature_range=(0, 1)).fit(data)
    data = scalar.transform(data)
    return data, scalar

def shuffle_data(data, seed=1310):
    data = shuffle(data, random_state=seed)
    return data

def transform(data_list):
    features = []
    labels = []
    raw_labels = []
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
        raw_labels.append(data[-1])
        seq_length.append(min(len(data) - 2, WINDOW_SIZE))
    return (np.stack(features).astype(np.double).reshape(len(features), -1),
            np.array(labels, dtype=int), np.array(seq_length, dtype=int),
            np.array(raw_labels))

def big_unpackbits(mynum, max_block=1):
    # return np.array([mynum])
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

    rawifR = pd.crosstab(raw_label, predicted_label)
    print(pd.DataFrame(rawifR))

    f1 = f1_score(label, predicted_label, average='binary', pos_label=1)
    precision = precision_score(label, predicted_label, average='binary', pos_label=1)
    recall = recall_score(label, predicted_label, average='binary', pos_label=1)
    accuracy = accuracy_score(label, predicted_label)
    return precision, recall, f1, accuracy

def get_label_n(predicted_score, contam):
    threshold = percentile(predicted_score, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label


def exec_lstm_autoencoder(train_labeled_data, train_unlabeled_data, test_data, epoch, save_name, device, batch_size, theta):
    print("now execute the model LSTM AutoEncoder by Pytorch!")
    lstmAutoencoder = LSTMAutoEncoder(train_unlabeled_data[0].shape[2], train_unlabeled_data[0].shape[1], device,save_name, theta=theta)
    lstmAutoencoder.train_model(train_labeled_data, train_unlabeled_data, test_data, epoch=epoch, batch_size=batch_size)
    predicted_label, predicted_score, classify_score, confidence_score = lstmAutoencoder.evaluate_model(test_data)
    # predicted_label = get_label_n(predicted_score, contam)

    roc=roc_auc_score(test_data[1], predicted_score)
    print("roc auc= %.6lf" %(roc))

    roc=roc_auc_score(test_data[1], classify_score)
    print("roc auc classify= %.6lf" %(roc))

    output_score((classify_score, predicted_score, test_data[1]), 'lstmAutoencoder.csv')
    precision, recall, f1_score, accuracy = eval_data(test_data[1], predicted_label, test_data[3])
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

    new_name = "{}_{:.4f}.json".format(save_name, roc)
    dicts = {}
    dicts['test_auc'] = roc
    dicts['f1_score'] = f1_score
    dicts['test_scores'] = list(predicted_score)
    dicts['conf_scores'] = list(confidence_score)
    dicts['test_label'] = list(test_data[1].astype(np.float))
    dicts['test_raw_label'] = list(test_data[3])
    with open(new_name,"w") as f:
        json.dump(dicts,f)
    f.close()
def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset, seed=args.seed)

    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)

    train_data = clear_specific_class(train_data, FILTER_CLASS_NAME[args.filter_class])

    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, args.ratio_label, False)
    train_labeled_feature, train_label, train_labeled_seqlen, train_raw_label = transform(train_labeled_data)
    train_unlabeled_feature, _, train_unlabeled_seqlen, _ = transform(train_unlabeled_data)
    test_feature, test_label, test_seqlen, test_raw_label = transform(test_data)

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

    save_name = "{}_{:.2f}_{}".format('ue-ssgru', args.ratio_label, FILTER_CLASS_NAME[args.filter_class])
    save_name = args.output_path + '/' + save_name

    train_labeled_data = (train_labeled_feature, train_label, train_labeled_seqlen)
    train_unlabeled_data = (train_unlabeled_feature, train_unlabeled_seqlen)
    test_data = (test_feature, test_label, test_seqlen, test_raw_label)
    exec_lstm_autoencoder(train_labeled_data, train_unlabeled_data, test_data, args.epoch, save_name, args.device, args.batch_size, args.theta)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
