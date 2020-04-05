import sys
import argparse
import numpy as np
import pandas as pd
import csv
from sklearn.utils import shuffle

from semi_supervised.classifier_estimate import ConfidenceAutoEncoder


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from keras.utils import to_categorical

FILTER_CLASS_NAME=['', 'DoS Hulk', 'DDoS', 'PortScan']
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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
    num_train = int(dataset.shape[0] * rate)
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

def scale_data(data, scalar=None, type=0):
    # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    if (scalar == None):
        if type == 0:
            scalar = MinMaxScaler().fit(data)
        else:
            scalar = StandardScaler().fit(data)
    data = scalar.transform(data)
    return data, scalar


def shuffle_data(data, seed=1310):
    data = shuffle(data, random_state=seed)
    return data

def read_data(path):
    dataset = pd.read_csv(path,low_memory=False)
    return dataset.values


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
    threshold = percentile(y_pred, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label

def output_score(scores, name):
    f = open(name,"w",newline='')
    writer=csv.writer(f,dialect='excel')
    for i in range(len(scores[0])):
        p = []
        for h in scores:
            p.append(h[i])
        writer.writerow(p)
    f.close()

def print_confidence_each_class(predicted_score, raw_label):
    sum_classes = {}
    num_classes = {}
    # h = []
    for i in range(len(predicted_score)):
        if raw_label[i] not in sum_classes:
            sum_classes[raw_label[i]] = 0
            num_classes[raw_label[i]] = 0
        # if raw_label[i] == FILTER_CLASS_NAME:
            # h.append(predicted_score[i])
            # print("class:%s confidence:%.6lf classify:%.6lf" %(raw_label[i], confidence[i], predicted_score[i]))
        sum_classes[raw_label[i]] += predicted_score[i]
        num_classes[raw_label[i]] += 1
    # print(h)
    for p in sum_classes:
        print("confidence of %s: %.6lf" %(p, sum_classes[p] / num_classes[p]))

def exec_autoencoder_estimator(train_labeled_data, train_unlabeled_feature, validate_data, test_data, epoch, batch_size,
                                theta, contam, device, save_name):
    autoencoder = ConfidenceAutoEncoder(train_labeled_data[0].shape[-1], device, theta, save_name)
    autoencoder.train_model(train_labeled_data, train_unlabeled_feature, validate_data, epoch=epoch, batch_size=batch_size)
    predicted_label, predicted_score, classify_score, confidence= autoencoder.evaluate_model(test_data)
    roc=roc_auc_score(test_data[1], classify_score)
    print("roc= %.6lf" %(roc))
    print_confidence_each_class(classify_score, test_data[2])
    # print()
    print_confidence_each_class(confidence, test_data[2])
    # predicted_score_std = StandardScaler().fit_transform(predicted_score.reshape(-1, 1)).reshape(-1)
    predicted_score_std = MinMaxScaler().fit_transform(predicted_score.reshape(-1, 1)).reshape(-1)
    # classify_score = StandardScaler().fit_transform(classify_score.reshape(-1, 1)).reshape(-1)
    predicted_score_new = predicted_score_std + 2*confidence*(classify_score * classify_score-0.05)
    # output_score((classify_score, confidence, predicted_score, predicted_score_std, predicted_score_new)
    #              , test_data[2], 'score_estimator.csv')
    roc=roc_auc_score(test_data[1], predicted_score_new)
    print("roc_new= %.6lf" %(roc))
    precision, recall, f1_score, accuracy = eval_data(test_data[1], predicted_label, test_data[2])
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))


def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset, seed=args.seed)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)
    # validate_data, test_data = split_dataset_horizontal(test_data, 0.5, True)
    validate_data=test_data

    
    train_data = clear_specific_class(train_data, FILTER_CLASS_NAME[args.filter_class])
    # validate_data = clear_specific_class(validate_data, FILTER_CLASS_NAME)


    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, args.ratio_label, False)

    train_labeled_feature, train_label, train_raw_label = split_dataset_vertical(train_labeled_data)
    train_unlabeled_feature, _, _ = split_dataset_vertical(train_unlabeled_data)

    validate_feature, validate_label, validate_raw_label = split_dataset_vertical(validate_data)
    test_feature, test_label, test_raw_label = split_dataset_vertical(test_data)


    train_unlabeled_feature, scalar = scale_data(train_unlabeled_feature, type=0)
    train_labeled_feature, _ = scale_data(train_labeled_feature, scalar)
    validate_feature, _ = scale_data(validate_feature, scalar)
    test_feature, _ = scale_data(test_feature, scalar)

    print("Preprocessing Data done......")

    train_labeled_data = (train_labeled_feature, train_label, train_raw_label)
    validate_data = (validate_feature, validate_label, validate_raw_label)
    test_data = (test_feature, test_label, test_raw_label)
    save_name = "{}_{:.2f}_{}".format("CUE_SSAD", args.ratio_label, FILTER_CLASS_NAME[args.filter_class])
    save_name = args.output_path + '/' + save_name
    exec_autoencoder_estimator(train_labeled_data, train_unlabeled_feature,
                           validate_data, test_data, args.epoch, args.batch_size, args.theta,
                           args.contam, args.device, save_name)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
