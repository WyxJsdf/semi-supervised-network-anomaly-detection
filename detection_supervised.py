import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from supervised.nn import SimpleNN
from supervised.nn_torch import ModelNNTorch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import tree, svm
# from sklearn.externals import joblib

FILTER_CLASS_NAME='DoS Hulk'

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('inputpath', type=str, help='Path of the feature matrix to load')
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

def scale_data(data, scalar=None):
    # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
    if (scalar == None):
        scalar = MinMaxScaler().fit(data)
    data = scalar.transform(data)
    return data, scalar

def clear_specific_class(data, class_name):
    index = data[:, -1] != class_name
    return (data[index])

def shuffle_data(data):
    data = shuffle(data, random_state=1312)
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
    accuracy = accuracy_score(label, predicted_label)
    return precision, recall, f1, accuracy

def get_label_n(predicted_score, contam):
    threshold = percentile(y_pred, 100 * (1 - contam))
    predicted_label = (predicted_score > threshold).astype('int')
    return predicted_label

def exec_simplenn_keras(train_labeled_feature, train_label, train_raw_label,
                     test_feature, test_label, test_raw_label, contam):
    simpleNN = SimpleNN(train_labeled_feature.shape[-1])
    simpleNN.train_model(train_labeled_feature, train_label,
                            test_feature, test_label)
    predicted_label = simpleNN.evaluate_model(test_feature)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_simplenn_torch(train_labeled_feature, train_label, train_raw_label,
                     test_feature, test_label, test_raw_label, contam):
    simpleNN = ModelNNTorch(train_labeled_feature.shape[-1])
    simpleNN.train_model(train_labeled_feature, train_label,
                            test_feature, test_label, epoch=20)
    predicted_label, predicted_score = simpleNN.evaluate_model(test_feature, test_label)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))
    roc=roc_auc_score(test_label, predicted_score)
    print("roc= %.6lf" %(roc))

def exec_svm(train_labeled_feature, train_label, train_raw_label,
                     test_feature, test_label, test_raw_label, contam):
    clf = svm.LinearSVC(random_state=1314)
    clf.fit(train_labeled_feature,train_label)
    # joblib.dump(clf, 'IDS_classifier.joblib')
    predicted_label = clf.predict(test_feature)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)
    train_data = clear_specific_class(train_data, FILTER_CLASS_NAME)

    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, 0.01, False)
    train_labeled_feature, train_label, train_raw_label = split_dataset_vertical(train_labeled_data)
    train_unlabeled_feature, _, _ = split_dataset_vertical(train_unlabeled_data)
    test_feature, test_label, test_raw_label = split_dataset_vertical(test_data)
    train_unlabeled_feature, scalar = scale_data(train_unlabeled_feature)
    train_labeled_feature, _ = scale_data(train_labeled_feature, scalar)
    test_feature, _ = scale_data(test_feature, scalar)
    print("Preprocessing Data done......")
    # exec_svm(train_labeled_feature, train_label, train_raw_label,
    #                  test_feature, test_label, test_raw_label, args.contam)
    # exec_simplenn_keras(train_labeled_feature, train_label, train_raw_label,
    #                  test_feature, test_label, test_raw_label, args.contam)
    exec_simplenn_torch(train_labeled_feature, train_label, train_raw_label,
                     test_feature, test_label, test_raw_label, args.contam)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
