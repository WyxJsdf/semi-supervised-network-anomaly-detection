import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from semi_supervised.AutoEncoder_keras import AutoEncoderKeras
from semi_supervised.AutoEncoder_torch import AutoEncoder
from semi_supervised.AutoEncoder_chain import AutoEncoderChain
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from semi_supervised.ladder_net import get_ladder_network_fc

from keras.utils import to_categorical


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

def shuffle_data(data):
    data = shuffle(data, random_state=1310)
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

def exec_autoencoder_keras(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, contam):
    autoencoder = AutoEncoderKeras(train_labeled_feature.shape[-1])
    autoencoder.train_model(train_labeled_feature, train_unlabeled_feature, train_label,
                            test_feature, test_label)
    predicted_label, predicted_score = autoencoder.evaluate_model(test_feature)
    roc=roc_auc_score(test_label, predicted_score)
    print("roc= %.6lf" %(roc))
    # predicted_label = get_label_n(predicted_score, contam)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_autoencoder_torch(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, contam):
    autoencoder = AutoEncoder(train_labeled_feature.shape[-1])
    autoencoder.train_model(train_labeled_feature, train_unlabeled_feature, train_label,
                            test_feature, test_label)
    predicted_label, predicted_score, classify_score = autoencoder.evaluate_model(test_feature, test_label)
    roc=roc_auc_score(test_label, predicted_score)
    print("roc= %.6lf" %(roc))
    predicted_score = StandardScaler().fit_transform(predicted_score.reshape(-1, 1)).reshape(-1)
    classify_score = StandardScaler().fit_transform(classify_score.reshape(-1, 1)).reshape(-1)
    predicted_score = predicted_score + classify_score
    roc=roc_auc_score(test_label, predicted_score)
    print("roc_new= %.6lf" %(roc))
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_autoencoder_chain(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, contam):
    autoencoder = AutoEncoderChain(train_labeled_feature.shape[-1])
    autoencoder.train_model(train_labeled_feature, train_unlabeled_feature, train_label,
                            test_feature, test_label)
    predicted_label, predicted_score = autoencoder.evaluate_model(test_feature, test_label)
    precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label, test_raw_label)
    print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
         %(precision, recall, f1_score, accuracy))

def exec_ladder_net(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, contam):
    n_rep = train_unlabeled_feature.shape[0] // train_labeled_feature.shape[0]
    train_labeled_feature = np.concatenate([train_labeled_feature]*n_rep)
    train_label = to_categorical(np.concatenate([train_label]*n_rep))
    model = get_ladder_network_fc(layer_sizes=[train_labeled_feature.shape[-1], 100, 80, 50, 10, 2])
    for _ in range(20):
        model.fit([train_labeled_feature, train_unlabeled_feature], train_label, epochs=1)
        predicted_label = model.test_model.predict(test_feature, batch_size=100)
        precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label.argmax(-1), test_raw_label)
        print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
            %(precision, recall, f1_score, accuracy))
        # print("Test accuracy : %f" % accuracy_score(test_label, predicted_label.argmax(-1)))
    # roc=roc_auc_score(test_label, predicted_score)
    # print("roc= %.6lf" %(roc))
    # predicted_label = get_label_n(predicted_score, contam)
    # precision, recall, f1_score, accuracy = eval_data(test_label, predicted_label.argmax(-1), test_raw_label)
    # print("precision = %.6lf\nrecall = %.6lf\nf1_score = %.6lf\naccuracy = %.6lf"
    #      %(precision, recall, f1_score, accuracy))


def main(args):
    dataset = read_data(args.inputpath)
    print("Reading Data done......")

    dataset = shuffle_data(dataset)
    train_data, test_data = split_dataset_horizontal(dataset, 0.6, True)
    train_labeled_data, train_unlabeled_data = split_dataset_horizontal(train_data, 0.01, False)
    train_labeled_feature, train_label, train_raw_label = split_dataset_vertical(train_labeled_data)
    train_unlabeled_feature, _, _ = split_dataset_vertical(train_unlabeled_data)
    test_feature, test_label, test_raw_label = split_dataset_vertical(test_data)
    train_labeled_feature, scalar = scale_data(train_labeled_feature)
    train_unlabeled_feature, _ = scale_data(train_unlabeled_feature, scalar)
    test_feature, _ = scale_data(test_feature, scalar)
    print("Preprocessing Data done......")
    # exec_autoencoder_keras(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
    #                  test_feature, test_label, test_raw_label, args.contam)
    # exec_autoencoder_torch(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
    #                  test_feature, test_label, test_raw_label, args.contam)
    # exec_autoencoder_chain(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
    #                  test_feature, test_label, test_raw_label, args.contam)
    exec_ladder_net(train_labeled_feature, train_label, train_raw_label, train_unlabeled_feature,
                     test_feature, test_label, test_raw_label, args.contam)


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
