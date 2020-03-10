import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    return 'cpu'
    return device

class Config(object):
  def __init__(self, embedding_size):
    self.device = get_device()
    self.labels = [1, 0]
    # 训练参数
    self.num_layers = 3
    self.batch_size = 64
    self.lr = 0.001
    self.save_path = "result"
    self.init()

    # 模型参数
    self.hidden_size = 128
    self.embedding_size = embedding_size
    self.num_classes = 2
    self.seq_length = 20

  def init(self):
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)

class TextRnn(nn.Module):
  def __init__(self, config):
    super(TextRnn, self).__init__()
    # self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    self.lstm      = nn.LSTM(
      input_size = config.embedding_size,
      hidden_size = config.hidden_size,
      num_layers = config.num_layers,
      bias = True,
      batch_first = True,
      dropout = 0.0,
      bidirectional = True
    )
    self.linear    = nn.Linear(
      in_features = config.hidden_size * 2,
      out_features = config.num_classes
    )
    self.softmax   = nn.Softmax()

  def forward(self, input_data):
    """
    :param input_data: [batch_size, seq_length]
    :return:
    """
    # [batch_size, seq_length, embedding_size]
    # output = self.embedding(input_data)
    # output [batch_size, seq_length, 2*hidden_size]
    output, _ = self.lstm(input_data)
    # [batch_size, 2*hidden_size]
    output = output[:, -1, :].squeeze(dim=1)
    # [batch_size, num_classes]
    output = self.linear(output)
    # output = self.softmax(output)
    return output


class LSTMClassifier():
    def __init__(self, embedding_size):
        self._config = Config(embedding_size)
        self._model = TextRnn(self._config).double()
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        self._log_interval = 100
        self._device = get_device()
        self._model = self._model.to(self._device)

    def train_model(self, feature_labeled, label, test_feature, test_label, epoch=10, batch_size=64):
        # feature_labeled = feature_labeled.trans
        print(feature_labeled.shape)
        label = torch.from_numpy(label)
        train_data_labeled = Data.TensorDataset(torch.from_numpy(feature_labeled),
                                             label)
        train_loader = Data.DataLoader(dataset=train_data_labeled, batch_size=batch_size, shuffle=True)
        train_loss = 0
        for epoch_id in range(epoch):
            for index, train_data in enumerate(train_loader):
                train_batch, train_label = train_data
                train_batch = train_batch.to(self._device)
                train_label = train_label.to(self._device)
                output = self._model(train_batch)
                loss = self._criterion(output, train_label.long())
                self._optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.cpu().numpy()
                self._optimizer.step()

                if index % self._log_interval == 0:
                    predict = torch.argmax(output.data, dim=1)
                    train_acc = accuracy_score(train_label.data.cpu(), predict.cpu())
                    print("train acc: {train_acc}; train loss: {loss}"
                        .format(train_acc=train_acc, loss=train_loss / self._log_interval))
                    train_loss = 0
                # torch.save(model.state_dict(), os.path.join(config.save_path, "model.ckpt"))

    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, feature, label):
        self._model.eval()
        test_data = Data.TensorDataset(torch.from_numpy(feature.reshape(feature.shape[1], -1, self._config.embedding_size)),
                                             torch.from_numpy(label))
        test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        output_feature = []
        output_label = []
        test_loss = 0
        ss = 0
        for test_batch, test_label in test_loader:
            test_batch = test_batch.to(self._device)
            test_label = test_label.to(self._device)

            output = self._model(test_batch)
            _, predicted = torch.max(output.data, 1)
            h = predicted.cpu().numpy()
            ss += sum(h)
            output_label.append(h)

        test_loss /= len(test_loader)                                           # loss function already averages over batch size
        print(ss)
        print('\nTesting set: Average loss: {:.4f}\n'.format(
        test_loss))
        predicted_label = np.concatenate(output_label, axis=0)
        return predicted_label
