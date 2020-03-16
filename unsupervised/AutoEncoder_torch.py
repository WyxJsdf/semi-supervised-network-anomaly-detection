import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:2'
    else:
        device = 'cpu'
    print(device)
    return 'cpu'
    return device

class ModelAutoEncoder(nn.Module):
    def __init__(self, num_features):
        super(ModelAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 50),
            nn.ReLU(True),
            nn.Linear(50, 25),
            nn.ReLU(True), nn.Linear(25, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 25),
            nn.ReLU(True),
            nn.Linear(25, 50),
            nn.ReLU(True),
            nn.Linear(50, num_features), nn.Tanh())
        self.classifier = nn.Linear(10, 2)
        self.supervised = True


    def forward(self, x):
        x = self.encoder(x)
        if self.supervised:
            x = F.log_softmax(self.classifier(x), dim=1)
            return x
        x = self.decoder(x)
        return x

    def set_supervised_flag(self,supervised):
        self.supervised = supervised


class AutoEncoder():
    def __init__(self, num_features):
        self._model = ModelAutoEncoder(num_features).double()
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._log_interval = 100
        self._device = get_device()
        # self._use_cuda = True
        # if self._use_cuda:
        #     self._model = self._model.cuda()
        self._model = self._model.to(self._device)

    def train_model(self, feature_unlabeled, test_feature, test_label, epoch=20, batch_size=64):
        train_data_unlabeled = Data.TensorDataset(torch.from_numpy(feature_unlabeled), torch.from_numpy(feature_unlabeled))
        train_loader_unlabeled = Data.DataLoader(dataset=train_data_unlabeled, batch_size=batch_size, shuffle=True)
        train_loss = 0
        for epoch_id in range(epoch):
            self._model.set_supervised_flag(False)
            for step, (train_batch, _) in enumerate(train_loader_unlabeled):
                train_batch = train_batch.to(self._device)
                decoded = self._model(train_batch)
                loss = self._criterion(decoded, train_batch)
                self._optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.cpu().numpy()
                self._optimizer.step()
                if (step + 1) % self._log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                        epoch_id, (step + 1)* len(train_batch), len(train_loader_unlabeled.dataset),
                        100. * (step + 1) / len(train_loader_unlabeled), train_loss / self._log_interval))
                    train_loss = 0
            predicted_score = self.evaluate_model(test_feature)
            self._model.train()
            roc=roc_auc_score(test_label, predicted_score)
            print("roc auc= %.6lf" %(roc))

    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, feature):
        self._model.eval()
        test_data = Data.TensorDataset(torch.from_numpy(feature), torch.from_numpy(feature))
        test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
        output_feature = []
        test_loss = 0

        for test_batch, _ in test_loader:
            test_batch = test_batch.to(self._device)
            output = self._model(test_batch)
            output_feature.append(output.data.cpu().numpy())
            test_loss += self._criterion(output, test_batch).data.cpu().numpy()
        test_loss /= len(test_loader)                                           # loss function already averages over batch size
        print('\nTesting set: Average loss: {:.4f}\n'.format(
        test_loss))
        predicted_score = np.concatenate(output_feature, axis=0)
        return self.get_distance(feature, predicted_score)
