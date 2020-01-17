import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:6" if use_cuda else "cpu")
# cudnn.benchmark = True

class ModelAutoEncoder(nn.Module):
    def __init__(self, num_features):
        super(ModelAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(True),
            nn.Linear(100, 80),
            nn.ReLU(True), nn.Linear(80, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 80),
            nn.ReLU(True),
            nn.Linear(80, 100),
            nn.ReLU(True),
            nn.Linear(100, num_features), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoder():
    def __init__(self, num_features):
        self._model = ModelAutoEncoder(num_features).double()
        self._criterion = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        self._log_interval = 10
        self._use_cuda = True
        if self._use_cuda:
            self._model = self._model.cuda()

    def train_model(self, feature_labeled, feature_unlabeled, label, epoch=20, batch_size=64):
        train_data_labeled = Data.TensorDataset(torch.from_numpy(feature_labeled), torch.from_numpy(label))
        train_data_unlabeled = Data.TensorDataset(torch.from_numpy(feature_unlabeled), torch.from_numpy(feature_unlabeled))
        train_loader_labeled = Data.DataLoader(dataset=train_data_labeled, batch_size=batch_size, shuffle=True)
        train_loader_unlabeled = Data.DataLoader(dataset=train_data_unlabeled, batch_size=batch_size, shuffle=True)

        for epoch_id in range(epoch):
            for step, (train_batch, _) in enumerate(train_loader_labeled):
                if self._use_cuda:
                    train_batch = train_batch.cuda()
                decoded = self._model(train_batch)
                loss = self._criterion(decoded, train_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                if step % self._log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch_id, step * len(train_batch), len(train_loader_labeled.dataset),
                        100. * step / len(train_loader_labeled), loss.data.cpu().numpy()))

    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, feature):
        self._model.eval()
        test_data = Data.TensorDataset(torch.from_numpy(feature), torch.from_numpy(feature))
        test_loader = Data.DataLoader(dataset=feature, batch_size=64, shuffle=False)
        output_feature = []
        test_loss = 0

        for test_batch, _ in test_loader:
            if self._use_cuda:
                test_batch = test_batch.cuda()
            output = self._model(test_batch)
            output_feature.append(output.data.cpu().numpy())
            test_loss += self._criterion(output, test_batch).data.cpu().numpy()
        test_loss /= len(test_loader)                                           # loss function already averages over batch size
        print('\nTesting set: Average loss: {:.4f}\n'.format(
        test_loss))
        return self.get_distance(X, predicted_score)
