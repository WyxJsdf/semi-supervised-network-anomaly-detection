import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sequence_model.rae import Encoder as RAEEncoder
from sequence_model.rae import Decoder as RAEDecoder

CUDA = 'cuda:4'

def get_device():
    if torch.cuda.is_available():
        device = CUDA
    else:
        device = 'cpu'
    print(device)
    return device

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self._device = get_device()

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        alpha = self.alpha[ids.data.view(-1)].to(self._device)

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Config(object):
  def __init__(self, embedding_size, window_size):
    self.device = get_device()
    self.labels = [1, 0]
    # training parameters
    self.num_layers = 2
    self.batch_size = 64
    self.lr = 0.0005
    self.save_path = "result"
    self.init()

    # model parameters
    self.hidden_size = 32
    self.embedding_size = embedding_size
    self.num_classes = 2
    self.seq_length = window_size

  def init(self):
    if not os.path.exists(self.save_path):
      os.mkdir(self.save_path)

class Encoder(nn.Module):
    def __init__(self):
        pass
    def forward(self, input_data):
        pass

class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self, input_data):
        pass

class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        # self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.config = config

        self.encoder = RAEEncoder(config.device, config.embedding_size, config.hidden_size)
        self.decoder = RAEDecoder(config.device, config.hidden_size, config.embedding_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.supervised = True

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def forward(self, input_data, seq_lengths):
        """
        :param input_data: [batch_size, seq_length]
        :return:
        """
        sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        packed_input = nn.utils.rnn.pack_padded_sequence(input_data[indices],
                                                         sorted_seq_lengths,
                                                         batch_first=True)
        classify_out, output = self.encoder(packed_input)
        if self.supervised:
            # output, _ = nn.utils.rnn.pad_packed_sequence(classify_out, batch_first=True)
            output = output[desorted_indices]
            # output = self.last_timestep(output, seq_lengths)
            return self.classifier(output)
        output = output.repeat(self.config.seq_length, 1)
        output = output.reshape(-1, self.config.seq_length, self.config.hidden_size)
        packed_output = nn.utils.rnn.pack_padded_sequence(output,
                                                          sorted_seq_lengths,
                                                          batch_first=True)
        output = self.decoder(packed_output)
        output = output[desorted_indices]
        if output.shape[1] < self.config.seq_length:
            output = F.pad(output, (0, 0, 0, self.config.seq_length - output.shape[1]))
        # output = self.linear(output)
        # output = self.softmax(output)
        return output

    def set_supervised_flag(self,supervised):
        self.supervised = supervised


class LSTMAutoEncoderChain():
    def __init__(self, embedding_size, window_size, cuda):
        global CUDA
        CUDA=cuda
        self._config = Config(embedding_size, window_size)
        self._device = get_device()
        # self._model = TextRnn(self._config).double()
        self._model = LSTMModel(self._config).double()
        self._criterion = nn.MSELoss(size_average=True)
        # self._criterion_classify = nn.CrossEntropyLoss()
        self._criterion_classify = FocalLoss(alpha=torch.Tensor([1, 0.25]))

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        self._log_interval = 100
        self._model = self._model.to(self._device)


    def train_model(self, train_labeled_data, train_unlabeled_data, validation_data, epoch=2000, batch_size=128):
        # feature_labeled = feature_labeled.trans
        feature_unlabeled, seq_length_unlabeled = train_unlabeled_data
        feature_labeled, label, seq_length_labeled = train_labeled_data
        print(feature_unlabeled.shape)
        train_dataset_labeled = Data.TensorDataset(torch.from_numpy(feature_labeled),
                                                   torch.from_numpy(label),
                                                   torch.from_numpy(seq_length_labeled))
        train_dataset_unlabeled = Data.TensorDataset(torch.from_numpy(feature_unlabeled),
                                                     torch.from_numpy(seq_length_unlabeled))
        train_loader_labeled = Data.DataLoader(dataset=train_dataset_labeled, batch_size=batch_size, shuffle=True)
        train_loader_unlabeled = Data.DataLoader(dataset=train_dataset_unlabeled, batch_size=batch_size, shuffle=True)
        self._model.train()
        for epoch_id in range(10):
            train_loss = 0
            self._model.set_supervised_flag(False)
            for step, (train_batch, train_seq_length) in enumerate(train_loader_unlabeled):
                train_batch = train_batch.to(self._device)
                output = self._model(train_batch, train_seq_length)
                loss = self._criterion(output, train_batch)
                self._optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.cpu().numpy()
                self._optimizer.step()

                if (step + 1)% self._log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                        epoch_id, (step + 1)* len(train_batch), len(train_loader_unlabeled.dataset),
                        100. * (step + 1) / len(train_loader_unlabeled), train_loss / self._log_interval))
                    train_loss = 0

        for epoch_id in range(epoch):
            train_loss = 0
            self._model.set_supervised_flag(True)
            for step, (train_batch, train_label, train_seq_length) in enumerate(train_loader_labeled):
                train_batch = train_batch.to(self._device)
                train_label = train_label.to(self._device)
                decoded = self._model(train_batch, train_seq_length)
                loss = self._criterion_classify(decoded, train_label.long())
                self._optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.cpu().numpy()
                self._optimizer.step()

                if (step + 1)% self._log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                        epoch_id, (step + 1)* len(train_batch), len(train_loader_labeled.dataset),
                        100. * (step + 1) / len(train_loader_labeled), train_loss / self._log_interval))
                    train_loss = 0
            if epoch_id % 200 == 0:
                val_label, val_score, classify_score = self.evaluate_model(validation_data)
                self._model.train()
                roc=roc_auc_score(validation_data[1], val_score)
                print("roc auc= %.6lf" %(roc))
                roc=roc_auc_score(validation_data[1], classify_score)
                print("roc auc classifer= %.6lf" %(roc))
                accuracy = accuracy_score(validation_data[1], val_label)
                f1 = f1_score(validation_data[1], val_label, average='binary', pos_label=1)
                print('Validation Data Accuray = %.6lf' %(accuracy))
                print('Validation Data F1 Score = %.6lf' %(f1))



    def get_distance(self, X, Y, seq_length):
        euclidean_sq = np.square(Y - X)
        return np.sum(np.sqrt(np.sum(euclidean_sq, axis=2)), axis=1).ravel() / (seq_length.astype(np.double))

    def evaluate_model(self, test_data):
        feature, label, seq_length = test_data
        self._model.eval()
        test_dataset = Data.TensorDataset(torch.from_numpy(feature),
                                       torch.from_numpy(seq_length))
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        output_feature = []
        output_label = []; output_score = []
        test_loss = 0
        for step, test_data in enumerate(test_loader):
            test_batch, test_seq_length = test_data
            test_batch = test_batch.to(self._device)

            self._model.set_supervised_flag(False)
            output = self._model(test_batch, test_seq_length)
            output_feature.append(output.data.cpu().numpy())
            test_loss += self._criterion(output, test_batch).data.cpu().numpy()

            self._model.set_supervised_flag(True)
            output = self._model(test_batch, test_seq_length)
            output = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            score = output.data[:, 1]
            h = predicted.cpu().numpy()
            output_label.append(h)
            output_score.append(score.cpu().numpy())


        test_loss /= len(test_loader)                                           # loss function already averages over batch size
        print('\nTesting set: Average loss: {:.4f}\n'.format(
        test_loss))
        predicted_score = np.concatenate(output_feature, axis=0)
        predicted_label = np.concatenate(output_label, axis=0)
        classify_score = np.concatenate(output_score, axis=0)
        return predicted_label, self.get_distance(feature, predicted_score, seq_length), classify_score
