import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
  def __init__(self, embedding_size):
    self.device = get_device()
    self.labels = [1, 0]
    # training parameters
    self.num_layers = 2
    self.batch_size = 64
    self.lr = 0.001
    self.save_path = "result"
    self.init()

    # model parameters
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
    self.config=config
    # self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
    self.lstm      = nn.LSTM(
      input_size = config.embedding_size,
      hidden_size = config.hidden_size,
      num_layers = config.num_layers,
      bias = True,
      batch_first = True,
      # dropout = 0.1,
      bidirectional = True
    )
    self.linear    = nn.Linear(
      in_features = config.hidden_size * 2,
      out_features = config.num_classes
    )
    self.softmax   = nn.Softmax()

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

    padded_output, (hn, cn) = self.lstm(packed_input)
    output, _ = nn.utils.rnn.pad_packed_sequence(padded_output, batch_first=True)
    output = output[desorted_indices]
    output = self.last_timestep(output, seq_lengths)
    # output, (hn, cn) = self.lstm(input_data)
    # output = output[:, -1, :].squeeze(dim=1)

    # hn = hn.reshape(-1, self.config.hidden_size)
    # output = self.linear(hn[desorted_indices])
    output = self.linear(output)
    return output


class LSTMClassifier():
    def __init__(self, embedding_size, cuda):
        global CUDA
        CUDA=cuda
        self._config = Config(embedding_size)
        self._model = TextRnn(self._config).double()
        self._criterion = nn.CrossEntropyLoss()
        self._criterion_classify = FocalLoss()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        self._log_interval = 10
        self._device = get_device()
        self._model = self._model.to(self._device)


    def train_model(self, train_labeled_data, validation_data, epoch=500, batch_size=64):
        # feature_labeled = feature_labeled.trans
        feature_labeled, label, seq_length = train_labeled_data
        print(feature_labeled.shape)
        train_data_labeled = Data.TensorDataset(torch.from_numpy(feature_labeled),
                                                torch.from_numpy(label),
                                                torch.from_numpy(seq_length))
        train_loader = Data.DataLoader(dataset=train_data_labeled, batch_size=batch_size, shuffle=True)
        train_loss = 0
        for epoch_id in range(epoch):
            for step, train_data in enumerate(train_loader):
                train_batch, train_label, train_seq_length = train_data
                nn.utils.rnn.pack_padded_sequence
                train_batch = train_batch.to(self._device)
                train_label = train_label.to(self._device)
                train_seq_length = train_seq_length.to(self._device)

                output = self._model(train_batch, train_seq_length)
                loss = self._criterion_classify(output, train_label.long())
                self._optimizer.zero_grad()
                loss.backward()
                train_loss += loss.data.cpu().numpy()
                self._optimizer.step()

                if (step + 1)% self._log_interval == 0:
                    predict = torch.argmax(output.data, dim=1)
                    train_acc = accuracy_score(train_label.data.cpu(), predict.cpu())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain acc: {:.8f}\tLoss: {:.8f}'.format(
                        epoch_id, (step + 1)* len(train_batch), len(train_loader.dataset),
                        100. * (step + 1) / len(train_loader), train_acc, train_loss / self._log_interval))
                    train_loss = 0
                # torch.save(model.state_dict(), os.path.join(config.save_path, "model.ckpt"))
            if epoch_id % 50 == 0:
                val_label, classify_score = self.evaluate_model(validation_data)
                self._model.train()
                accuracy = accuracy_score(validation_data[1], val_label)
                f1 = f1_score(validation_data[1], val_label, average='binary', pos_label=1)
                roc=roc_auc_score(validation_data[1], classify_score)
                print("roc_classify= %.6lf" %(roc))
                print('Validation Data Accuray = %.6lf' %(accuracy))
                print('Validation Data F1 Score = %.6lf' %(f1))


    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, test_data):
        feature, label, seq_length = test_data
        self._model.eval()
        test_data = Data.TensorDataset(torch.from_numpy(feature),
                                       torch.from_numpy(label),
                                       torch.from_numpy(seq_length))
        test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)
        output_feature = []
        output_label = []; output_score = []
        test_loss = 0
        ss = 0
        for test_batch, test_label, test_seq_length in test_loader:
            test_batch = test_batch.to(self._device)
            test_label = test_label.to(self._device)
            test_seq_length = test_seq_length.to(self._device)

            output = self._model(test_batch, test_seq_length)
            _, predicted = torch.max(output.data, 1)
            score = output.data[:, 1]
            h = predicted.cpu().numpy()
            ss += sum(h)
            output_label.append(h)
            output_score.append(score.cpu().numpy())

        test_loss /= len(test_loader)                                           # loss function already averages over batch size
        print(ss)
        print('\nTesting set: Average loss: {:.4f}\n'.format(
        test_loss))
        predicted_label = np.concatenate(output_label, axis=0)
        classify_score = np.concatenate(output_score, axis=0)
        return predicted_label, classify_score
