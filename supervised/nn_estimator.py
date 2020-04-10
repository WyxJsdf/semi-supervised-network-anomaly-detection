import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.autograd import Variable


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    # return 'cpu'
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

class ModelAutoEncoder(nn.Module):
    def __init__(self, num_features):
        super(ModelAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(True),
            nn.Linear(100, 80),
            nn.ReLU(True), nn.Linear(80, 50))
        self.decoder = nn.Sequential(
            nn.Linear(10, 25),
            nn.ReLU(True),
            nn.Linear(25, 50),
            nn.ReLU(True),
            nn.Linear(50, num_features), nn.Tanh())
        self.classifier = nn.Linear(50, 2)
        self.supervised = True
        self.confidence = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Linear(50, 50),
            nn.ReLU(True),
            nn.Linear(50, 1)
        )


    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x), self.confidence(x)
        # return x

    def set_supervised_flag(self,supervised):
        self.supervised = supervised


class ModelNNTorch():
    def __init__(self, num_features, focal):
        self._model = ModelAutoEncoder(num_features).double()
        if focal:
            self._criterion_classify = FocalLoss(alpha=torch.Tensor([[1],[0.25]]))
        else:
            self._criterion_classify = nn.NLLLoss()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        # self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        # self._criterion_classify = F.binary_cross_entropy()
        # self._optimizer = torch.optim.Adagrad(self._model.parameters(), lr=0.01)
        self._log_interval = 10
        self._device = get_device()
        self._model = self._model.to(self._device)
        self.budget = 0.3
        self.theta = 1

    def encode_onehot(self, labels, n_classes=2):
        onehot = torch.FloatTensor(labels.size()[0], n_classes)
        labels = labels.data.long()
        onehot = onehot.to(self._device)
        onehot.zero_()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        return onehot
    def train_model(self, train_labeled_data, validate_data, epoch=20, batch_size=128):
        train_dataset_labeled = Data.TensorDataset(torch.from_numpy(train_labeled_data[0]), torch.from_numpy(train_labeled_data[1]))
        train_loader_labeled = Data.DataLoader(dataset=train_dataset_labeled, batch_size=batch_size, shuffle=True)
        lmbda = 0.1

        train_loss_0 = 0; train_loss_1 = 0; epoch_id = 0; step = 0
        iter_labeled = iter(train_loader_labeled)
        self._model.train()
        while epoch_id < epoch:
            try:
                train_batch, train_label = next(iter_labeled)
            except StopIteration:
                epoch_id += 1
                step = 0
                train_loss_0 = 0; train_loss_1 = 0
                if epoch_id % 1 == 0:

                    val_label, classify_score, confidence_score = self.evaluate_model(train_labeled_data)
                    self._model.train()
                    print('min conf = %.6lf\t max conf = %.6lf\t mean conf = %.6lf' %(np.min(confidence_score), np.max(confidence_score), np.mean(confidence_score)))
                    print_confidence_each_class(classify_score, train_labeled_data[2])
                    print_confidence_each_class(confidence_score, train_labeled_data[2])

                    val_label, classify_score, confidence_score = self.evaluate_model(validate_data)
                    self._model.train()
                    print('min conf = %.6lf\t max conf = %.6lf\t mean conf = %.6lf' %(np.min(confidence_score), np.max(confidence_score), np.mean(confidence_score)))
                    print_confidence_each_class(classify_score, validate_data[2])
                    print_confidence_each_class(confidence_score, validate_data[2])
                    accuracy = accuracy_score(validate_data[1], val_label)
                    f1 = f1_score(validate_data[1], val_label, average='binary', pos_label=1)
                    # print('Validation Data Accuray = %.6lf' %(accuracy))
                    print('Validation Data F1 Score = %.6lf' %(f1))
                    roc=roc_auc_score(validate_data[1], classify_score)
                    print("roc_classify= %.6lf" %(roc))
                iter_labeled = iter(train_loader_labeled)
                train_batch, train_label = next(iter_labeled)

            train_batch = train_batch.to(self._device)
            train_label = train_label.to(self._device)
            labels_onehot = Variable(self.encode_onehot(train_label))
            
            pred_original, confidence = self._model(train_batch)
            confidence = F.sigmoid(confidence)
            pred_original = F.softmax(pred_original, dim=-1)

            # Make sure we don't have any numerical instability
            eps = 1e-12
            # pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            # b = Variable(torch.bernoulli(torch.ones(confidence.size())*0.8)).to(self._device)
            b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).to(self._device)
            conf = confidence * b + (1 - b)
            # conf = confidence
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
            pred_new = torch.log(pred_new)

            xentropy_loss = self._criterion_classify(pred_new, train_label.long())
            confidence_loss = -torch.log(confidence)
            confidence_loss[train_label == 1] = confidence_loss[train_label == 1] * self.theta
            # confidence_loss[train_label == 0] = confidence_loss[train_label == 0] * (1-self.theta)
            confidence_loss = torch.mean(confidence_loss)


            total_loss = xentropy_loss + (lmbda * confidence_loss)

            if True:#step % 10 == 0:
                if self.budget > confidence_loss.data:
                    lmbda = lmbda / 1.01
                elif self.budget <= confidence_loss.data:
                    lmbda = lmbda / 0.99

            self._optimizer.zero_grad()
            total_loss.backward()
            train_loss_1 += confidence_loss.data.cpu().numpy()
            train_loss_0 += xentropy_loss.data.cpu().numpy()
            self._optimizer.step()

            if (step + 1) % self._log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Train Loss: {:.8f} \t Conf Loss: {:.8f} \t Lmbda:{}'.format(
                    epoch_id, (step + 1)* len(train_batch), len(train_loader_labeled.dataset),
                    100. * (step + 1) / len(train_loader_labeled), train_loss_0 / self._log_interval, train_loss_1 / self._log_interval, lmbda))
                train_loss_0 = 0; train_loss_1 = 0


            step += 1


    def get_distance(self, X, Y):
        euclidean_sq = np.square(Y - X)
        return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()

    def evaluate_model(self, test_data):
        self._model.eval()
        test_dataset = Data.TensorDataset(torch.from_numpy(test_data[0]), torch.from_numpy(test_data[1]))
        test_loader = Data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        output_feature = []
        output_label = []; output_score = []
        output_confidence = []

        test_loss = 0
        for test_batch, test_label in test_loader:
            test_batch = test_batch.to(self._device)
            test_label = test_label.to(self._device)
            self._model.set_supervised_flag(True)
            output, confidence = self._model(test_batch)
            output = F.softmax(output, dim=1)
            confidence = F.sigmoid(confidence)
            _, predicted = torch.max(output.data, 1)
            score = output.data[:, 1]
            h = predicted.cpu().numpy()
            output_label.append(h)
            output_score.append(score.cpu().numpy())
            output_confidence.append(confidence.data.view(-1).cpu().numpy())

        # test_loss /= len(test_loader)                                           # loss function already averages over batch size
        # print('\nTesting set: Average loss: {:.4f}\n'.format(
        # test_loss))
        predicted_label = np.concatenate(output_label, axis=0)
        classify_score = np.concatenate(output_score, axis=0)
        confidence_score = np.concatenate(output_confidence, axis=0)

        return predicted_label, classify_score, confidence_score
