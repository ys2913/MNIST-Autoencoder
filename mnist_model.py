from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import Loader
import pandas as pd
from torch.autograd import Variable


FILE_TRAIN_LABELED      = "data/train_labeled_aug.p"
FILE_TRAIN_UNLABELED    = "data/train_unlabeled.p"
FILE_VALIDATION         = "data/validation.p"
FILE_TEST               = "data/test.p"

MODEL_PARAMS            = 'model_params_SAWASA.p'


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs-unsupervised', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--epochs-supervised', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 120)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-as', type=str, default="starter_ae", metavar='N',
                    help='Name of file in which to store model')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_loss = []
train_accuracy = []
test_accuracy = []
validation_accuracy = []

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('loading data!')

class Loader():
    def __init__(self, train, utrain, valid, test, kwargs):
        self.train_labeled = train
        self.train_unlabeled = utrain
        self.validation = valid
        self.test = test
        self.kwargs = kwargs

    def getLabeledtrain(self):
        print('Loading Labeled Training Data!')
        trainset_labeled = pickle.load(open(self.train_labeled, "rb"))
        train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **self.kwargs)
        print('Loaded Labeled Training Data!')
        return train_loader

    def getUnlabeledtrain(self):
        print('Loading Unlabeled Training Data!')
        trainset_unlabeled = pickle.load(open(self.train_unlabeled, "rb"))
        labels = torch.Tensor(trainset_unlabeled.train_data.size()[0])
        labels.fill_(0)
        trainset_unlabeled.train_labels = labels
        train_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True, **self.kwargs)
        print('Loaded Unlabeled Training Data!')
        return train_loader

    def getValidation(self):
        print('Loading Validation Data!')
        valid_set = pickle.load(open(self.validation, "rb"))
        train_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False, **self.kwargs)
        print('Loaded Validation Data!')
        return train_loader

    def getTest(self):
        print('Loading Test Data!')
        test_set = pickle.load(open(self.test, "rb"))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, **self.kwargs)
        print('Loaded Test Data!')
        return test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 28 * 28 goes to 24 * 24 followed by 12 * 12 on maxpooling
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 12 * 12 goes to 8 * 8 followed by 4 * 4 on maxpooling
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)   # 4 * 4 * 20 = 320
        self.fc2 = nn.Linear(50, 10)

        # Decoder
        self.dfc2 = nn.Linear(10, 50)
        self.dfc1 = nn.Linear(50, 320)
        self.dconv2 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.dconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.supervised = False

    def forward(self, x):
        # ENCODER
        x, indices1 = F.max_pool2d(self.conv1(x), 2, return_indices=True)
        x = F.relu(x)
        x, indices2 = F.max_pool2d(self.conv2_drop(self.conv2(x)), 2, return_indices=True)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # For Supervised Data- encoder output
        if self.supervised:
            return F.log_softmax(x)

        # DECODER
        x = self.dfc2(x)
        x = self.dfc1(x)
        x = x.view(-1, 20, 4, 4)
        x = F.relu(self.dconv2(F.max_unpool2d(x, kernel_size=2, indices=indices2, stride=2)))
        x = F.relu(self.dconv1(F.max_unpool2d(x, kernel_size=2, indices=indices1, stride=2)))
        return x

    def set_supervised_flag(self,supervised):
        self.supervised = supervised

data_loader = Loader(FILE_TRAIN_LABELED, FILE_TRAIN_UNLABELED, FILE_VALIDATION, FILE_TEST, kwargs)
train_loader = data_loader.getLabeledtrain()
unlabeled_loader = data_loader.getUnlabeledtrain()
valid_loader = data_loader.getValidation()

model = Net()
if args.cuda:
    model.cuda()

# L2 loss for reconstruction loss
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train_unsupervised(epoch):
    model.set_supervised_flag(False)
    model.train()
    for batch_idx, (data,target) in enumerate(unlabeled_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(unlabeled_loader.dataset),
                100. * batch_idx / len(unlabeled_loader), loss.data[0]))
    return

def train_supervised(epoch):
    model.set_supervised_flag(True)
    model.train()
    correct = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1)[1]                                        # get the index of the max log-probability        
        correct += pred.eq(target.data).cpu().sum()
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    train_accuracy.append((100.0 * correct) / len(train_loader.dataset))    
    train_loss.append(total_loss)
    return

testing = False
def test(epoch, test_loader):
    model.set_supervised_flag(True)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]                                        # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)                                           # loss function already averages over batch size
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if testing:
        test_accuracy.append((100.0 * correct) / len(test_loader.dataset))
    else:
        validation_accuracy.append((100.0 * correct) / len(test_loader.dataset))
    return    







# Pretraining the weights for encoder by training the autoencoder
for epoch in range(1, args.epochs_unsupervised + 1):
    train_unsupervised(epoch)

# Training the encoder of the weight initialized encoder 
for epoch in range(1, args.epochs_supervised + 1):
    if epoch == 50:
        #Updating learning rate and momentum after 50 epochs
        optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.9)
    train_supervised(epoch)
    test(epoch, valid_loader)

# Storing the training data and validation accuracy per epoch for plotting
"""
with open(args.save_as + "_train_acc.p", 'wb') as file:
    pickle.dump(train_accuracy, file)

with open(args.save_as + "_valid_acc.p", 'wb') as file:
    pickle.dump(validation_accuracy, file) 

with open(args.save_as + "_train_loss.p", 'wb') as file:
    pickle.dump(train_loss, file)
"""

# saving model params
torch.save(model.state_dict(), MODEL_PARAMS)

print("Model parameters saved to : " + MODEL_PARAMS)

