import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse
import numpy as np

from models import *
import models
import torchvision
import torchvision.transforms as transforms

import time
import math

parser = argparse.ArgumentParser(description='PyTorch CIFAR Testing')
parser.add_argument('--data', default='data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
parser.add_argument('--gpu-id', type=str, default='0')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------------------
# dataset
if args.dataset == 'cifar10':
    num_classes = 10
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                           transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                 [0.24703233, 0.24348505, 0.26158768]),
                                            ]))

elif args.dataset == 'cifar100':
    num_classes = 100
    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761]),
                                            ]))

elif args.dataset == 'folder':
    num_classes = 10
    testset = torchvision.datasets.ImageFolder(root=args.data+'testing',
                                            transform=transforms.Compose([
                                             transforms.Resize(32),
                                             # transforms.Grayscale(1), # if use gray scale image
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                  [0.24703233, 0.24348505, 0.26158768]),
                                             # transforms.Normalize((0.4505,), (0.3119,))
                                           ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

# --------------------------------------------------------------------------------------------

# Model
print('==> Loading model..')
modelfilename = 'best.pth'

# net = torch.load(modelfilename)

# for trained on GPU but run on CPU
net = torch.load(modelfilename, map_location=torch.device('cpu'))
net = net.module

print('==> Model built.')

net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(testset.imgs[batch_idx][0].split('/')[-1].split('.')[0])

    acc = 100. * correct/total
    print('Test accuracy: ', acc)
    print('Test loss: ', test_loss/len(testset))

if __name__ == '__main__':
    global_time = time.time()
    test()
    print(time.time()-global_time)
