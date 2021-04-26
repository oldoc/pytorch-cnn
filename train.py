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
from utils import cal_param_size, cal_multi_adds

from bisect import bisect_right
import time
import math
from regularization.dropblock import LinearScheduler, SGDRScheduler

import vis
import cv2

# Cutout data enhance
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name')
parser.add_argument('--arch', default='HCGNet_A1', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-type', default='SGDR', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 225], type=list, help='milestones for lr-multistep')
parser.add_argument('--sgdr-t', default=10, type=int, dest='sgdr_t',help='SGDR T_0')
parser.add_argument('--epochs', type=int, default=1270, help='number of epochs to train')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--trainhcgnet', type=bool, default=False)
parser.add_argument('--savebest', type=bool, default=True)
parser.add_argument('--savelast', type=bool, default=True)

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
best_acc = 0.0  # best test accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
drop_scheduler = SGDRScheduler  # or 'LinearScheduler'
# -----------------------------------------------------------------------------------------
# dataset
if args.dataset == 'cifar10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True,
                                            transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                  [0.24703233, 0.24348505, 0.26158768]),
                                             Cutout(n_holes=1, length=16)
                                            ]))
    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True,
                                           transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                 [0.24703233, 0.24348505, 0.26158768]),
                                            ]))

elif args.dataset == 'cifar100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                             transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761])
                                            ]))

    testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761]),
                                            ]))

elif args.dataset == 'folder':
    num_classes = 4
    trainset = torchvision.datasets.ImageFolder(root=args.data+'training',
                                            transform=transforms.Compose([
                                             transforms.Resize(32),
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             #transforms.Grayscale(1), # if use gray scale image
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                  [0.24703233, 0.24348505, 0.26158768]),
                                             # transforms.Normalize((0.4505,), (0.3119,)), # for gray scale image
                                             Cutout(n_holes=1, length=16)
                                            ]))

    testset = torchvision.datasets.ImageFolder(root=args.data+'testing',
                                            transform=transforms.Compose([
                                             transforms.Resize(32),
                                             # transforms.Grayscale(1), # if use gray scale image
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                                                  [0.24703233, 0.24348505, 0.26158768]),
                                             # transforms.Normalize((0.4505,), (0.3119,))
                                           ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))

# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
if args.trainhcgnet:
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes)
else:
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetV2(0.5)
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
print('==> Model built.')

net = net.to(device)

net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/cp.pth.tar')
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
'''

def adjust_lr(optimizer, epoch, eta_max=0.1, eta_min=0.):
    cur_lr = 0.
    if args.lr_type == 'SGDR':
        i = int(math.log2(epoch / args.sgdr_t + 1))
        T_cur = epoch - args.sgdr_t * (2 ** (i) - 1)
        T_i = (args.sgdr_t * 2 ** i)

        cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

    elif args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    """
    if args.arch == 'HCGNet_A2' or args.arch == 'HCGNet_A3' and epoch > 629:
        epoch = epoch - 630
    """

    lr = adjust_lr(optimizer, epoch)
    drop_scheduler.global_epoch = epoch
    if drop_scheduler is LinearScheduler:
        drop_scheduler.num_epochs =epoch
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 1.)

        optimizer.zero_grad()
        outputs = net(inputs)
        
        sparse = False
        regu_loss_weight = 0.0001
        if sparse:
            regu_loss = 0
            for param in net.parameters():
                regu_loss += torch.sum(torch.abs(param))

            loss = regu_loss_weight * regu_loss +  mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Epoch:{0}\t lr:{1:.3f}\t duration:{2:.3f}'
          .format(epoch, lr, time.time()-start_time, 100. * correct/total, train_loss/len(trainset)))
    with open('result/'+ os.path.basename(__file__).split('.')[0] +'.csv', 'a+') as f:
        f.write('{0},{1:.4f}'.format(epoch, lr))

    # transfer weight to image and save to img folder
    #cv2.imwrite("img/"+str(epoch).zfill(4)+".jpg", vis.vis(net))

def test(epoch):
    global best_acc
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

        with open('result/'+ os.path.basename(__file__).split('.')[0] +'.csv', 'a+') as f:
            f.write(','+str(correct / total)+'\n')

    # Save checkpoint.
    acc = 100. * correct/total
    print('Test accuracy: ', acc)
    print('Test loss: ', test_loss/len(testset))
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }

    if args.savebest and (acc > best_acc):
        best_acc = acc
        torch.save(net, 'best.pth')
        print('Saved new best!')

if __name__ == '__main__':
    global_time = time.time()

    ep = open('result/'+ os.path.basename(__file__).split('.')[0] +'.csv', "w")
    ep.write("epoch,lr,acc\n")
    ep.close()

    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)

    if args.savelast:
        torch.save(net, 'last.pth')
        print('Saved last!')
    print(time.time()-global_time)




