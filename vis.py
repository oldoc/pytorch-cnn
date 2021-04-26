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
import cv2

def vis(net,czoom = 10, fzoom = 2):
    layers = []

    for param in net.parameters():
        shape = param.data.shape
        # 如果是卷积层
        if len(shape) == 4:
            l = param.data.cpu().numpy()
            l = l.transpose((1,2,0,3)).reshape((shape[1]*shape[2],shape[0]*shape[3]))
            #print(l.shape)

            img = np.zeros((l.shape[0]*czoom, l.shape[1]*czoom), dtype=float)
            for i in range(l.shape[0]*czoom):
                for j in range(l.shape[1]*czoom):
                    img[i][j] = l[i//czoom][j//czoom]

            #间隔线
            xitv = np.ones(l.shape[1]*czoom)
            for i in range(shape[1]):
                img = np.insert(img, i + i * shape[3] * czoom, xitv, axis=0)
            yitv = np.ones(img.shape[0])
            for i in range(shape[0]+1):
                img = np.insert(img, i + i * shape[2] * czoom, yitv, axis=1)

        #如果是全连接层
        if len(shape) == 2:
            l = param.data.cpu().numpy()

            img = np.zeros((l.shape[0]*fzoom, l.shape[1]*fzoom), dtype=float)
            for i in range(l.shape[0]*fzoom):
                for j in range(l.shape[1]*fzoom):
                    img[i][j] = l[i//fzoom][j//fzoom]
            #print(img.shape)

            xitv = np.ones(img.shape[1])
            img = np.insert(img, img.shape[0], xitv, axis=0)

            yitv = np.ones(img.shape[0])
            img = np.insert(img, img.shape[1], yitv, axis=1)

        layers.append(img)
    maxy = max(layer.shape[1] for layer in layers)

    img = np.pad(layers[0],((0,0),(0,maxy - layers[0].shape[1])), 'constant')
    for i in range(1, len(layers)):
        img = np.concatenate((img, np.pad(layers[i],((0,0),(0,maxy - layers[i].shape[1])), 'constant')))

    img2 = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    img2 = img * 255
    
    img2 = img2.reshape((img2.shape[0], img2.shape[1],1))

    return img2
    
'''
#for test
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelfilename = 'best.pth'

    # for trained on GPU but run on CPU
    net = torch.load(modelfilename, map_location=torch.device('cpu'))
    net = net.module
    net = net.to(device)

    cv2.imwrite("result.jpg", vis(net))
'''
