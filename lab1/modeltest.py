import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import CNN_model
import json

import os

if __name__ == '__main__':
    model=CNN_model.SimpleCNN()
    # model.load_model('./exp2/epoch120_acc83.27.pt')
    
    model.load_model('./exp7/final.pt')
    datapath= 'D:/code/Datasets/'

    model.load_data(datapath)
    
    num_epochs=120
    batch_size=256
    learning_rate=0.001
    weight_decay=1e-4
    dropout=0.1
    val_per_epoch=10
    # optimizer = 'SGD'
    optimizer = 'Adam'

    #test
    model.test(visualize=True)