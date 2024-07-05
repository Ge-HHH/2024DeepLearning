import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import LSTM
import patchLSTM
import patchLSTMCNN
import json

import os

if __name__ == '__main__':
    

    datapath= 'D:/code/Datasets/'  
    checkpoint_path='./exp5'
    num_epochs=200
    batch_size=256
    learning_rate=5e-3
    weight_decay=0.0001
    dropout=0.2
    val_per_epoch=10
    model_name='patchLSTM'
    if checkpoint_path is not None:
        with open(checkpoint_path+'/config.json','r') as f:
            dict=json.load(f)
            # num_epochs=dict['num_epochs']
            batch_size=dict['batch_size']
            learning_rate=dict['learning_rate']
            weight_decay=dict['weight_decay']
            dropout=dict['dropout']
            val_per_epoch=dict['val_per_epoch'] if 'val_per_epoch' in dict else 10
            if 'model_name' in dict:
                model_name=dict['model_name']
            else:
                model_name='LSTM'

    if model_name=='LSTM':
        model=LSTM.SimpleLSTM()
    elif model_name=='patchLSTM':
        model=patchLSTM.SimpleLSTM()
    elif model_name=='patchLSTMCNN':
        model=patchLSTMCNN.SimpleLSTM()
    else:
        raise ValueError('Model not found')
    if checkpoint_path is not None:
        model.load_model(checkpoint_path+'/final.pt')
        with open(checkpoint_path+'/log.txt','r') as f:
            lines=f.readlines()
            i=-1
            if 'val' in lines[-1]:
                i=-2
            #find 'epoch' and ' ' in the line,between them is the epoch number
            current_epoch=int(lines[i][lines[i].find('epoch')+5:lines[i].find(' ',lines[i].find('epoch')+5)])+1       
    model.load_data(datapath)
    model.test(visualize=True)