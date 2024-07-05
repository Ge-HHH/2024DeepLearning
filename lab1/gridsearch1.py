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

def gridsearch(path, num_epochs=50, batch_size=256, learning_rate=0.001, weight_decay=1e-5, dropout=0.2, val_per_epoch=10, optimizer='Adam'):
    model=CNN_model.SimpleCNN()

    datapath= 'D:/code/Datasets/'
    num=0
    while os.path.exists(path+str(num)):
        num+=1
    path=path+str(num)
    os.mkdir(path)

    model.load_data(datapath)

    dict={'num_epochs':num_epochs,'batch_size':batch_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'dropout':dropout,'optimizer':optimizer,'val_per_epoch':val_per_epoch}

    with open(path+'/config.json','w') as f:
        json.dump(dict,f)

    current_epoch=0
    with open(path+'/log.txt','w') as f:
        while current_epoch<num_epochs:
            train_loss=model.train(num_epochs=val_per_epoch,batch_size=batch_size,lr=learning_rate,weight_decay=weight_decay,dropout=dropout,optimizer=optimizer,currepoch=current_epoch)
            for i,loss in enumerate(train_loss):
                f.write(f'train:epoch{current_epoch+i} loss:{loss}\n')
            current_epoch+=val_per_epoch
            acc,acc_class,loss=model.test()
            model.save_model(path+f'/epoch{current_epoch}_acc{acc}.pt')
            f.write(f'val:epoch{current_epoch} acc:{acc} loss:{loss}\n')
    f.close()
    print('Finished Training')

    model.save_model(path+'/final.pt')
if __name__ == '__main__':
    weight_decay_list = ['1e-6','1e-5',' 1e-4'] 
    dropout_list = ['0.1','0.2','0.3']

    for a in weight_decay_list:
        for b in dropout_list:
            weight_decay = float(a)
            dropout = float(b)
            path='gridsearch_'+a+'_'+b
            gridsearch(path, num_epochs=2, batch_size=256, learning_rate=0.001, weight_decay=weight_decay, dropout=dropout, val_per_epoch=2, optimizer='Adam')