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
    model.load_model('./gridsearch_exp3/1e-4_0.1/epoch60_acc80.71.pt')
    datapath= 'D:/code/Datasets/'
    path='./exp'
    num=0
    while os.path.exists(path+str(num)):
        num+=1
    path=path+str(num)
    os.mkdir(path)

    model.load_data(datapath)
    
    num_epochs=120
    batch_size=256
    learning_rate=0.001
    weight_decay=1e-4
    dropout=0.1
    val_per_epoch=10
    # optimizer = 'SGD'
    optimizer = 'Adam'

    dict={'num_epochs':num_epochs,'batch_size':batch_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'dropout':dropout,'optimizer':optimizer,'val_per_epoch':val_per_epoch}

    with open(path+'/config.json','w') as f:
        json.dump(dict,f)

    current_epoch=60
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

    #test
    model.test(visualize=True)