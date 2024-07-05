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

    datapath= 'D:/code/Datasets/'
    path='./exp_gridsearch'
    num=0
    while os.path.exists(path+str(num)):
        num+=1
    path=path+str(num)
    os.mkdir(path)

    
    num_epochs=2
    batch_size=256
    learning_rate=0.001
    # weight_decay=1e-5
    # dropout=0.2
    val_per_epoch=2
    optimizer = 'Adam'
    weight_decay_list = ['1e-6','1e-5',' 1e-4']
    dropout_list = ['0.1','0.2','0.3']
    dict={'num_epochs':num_epochs,'batch_size':batch_size,'learning_rate':learning_rate,'weight_decay':0,'dropout':0,'optimizer':optimizer,'val_per_epoch':val_per_epoch}

    with open(path+'/config.json','w') as f:
        json.dump(dict,f)
    for a in weight_decay_list:
        for b in dropout_list:
            weight_decay = float(a)
            dropout = float(b)
            file_name=path+f'/weight_decay_{a}_dropout_{b}'
            model=CNN_model.SimpleCNN()
            model.load_data(datapath)
            current_epoch=0
            best_acc = 0
            best_loss = 100
            dict['weight_decay'] = weight_decay
            dict['dropout'] = dropout
            while current_epoch<num_epochs:
                train_loss=model.train(num_epochs=val_per_epoch,batch_size=batch_size,lr=learning_rate,weight_decay=weight_decay,dropout=dropout,optimizer=optimizer,currepoch=current_epoch)
                current_epoch+=val_per_epoch
                acc,acc_class,loss=model.test()
                if best_acc < acc:
                    best_acc = acc
                    best_loss = loss
                    model.save_model(file_name+f'_best.pt')
                    with open(file_name+'.json','w') as f:
                        json.dump(dict,f)
            print('Finished Training')

    # #test
    # model.test(visualize=True)