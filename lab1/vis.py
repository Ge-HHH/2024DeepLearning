import numpy as np
import matplotlib.pyplot as plt
import re

path= './exp2'
path='./gridsearch_exp3/1e-4_0.1'
path='./exp7'

with open(path+'/log.txt','r') as f:
    lines=f.readlines()
    train_loss=[]
    val_loss=[]
    val_acc=[]
    train_epoch=[]
    val_epoch=[]
    for line in lines:
        sp=line.split(' ')
        #正则表达式去除非数字字符，只保留数字和小数点
        a=[re.sub(r'[^0-9.]', '',tmp) for tmp in sp]
        if len(a)==2:
            train_epoch.append(int(a[0]))
            train_loss.append(float(a[1]))
        elif len(a)==3:
            val_epoch.append(int(a[0]))
            val_acc.append(float(a[1]))
            val_loss.append(float(a[2]))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_epoch,train_loss,label='train loss')
    plt.plot(val_epoch,val_loss,label='val loss')
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(val_epoch,val_acc)
    plt.title('validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(path+'/loss_acc.png')
    plt.show()