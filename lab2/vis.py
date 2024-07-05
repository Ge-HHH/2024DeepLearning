import numpy as np
import matplotlib.pyplot as plt
import re

path= './exp4'
# path='./gridsearch_exp3/1e-4_0.1'
# path='./exp7'

def visualize(path,model_name='patchLSTM'):
    with open(path+'/log.txt','r') as f:
        lines=f.readlines()
        train_loss=[]
        val_loss=[]
        val_acc=[]
        train_epoch=[]
        val_epoch=[]
        for line in lines:
            if 'train' not in line and 'val' not in line:
                continue
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
        plt.subplot(1,2,1)
        plt.plot(train_epoch,train_loss,label=model_name)
        plt.title('Training loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(val_epoch,val_acc,label=model_name)
        plt.title('Validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()

paths=['./exp6','./exp4','./exp5']
names=['LSTM','PatchLSTM','PatchLSTM+CNN']

plt.figure(figsize=(10,4))

for i in range(len(paths)):
    visualize(paths[i],names[i])

plt.show()
