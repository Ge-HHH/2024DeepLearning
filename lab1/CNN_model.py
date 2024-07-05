import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class SimpleCNN:
    def __init__(self):
        # 定义数据预处理
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # 实例化模型
        self.net = self._create_model()

        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

        # 检查并设置设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
    def load_data(self, path):
        # 加载数据集
        self.trainset = CIFAR10(root=path, train=True, download=False, transform=self.train_transform)
        self.testset = CIFAR10(root=path, train=False, download=False, transform=self.test_transform)

    def _create_model(self):
        # 定义CNN模型
        model = nn.Sequential(
            #input [3, 32, 32]
            nn.Conv2d(3, 16, 5,padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), #output [16, 16, 16]
            nn.Dropout(0.2), #4

            nn.Conv2d(16, 32, 5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2), #output [32, 8, 8]
            nn.Dropout(0.2), #9

            nn.Conv2d(32, 64, 3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #output [64, 4, 4]
            nn.Dropout(0.2), #14

            nn.Flatten(),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        return model

    def train(self, num_epochs=10, batch_size=256,lr=0.001,weight_decay=0.0001,dropout=0.2,currepoch=0,optimizer='SGD'):
        # 加载数据
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        # 训练网络
        self.net.train()
        if optimizer=='SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer=='Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('optimizer must be SGD or Adam')
        for layer in self.net.children():
            if isinstance(layer, nn.Dropout):
                layer.p = dropout
        train_loss=[]
        for epoch in range(num_epochs):
            running_loss = 0.0
            loss_list = []
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print('epoch %d step %d loss: %.3f' % (currepoch+epoch + 1, i + 1, loss.item()))
                loss_list.append(loss.item())
            train_loss.append(running_loss/len(trainloader))
        return train_loss
        # print('Finished Training')

    def test(self,visualize=False):
        # 加载测试集
        testloader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)
        # 测试网络
        self.net.eval()
        correct = 0
        total = 0
        classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        pred=[]
        GT=[]
        avg_loss=0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                pred.extend(predicted.cpu().numpy())
                GT.extend(labels.cpu().numpy())
                for i in range(10):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                loss=self.criterion(outputs,labels)
                avg_loss+=loss.item()
        acc=100 * correct / total
        acc_class=[100 * class_correct[i] / class_total[i] for i in range(10)]
        avg_loss/=len(testloader)
        print('Accuracy of the network on the %d test images: %.2lf %%' % (total,100 * correct / total))
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        print('loss: %.3f'%avg_loss)
        # 混淆矩阵
        if visualize:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import numpy as np
            cm = confusion_matrix(GT, pred)
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            for i in range(len(cm)):
                    ax.text(i, i, "%d"%(cm[i][i]/np.sum(cm[i])*100)+"%", va='center', ha='center',color='white')
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()
        return acc,acc_class,avg_loss

    def predict(self, image):
        # 对单张图像进行预测
        image = self.test_transform(image).unsqueeze(0).to(self.device)  # 添加批次维度并移动到GPU
        with torch.no_grad():
            output = self.net(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()
    
    def save_model(self, path):
        torch.save(self.net, path)
    
    def load_model(self, path): 
        self.net = torch.load(path)
        self.net.to(self.device)

if __name__=="__main__":

    model=SimpleCNN()
    model.load_data('D:/code/Datasets/')
    model.load_model('cnn_e50_d0.2.pt')
    model.train(num_epochs=10, batch_size=256,lr=0.01,weight_decay=0.001,dropout=0.2)
    model.test()
    # 保存模型
    model.save_model('cnn_e60_d0.2.pt')



