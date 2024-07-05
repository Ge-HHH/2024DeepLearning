import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=32*3, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 32, 32*3)  # Reshape input to (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = nn.ReLU()(self.fc1(lstm_out))
        x = self.fc2(x)
        return x
class SimpleLSTM:
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
        return LSTMClassifier()

    def train(self, num_epochs=10, batch_size=256, lr=0.001, weight_decay=0.0001, dropout=0.2, currepoch=0):
        # 加载数据
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        # 训练网络
        self.net.train()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        train_loss = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print('epoch %d step %d loss: %.3f' % (currepoch + epoch + 1, i + 1, loss.item()))
            train_loss.append(running_loss / len(trainloader))
        return train_loss

    def test(self, visualize=False):
        # 加载测试集
        testloader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=2)
        # 测试网络
        self.net.eval()
        correct = 0
        total = 0
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        pred = []
        GT = []
        avg_loss = 0
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
                loss = self.criterion(outputs, labels)
                avg_loss += loss.item()
        acc = 100 * correct / total
        acc_class = [100 * class_correct[i] / class_total[i] for i in range(10)]
        avg_loss /= len(testloader)
        print('Accuracy of the network on the %d test images: %.2lf %%' % (total, 100 * correct / total))
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        print('loss: %.3f' % avg_loss)
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
        return acc, acc_class, avg_loss

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

    model=SimpleLSTM()
    model.load_data('D:/code/Datasets/')
    # model.load_model('cnn_e50_d0.2.pt')
    model.train(num_epochs=5, batch_size=256,lr=0.01,weight_decay=0.001,dropout=0.2)
    model.test()
    # 保存模型
    model.save_model('lstm_e30.pt')