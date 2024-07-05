import numpy as np
import pickle

def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def load_cifar10(path):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # 读取训练集
    for i in range(1, 6):
        batch_data = load_cifar10_batch(path+f'/data_batch_{i}')
        train_images.append(batch_data[b'data'])
        train_labels += batch_data[b'labels']
    
    # 读取测试集
    test_batch = load_cifar10_batch(path+'/test_batch')
    test_images.append(test_batch[b'data'])
    test_labels += test_batch[b'labels']

    # 将列表转换为NumPy数组
    train_images = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32) / 255.0
    train_labels = np.array(train_labels)
    test_images = np.array(test_images).reshape(-1, 3, 32, 32) / 255.0
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

if __name__=='__main__':
    dataset_path = 'D:/code/datasets/cifar-10-batches-py/'
    # 加载数据集
    train_images, train_labels, test_images, test_labels = load_cifar10(path=dataset_path)

    # 打印数据集形状和标签
    print("训练集：")
    print("图像形状:", train_images.shape)
    print("标签形状:", train_labels.shape)
    print("测试集：")
    print("图像形状:", test_images.shape)
    print("标签形状:", test_labels.shape)