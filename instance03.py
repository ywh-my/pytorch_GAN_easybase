import torch
import torch.nn as nn
import numpy
import pandas
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset,DataLoader

'''

GAN + mnist,生成手写数字图片.会发生mode collapse哦
'''

def generate_random(size):
    return  torch.rand(size)
class MnistDataSet(Dataset):
    def __init__(self,csv_file):
        self.data_df = pandas.read_csv(csv_file,header=None)
        pass
    def __getitem__(self,index):
        # 目标图像label
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0 # 1*10的one hot

        #图像数据，取值在0-255，标准化为0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values)/255.0

        #返回标签、图像
        return label,image_values,target

        pass
    def __len__(self):
        return len(self.data_df)


    def plot_image(self,index):
        arr = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = "+str(self.data_df.iloc[index,0]))
        plt.imshow(arr,interpolation='none',cmap='Blues')
        plt.show()
        pass

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.Sigmoid(),
            nn.Linear(200,1),
            nn.Sigmoid()
        )
        self.lossfunction  = nn.MSELoss()
        self.optimizer = torch.optim.Adam (self.parameters(),lr=0.01)
        self.counter = 0
        self.progress = []


    def forward(self,x):
        return self.model(x)

    def trainit(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.lossfunction(outputs, targets)

        # 每训练10次增加一次计数
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            pass
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
            pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass

    def plot_progress(self):
        plt.figure(figsize=(16,8))
        plt.ylim([0,1])
        plt.plot(range(len(self.progress)),self.progress)
        plt.show()

        pass



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1,200),
            nn.Sigmoid(),
            nn.Linear(200,784),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
        self.counter = 0
        self.progress = []


    def forward(self,x):
        return self.model(x)

    def trainit(self, D:Discriminator,inputs, targets):
        #计算生成器输出
        g_output = self.forward(inputs)
        #输入鉴别器
        d_outpu = D.forward(g_output)
        loss = D.lossfunction(d_outpu,targets)

        # 每训练10次增加一次计数
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            pass
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
            pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        pass
    def plot_progress(self):
        plt.figure(figsize=(16,8))
        plt.ylim([0,1])
        plt.plot(range(len(self.progress)),self.progress)
        plt.show()

        pass
# 尝试训练D，令其分辨真实图像
def train_D():
    D = Discriminator()
    loader = DataLoader(MnistDataSet('./mnist_train.csv'), batch_size=60, shuffle=True, drop_last=True)
    epoch = 2
    for i in range(epoch):
        for batch in loader:
            label, image, target = batch
            # 真实数据
            D.trainit(image,torch.ones(loader.batch_size,1))
            # 随机噪声
            D.trainit(generate_random((loader.batch_size,784)),torch.zeros(loader.batch_size,1))
            pass
    D.plot_progress()
# train_D()

def test_Gout():
    G =Generator()
    output = G(generate_random(1))
    img = output.detach().numpy().reshape(28,28)
    plt.imshow(img,interpolation='none',cmap='Blues')
    plt.show()
# test_Gout()


def train_mnist_GAN():
    G = Generator()
    D = Discriminator()
    loader = DataLoader(MnistDataSet('./mnist_train.csv'), batch_size=60, shuffle=True, drop_last=True)
    epoch = 3
    for i in range(epoch):
        for batch in loader:
            label, image, target = batch
            # 真实数据训练 D
            D.trainit(image,torch.ones(loader.batch_size,1))  #  label 为 B*1 （B个1）
            # 使用G 的生成样本训练 D
            D.trainit(G.forward(generate_random((loader.batch_size,1))),torch.zeros(loader.batch_size,1))
            # 训练生成器
            G.trainit(D,generate_random((loader.batch_size,1)),   torch.ones(loader.batch_size,1) )

            pass
    D.plot_progress()
    G.plot_progress()
    generate_newimage(G)

def generate_newimage(G:Generator):
    f,axarr = plt.subplots(2,3,figsize=(16,8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random(1))
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img,interpolation='none',cmap='Blues')
    plt.show()
    pass

import time
start =time.time()
train_mnist_GAN()
end=time.time()
print('Running time: %s Seconds'%(end-start))

