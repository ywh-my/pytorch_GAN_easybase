import torch
import torch.nn as nn
import numpy
import pandas
import matplotlib.pyplot as plt
import random


'''
GAN生成1010格式规律
D：能够区别符合1010规律的数字，符合为1，不符合为0。
G：能够生成1010规律的数字。
使用BCEloss，理论值为0.69
'''
def generate_real():
    return  torch.FloatTensor([
        random.uniform(0.8,1.0),
        random.uniform(0.0,0.2),
        random.uniform(0.8, 1.0),
        random.uniform(0.0,0.2),
    ])

def generate_random(size):
    return  torch.rand(size)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4,3),
            nn.Sigmoid(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )
        self.lossfunction  = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.01)
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
        plt.plot(range(len(self.progress)),self.progress)
        plt.show()

        pass

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1,3),
            nn.Sigmoid(),
            nn.Linear(3,4),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.01)
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
        plt.plot(range(len(self.progress)),self.progress)
        plt.show()

        pass

def train_D():
    D = Discriminator()
    for i in range(10000):
        #真实数据
        D.trainit(generate_real(),torch.FloatTensor([1.0])   )
        # 噪声数据
        D.trainit(generate_random(4), torch.FloatTensor([0.0])   )
        pass
    D.plot_progress()

def train_thisGAN():

    D = Discriminator()
    G = Generator()
    imagelist = []

    for i in range(10000):
        # 记录生成器的变化
        if i % 1000 == 0:
            imagelist.append(G.forward(torch.FloatTensor([random.uniform(0.19,0.21)])).detach().numpy())

        # 用真实样本训练 D：
        D.trainit(generate_real(),torch.FloatTensor([1.0]))

        # 用生成器的生成样本训练鉴别器 ,随机一个数字给生成器，生成四个数字。label为 0
        D.trainit(G.forward(torch.FloatTensor([random.uniform(0.19,0.21)]).detach()),torch.FloatTensor([0.0]))
        # 数据的流动是从 G->D->LOSS,这里的.detach()能够切断从D->G的梯度传播。使得网络仅计算 D的梯度，不计算G的。这样节省资源。


        #训练生成器
        G.trainit(D,torch.FloatTensor([random.uniform(0.19,0.21)]), torch.FloatTensor([1.0]))

    D.plot_progress()
    G.plot_progress()
    plt.figure(figsize=(16,8))
    plt.imshow(numpy.array(imagelist).T,interpolation='none',cmap='Blues')
    plt.show()
def train_thisGAN2():

    D = Discriminator()
    G = Generator()
    for i in range(50000):
        # 用真实样本训练 D：
        D.trainit(generate_real(), torch.FloatTensor([1.0]))
        # 用生成器的生成样本训练鉴别器 ,随机一个数字给生成器，生成四个数字。label为 0
        D.trainit(G.forward(torch.FloatTensor([random.uniform(0.5)]).detach()), torch.FloatTensor([0.0]))

        # 训练生成器
        G.trainit(D, torch.FloatTensor([random.uniform(0.5)]), torch.FloatTensor([1.0]))

    D.plot_progress()
    G.plot_progress()

import time
start =time.time()
train_thisGAN()
end=time.time()
print('Running time: %s Seconds'%(end-start))