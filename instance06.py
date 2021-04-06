import torch
import torch.nn as nn
import numpy
import pandas
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset,DataLoader
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
'''
GAN + mnist,生成手写数字图片，改良GAN，解决模式崩溃。种子实验
'''

def generate_random_seed(size):
    return  torch.randn(size)

def generate_random_image(size):
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
            #nn.Linear(200, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,1),
            nn.Sigmoid()
        )
        self.lossfunction  = nn.BCELoss()
        self.optimizer = torch.optim.Adam (self.parameters(),lr=0.0001)
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
        plt.title("D loss")
        plt.plot(range(len(self.progress)),self.progress)
        plt.show()

        pass



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100,200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,784),
            nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0001)
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
        plt.title("G loss")
        plt.plot(range(len(self.progress)), self.progress)
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
            D.trainit(generate_random_image((loader.batch_size,784)),torch.zeros(loader.batch_size,1))
            pass
    D.plot_progress()
# train_D()

def test_Gout():
    G =Generator()
    output = G(generate_random_seed(1))
    img = output.detach().numpy().reshape(28,28)
    plt.imshow(img,interpolation='none',cmap='Blues')
    plt.show()
# test_Gout()


def train_mnist_GAN():
    G = Generator()
    D = Discriminator()
    loader = DataLoader(MnistDataSet('./mnist_train.csv'), batch_size=60, shuffle=True, drop_last=True)
    epoch = 8
    for i in range(epoch):
        for batch in loader:
            label, image, target = batch
            # 真实数据训练 D
            # 使用G 的生成样本训练 D
            # 若此步重复多次。
            for k in range(3):
                D.trainit(image,torch.ones(loader.batch_size,1))  #  label 为 B*1 （B个1）
                D.trainit(G.forward(generate_random_seed((loader.batch_size,100))).detach(),torch.zeros(loader.batch_size,1))

            # 训练生成器
            G.trainit(D,generate_random_seed((loader.batch_size,100)),   torch.ones(loader.batch_size,1) )

            pass
    D.plot_progress()
    G.plot_progress()
    generate_newimage(G)
    seed_experiment_smooth(G)


def generate_newimage(G:Generator):
    f,axarr = plt.subplots(2,3,figsize=(16,8))
    plt.title("随机生成图片")
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed((1,100)))
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img,interpolation='none',cmap='Blues')
    plt.show()
    pass

def seed_experiment_smooth(G):
    count = 0
    seed1 = generate_random_seed(100)
    seed2 = generate_random_seed(100)
    plt.figure(figsize=(16,8))
    plt.title("seed1")
    out1 = G.forward(seed1)
    img1 = out1.detach().numpy().reshape(28, 28)
    plt.imshow(img1, interpolation='none', cmap='Blues')
    plt.show()

    plt.figure(figsize=(16,8))
    plt.title("seed2")
    out2 = G.forward(seed2)
    img2 = out2.detach().numpy().reshape(28, 28)
    plt.imshow(img2, interpolation='none', cmap='Blues')
    plt.show()


    f,axarr = plt.subplots(3,4,figsize=(16,8))
    plt.title("种子实验:平滑")
    for i in range(3):
        for j in range(4):
            seed = seed1 + (seed2-seed1)/11*count
            output = G.forward(seed)
            img  = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img,interpolation='none',cmap='Blues')
            count += 1
    plt.show()

    plt.figure(figsize=(16,8))
    plt.title("种子实验:相加")
    seed3 = seed1+seed2
    out3 = G.forward(seed3)
    img3 = out3.detach().numpy().reshape(28, 28)
    plt.imshow(img3, interpolation='none', cmap='Blues')
    plt.show()

    plt.figure(figsize=(16,8))
    plt.title("种子实验:相减")
    seed4 = seed1-seed2
    out4 = G.forward(seed4)
    img4 = out4.detach().numpy().reshape(28, 28)
    plt.imshow(img4, interpolation='none', cmap='Blues')
    plt.show()


import time
start =time.time()
train_mnist_GAN()
end=time.time()
print('Running time: %s Seconds'%(end-start))

