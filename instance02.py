import torch
import torch.nn as nn
import numpy
import pandas
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset,DataLoader

'''
mnist_数据集。
'''

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,10),
            nn.Softmax(dim=0)
        )
        self.lossfunction  = nn.MSELoss()
        self.optimizer = torch.optim.Adam (self.parameters(),lr=0.001)
        self.counter = 0
        self.progress = []


    def forward(self,x):
        return self.model(x)
    def trainit(self,inputs,targets):
        outputs = self.forward(inputs)
        loss = self.lossfunction(outputs,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每训练10次增加一次计数
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
            pass
        if self.counter % 10000 == 0:
            print("counter = ", self.counter)
            pass

    def plot_progress(self):
        plt.plot(range(len(self.progress)), self.progress)
        plt.show()

        pass
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
def look_a_image(index):
    f = './mnist_train.csv'
    mnist_d = MnistDataSet(f)
    mnist_d.plot_image(index)

def train_test_Mnist():
    C = Classifier()
    loader = DataLoader(MnistDataSet('./mnist_train.csv'),batch_size=60,shuffle=True,drop_last=True)
    epoch = 5
    for i in range(epoch):
        for batch in loader:
            label,image,target = batch
            C.trainit(image,target)
            pass
        pass
    C.plot_progress()
    print("训练完成")
    test_Mnist(C)


def test_Mnist(C:Classifier):
    score = 0
    items = 0
    loader = DataLoader(MnistDataSet('./mnist_test.csv'),batch_size=5,shuffle=True,drop_last=True)
    avg_acc=  []
    for batch in loader:
        label, image, target = batch
        answer = C.forward(image)
        v, f = torch.max(answer, 1)
        score += (f==label).sum()
        items += loader.batch_size
        avg_acc.append(score/items)
    print("测试数：{},分数：{},最终准确率：{}".format(items,score,score/items))
    plt.plot(range(len(avg_acc)), avg_acc)
    plt.show()

import time
start =time.time()
train_test_Mnist()
end=time.time()
print('Running time: %s Seconds'%(end-start))
#