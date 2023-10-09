import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
from torch.utils.data import Dataset

from CNN import ECG_CNN

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 超参数
EPOCH = 1  # 前向后向传播迭代次数
LR = 0.001  # 学习率 learning rate
BATCH_SIZE = 50  # 批量训练时候一次送入数据的size
DOWNLOAD_MNIST = True


class MyECGDataset(Dataset):

    def __init__(self, Data_path, Label_path):
        data = np.load(Data_path)  # 读取数据和标签
        label = np.load(Label_path)
        self.Data = data
        self.Label = label
        # print(self.Data.shape)
        # print(self.Label.shape)

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, idx):
        x = self.Data[idx]
        y = self.Label[idx]

        return x, y

    def getData(self):
        return self.Data

    def getLable(self):
        return self.Label


ECG_DATA = MyECGDataset(Data_path="../Data/ECG_signals/Data.npy", Label_path="../Data/ECG_signals/Label.npy")
train_dataset, test_dataset = Data.random_split(dataset=ECG_DATA, lengths=[100000, 687])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
test_x, test_y = test_dataset[:]


cnn = ECG_CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 定义优化器
loss_func = nn.CrossEntropyLoss()  # 定义损失函数

for epoch in range(EPOCH):

    for step, (batch_x, batch_y) in enumerate(train_loader):
        pred_y = cnn(batch_x)
        print(type(pred_y))
        loss = loss_func(pred_y, batch_y)
        optimizer.zero_grad()  # 清空上一层梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新优化器的学习率，一般按照epoch为单位进行更新

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
            # 返回的形式为torch.return_types.max(
            #           values=tensor([0.7000, 0.9000]),
            #           indices=tensor([2, 2]))
            # 后面的[1]代表获取indices
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

# 打印前十个测试结果和真实结果进行对比
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
