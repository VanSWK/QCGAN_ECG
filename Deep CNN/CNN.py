# coding=UTF8
'''
原始的卷积网络，260维度,5分类
'''
import torch
from torch import nn

class ECG_CNN(nn.Module):
    def __init__(self):
        super(ECG_CNN, self).__init__()
        self.model1 = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3),
            nn.BatchNorm1d(num_features=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 第二层卷积
            nn.Conv1d(in_channels=5, out_channels=10, kernel_size=4),
            nn.BatchNorm1d(num_features=10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 第三层卷积
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=4),
            nn.BatchNorm1d(num_features=20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # 展开
            nn.Flatten(),
            # 全连接层
            nn.Linear(30 * 20, 30),
            nn.Linear(30, 20),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        outputs = self.model1(inputs)
        return outputs


if __name__ == "__main__":
    x = torch.rand(2, 1, 260)
    print(type(x))
    cnn = ECG_CNN()
    print(cnn.model1(x))
