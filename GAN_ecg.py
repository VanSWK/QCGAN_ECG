# 两个周期，64维，D、G是量子，Strong纠缠门，K步D，1步G
import math
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Set the random seed for reproducibility
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载ECG数据集
class ECGDataset(Dataset):
    '''
    lable:过滤的标签
    N:载入的数据量
    '''

    def __init__(self, csv_file, label=1, N=100, transform=None):
        self.csv_file = csv_file
        self.transform = transform

        assert os.path.isfile(csv_file)
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
        print("total_data_size:", data.shape)

        filter_data = []
        for i in range(len(data)):
            if len(filter_data) < N and data[i][-1] == label:
                filter_data.append(data[i][0:64])
                # print(i)
        filter_data = torch.tensor(filter_data, dtype=torch.float32)
        print("filter_data_size:", filter_data.shape)

        # 数据归一pi*（0，1）
        nomal_data = (np.pi / 2) * (
                (filter_data - torch.min(filter_data)) / (torch.max(filter_data) - torch.min(filter_data)))  # 最值归一化
        self.data = nomal_data
        print(f'normal_data:', nomal_data.shape)
        print("max_normal_data", torch.max(nomal_data))
        print("min_normal_data", torch.min(nomal_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx, :]

        if self.transform:
            item = self.transform(self.data[idx, :])

        return item


ecg_size = 64  # length of the ecgs
batch_size = 5
NUM = 10000  # 抽取的总数量
LABEL = 1

# transform = transforms.Compose([transforms.ToTensor()])
dataset = ECGDataset(csv_file="./DATA/data_64_2T/ar_MLII_5.csv", label=LABEL, N=NUM, transform=None)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

print(len(dataset[0]))
print(dataset[0].size)


# plt.figure(figsize=(10, 1))
# for i in range(5):
#     ecg = dataset[i]
#     plt.subplot(1, 5, i + 1)
#     plt.axis('off')
#     plt.plot(ecg)
# plt.show()

# 判别器
class Discriminator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(ecg_size, 128),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# 生成器
class Generator(nn.Module):
    """Fully connected classical discriminator"""

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(16, 32),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(128, 64),
        )

    def forward(self, x):
        return self.model(x)


lrG = 0.001  # Learning rate for the generator
lrD = 0.001  # Learning rate for the discriminator

discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(5, 16, device=device) * math.pi / 2  # 0-pi/2之间的随机噪声

# Iteration counter
counter_D = 0  # 记录判别器训练迭代数

# Collect images for plotting later
gen_ecgs = []
lossD = []
lossG = []

# 判别器训练迭代数
K = 10

EPOCH = 1000

for epoch in range(EPOCH):
    # 先训练K次判别器
    for i, data in enumerate(dataloader):
        # Data for training the discriminator
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        # noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
        fake_data = generator(fixed_noise)
        # print(f"fake_data:{fake_data}")

        # Training the discriminator
        discriminator.zero_grad()
        outD_real = discriminator(real_data).view(-1)
        outD_fake = discriminator(fake_data.detach()).view(-1)

        errD_real = criterion(outD_real, real_labels)
        errD_fake = criterion(outD_fake, fake_labels)
        # Propagate gradients
        errD_real.backward()
        errD_fake.backward()

        errD = errD_real + errD_fake
        optD.step()
        counter_D += 1

        # 训练K次判别器后，训练一次生成器
        if counter_D % K == 5:
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            lossD.append(errD.detach().numpy())
            lossG.append(errG.detach().numpy())
            print(f'Iteration: {counter_D}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            fileName = 'results02/1/log1.txt'
            with open(fileName, 'a') as file:
                file.write(f'Iteration: {counter_D}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}\n')

    # 每个epoch打印一次loss
    print(f'epoch: {epoch + 1}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
    fileName = 'results02/1/log1.txt'
    with open(fileName, 'a') as file:
        file.write(f'epoch: {epoch + 1}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}\n')

    lossD.append(errD.detach().numpy())
    lossG.append(errG.detach().numpy())

    # 每个epoch查看一次生成图
    test_ecgs = fake_data.cpu().detach()
    gen_ecgs.append(test_ecgs)

loss = np.concatenate([lossD, lossG], axis=0)
np.savetxt('./results03/loss.csv', loss, delimiter=',')

plt.plot(lossD)
plt.plot(lossG)
plt.show()
plt.savefig('./results03/loss.jpg', dpi=600, bbox_inches='tight')

plt.figure(figsize=(10, EPOCH))
for i in range(EPOCH):
    for j in range(5):
        ecg = gen_ecgs[i][j]
        plt.subplot(EPOCH, 5, i * 5 + j + 1)
        plt.axis('off')
        plt.plot(ecg)
# plt.show()
plt.savefig('./results03/ecgs.jpg', dpi=600, bbox_inches='tight')
