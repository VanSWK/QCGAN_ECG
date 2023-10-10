# 两个周期，256维，D、G是量子，Strong纠缠门，K步D，1步G
# 心跳数据集换为0.9s的一个心拍
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


# 加载ECG数据集
class ECGDataset(Dataset):
    '''
    lable:过滤的标签
    N:载入的数据量
    '''

    def __init__(self, csv_file_data, csv_file_label, mark=0, N=100, transform=None):
        self.csv_file_data = csv_file_data
        self.csv_file_label = csv_file_label
        self.transform = transform

        data = np.load(csv_file_data)
        label = np.load(csv_file_label)
        print("total_data_size:", data.shape)
        print("total_label_size:", data.shape)

        filter_data = []
        for i in range(len(label)):
            if len(filter_data) < N and label[i] == mark:
                filter_data.append(data[i])
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
NUM = 100  # 抽取的总数量
LABEL = 0
n_qubits_D = 6

# transform = transforms.Compose([transforms.ToTensor()])
dataset = ECGDataset(csv_file_data="D:\Pycharm_Projects\DATA\data_64_0.9s\mit-bih-arrhythmia-database-1.0.0_Data.npy",
                     csv_file_label="D:\Pycharm_Projects\DATA\data_64_0.9s\mit-bih-arrhythmia-database-1.0.0_Label.npy",
                     mark=LABEL, N=NUM, transform=None)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

print(len(dataset[0]))
plt.figure(figsize=(10, 1))
print(dataset[0].size)
for i in range(5):
    ecg = dataset[i]
    plt.subplot(1, 5, i + 1)
    plt.axis('off')
    plt.plot(ecg)
plt.show()

# 判别器电路
devD = qml.device("default.qubit", wires=n_qubits_D)


@qml.qnode(devD, interface="torch", diff_method="parameter-shift")
def quantum_circuit_D(inputs, weights):
    # 使用振幅编码
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits_D), normalize=True)

    # entangle layer
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits_D))

    return qml.expval(qml.PauliZ(wires=0))  # 返回第一个qubit的期望


# # 画出判别器电路图
# inputs = np.random.uniform(0, np.pi / 2, size=ecg_size)
# shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits_D)
# weights = np.random.uniform(0, np.pi / 2, size=shape)
# fig, ax = qml.draw_mpl(quantum_circuit_D, decimals=2, expansion_strategy='device')(inputs, weights)
# plt.show()


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits_D)
        weight_shapes = {"weights": shape}
        self.qfc = qml.qnn.TorchLayer(quantum_circuit_D, weight_shapes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.qfc(x)
        outputs = self.sigmoid(x)
        return outputs


# Quantum variables
n_qubits = 5  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 4  # Number of subgenerators for the patch method / N_G

# Quantum simulator
devG = qml.device("default.qubit", wires=n_qubits)
# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 画出生成器电路图

# strongentanle 纠缠
@qml.qnode(devG, interface="torch", diff_method="parameter-shift")
def quantum_circuit_G(noise, weights):
    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))

    return qml.probs(wires=list(range(n_qubits)))  # 每个基态的概率


# noise = np.random.uniform(0, np.pi / 2, size=n_qubits)
# shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_qubits)
# weights = np.random.uniform(0, np.pi / 2, size=shape)
# fig, ax = qml.draw_mpl(quantum_circuit_G, decimals=2, expansion_strategy='device')(noise, weights)
# plt.show()


def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit_G(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=n_qubits)
        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(self.shape[0] * self.shape[1] * self.shape[2]), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        ecgs = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:
            params = params.reshape(self.shape)  # 参数需要和StrongEntgangle层形状一致
            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)  # 每次输出为16
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            ecgs = torch.cat((ecgs, patches), 1)
        return ecgs


lrG = 0.1  # Learning rate for the generator
lrD = 0.1  # Learning rate for the discriminator

discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators).to(device)

# Binary cross entropy
criterion = nn.BCELoss()

# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)

# Fixed noise allows us to visually track the generated images throughout training
fixed_noise = torch.rand(5, n_qubits, device=device) * math.pi / 2  # 0-pi/2之间的随机噪声

# Iteration counter
counter_D = 0  # 记录判别器训练迭代数

# Collect images for plotting later
gen_ecgs = []
lossD = []
lossG = []

# 判别器训练迭代数
K = 1

EPOCH = 30

for epoch in range(EPOCH):
    # 先训练K次判别器
    for i, data in enumerate(dataloader):
        # Data for training the discriminator
        real_data = data.to(device)

        # Noise follwing a uniform distribution in range [0,pi/2)
        noise = torch.rand(batch_size, n_qubits, device=device) * math.pi / 2
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
        if counter_D % K == 0:
            generator.zero_grad()
            outD_fake = discriminator(fake_data).view(-1)
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            lossD.append(errD.detach().numpy())
            lossG.append(errG.detach().numpy())
            print(f'Iteration: {counter_D}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

            fileName = 'results03/2/log.txt'
            with open(fileName, 'a') as file:
                file.write(f'Iteration: {counter_D}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}\n')

    # 每个epoch打印一次loss
    print(f'epoch: {epoch + 1}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
    fileName = 'results03/2/log.txt'
    with open(fileName, 'a') as file:
        file.write(f'epoch: {epoch + 1}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}\n')

    lossD.append(errD.detach().numpy())
    lossG.append(errG.detach().numpy())

    # 每个epoch查看一次生成图
    test_ecgs = fake_data.cpu().detach()
    gen_ecgs.append(test_ecgs)

loss = np.concatenate([lossD, lossG], axis=0)
np.savetxt('./results02/2/loss.csv', loss, delimiter=',')

plt.plot(lossD)
plt.plot(lossG)
# plt.show()
plt.savefig('./results03/2/loss.jpg', dpi=600, bbox_inches='tight')

plt.figure(figsize=(10, EPOCH))
for i in range(EPOCH):
    for j in range(5):
        ecg = gen_ecgs[i][j]
        plt.subplot(EPOCH, 5, i * 5 + j + 1)
        plt.axis('off')
        plt.plot(ecg)
# plt.show()
plt.savefig('./results03/2/ecgs.jpg', dpi=600, bbox_inches='tight')
