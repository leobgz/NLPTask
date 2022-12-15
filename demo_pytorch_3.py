import torch
import torchvision
import torchvision.transforms as transforms

# 导入画图包和numpy
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# 训练集格式修改
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 数据封装成数据迭代器（每批四条数据，shuffle为true表示数据打乱，num_workers为2个线程）
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 构建展示图片的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # 从数据迭代器中读取一张图片
# # dataiter = iter(trainloader)
# # images, labels = dataiter.next()
#
# # 展示图片
# imshow(torchvision.utils.make_grid(images))
# # 打印标签label
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data中包含输入图像张量inputs, 标签张量labels
        inputs, labels = data

        # 首先将优化器梯度归零
        optimizer.zero_grad()

        # 输入图像张量进网络, 得到输出张量outputs
        outputs = net(inputs)

        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)

        # 反向传播+参数更新, 是标准代码的标准流程
        loss.backward()
        optimizer.step()

        # 打印轮次和损失值
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# 首先设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)