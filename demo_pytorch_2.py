# 导入若干工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 定义一个简单的网络类
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积神经网络, 输入通道维度=1, 输出通道维度=6, 卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 定义第二层卷积神经网络, 输入通道维度=6, 输出通道维度=16, 卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 定义三层全连接网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在(2, 2)的池化窗口下执行最大池化操作,卷积层必须经过激活层和池化层
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 计算size, 除了第0个维度上的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 模型中所有的可训练参数都可以通过net.parameters()来获得
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 假设图像的输入尺寸为32 * 32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
print(out.size())

# 有了输出张量后, 就可以执行梯度归零和反向传播的操作了
# net.zero_grad()
# out.backward(torch.randn(1, 10))


# 应用nn.MSELoss计算损失的一个例子:
output = net(input)
target = torch.randn(10)

# 改变target的形状为二维张量, 为了和output匹配
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# Pytorch中执行梯度清零的代码
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

# Pytorch中执行反向传播的代码
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 用传统的Python代码来实现SGD如下:
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# Pytorch官方推荐的标准代码如下
# 通过optim创建优化器对象
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 将优化器执行梯度清零的操作
optimizer.zero_grad()

output = net(input)
loss = criterion(output, target)

# 对损失值执行反向传播的操作
loss.backward()
# 参数的更新通过一行标准代码来执行
optimizer.step()