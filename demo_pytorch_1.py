from __future__ import print_function
import torch
#
# x = torch.empty(5, 3)
# print(x)
#
# x = torch.rand(5, 3)
# print(x)
#
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
#
# x = torch.tensor([2.5, 3.5])
# print(x)
#
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# y = torch.randn_like(x, dtype=torch.float)
# print(y)
#
# print(x.size())
# a, b = x.size()

if torch.cuda.is_available():
    x = torch.randn(1)
    print(x)
    #定义一个设备对象，这里指定成CUDA,即使用GPU
    device = torch.device("cuda")
    #直接在GPU上创建一个Tensor
    y = torch.ones_like(x, device=device)
    #将在CPU上面的x张量移动到GPU上面
    x = x.to(device)
    #x和y都在GPU上面，才支持加法运算
    z = x + y
    #此处的仗来给你z在GPU上
    print(z)
    #也可以将z转移到CPU上面，并同时指定张量元素的数据类型
    print(z.to("cpu",torch.double))

