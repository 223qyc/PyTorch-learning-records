# 开发日期  2024/7/28
import torch
import  torchvision
from torch import nn
from torch.nn import Conv2d
from torchvision.models.vgg import VGG16_Weights

#1.对应第一个保存路径
model1=torch.load("./net/vgg16_method1.pth")
print(model1)

#2.
model2=torch.load("./net/vgg16_method2.pth")
print(model2)

#第二种方式如何展示完整的网络

model3=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
model3.load_state_dict(torch.load("./net/vgg16_method2.pth"))
print(model3)


#采用第一种方法可能出现的问题
class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.conv=Conv2d(3,6,3)

    def forward(self,x):
        x=self.conv(x)
        return x

model4=torch.load("./net/test_net.pth")
print(model4)

