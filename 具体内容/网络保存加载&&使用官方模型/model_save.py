# 开发日期  2024/7/28
import torch
import  torchvision
from torch import nn
from torch.nn import Conv2d
from torchvision.models.vgg import VGG16_Weights
vgg16=torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

#1.模型结构+参数
torch.save(vgg16,"./net/vgg16_method1.pth")

#2.模型参数
torch.save(vgg16.state_dict(),"./net/vgg16_method2.pth")

#自定义可能带来的问题
class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.conv=Conv2d(3,6,3)

    def forward(self,x):
        x=self.conv(x)
        return x

test_net=OneNet()
torch.save(test_net,"./net/test_net.pth")
