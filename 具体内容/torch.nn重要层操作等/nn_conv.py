# 开发日期  2024/7/26
import torch
import torchvision
from torch import  nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("./nn_data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.conv=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv(x)
        return x

nn_conv=OneNet()

writer=SummaryWriter("nn_conv-logs")
step=0
for data in dataloader:
    imgs,tags=data
    writer.add_images("imgs",imgs,step)
    output=nn_conv(imgs)
    #torch.Size([64, 3, 32, 32])   imgs.shape
    #torch.Size([64, 6, 30, 30])    output.shape    考虑到tensoerboard的图像绘制，需要转化为三通道处理
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step=step+1


