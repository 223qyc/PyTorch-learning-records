# 开发日期  2024/7/26
import torch
import  torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("./nn_data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64,drop_last=True)

class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.linear=Linear(196608,10)
    def forward(self,input):
        output=self.linear(input)
        return output

nn_linear=OneNet()


for data in dataloader:
    imgs,tags=data
    print(imgs.shape)  #torch.Size([64, 3, 32, 32])
   # output=torch.reshape(imgs,(1,1,1,-1)) 直接用flatten替代
    output=torch.flatten(imgs)
    print(output.shape)  #torch.Size([1, 1, 1, 196608])
    output=nn_linear(output)
    print(output.shape)        #torch.Size([1, 1, 1, 10])
'''
torch.Size([64, 3, 32, 32])
torch.Size([196608])
torch.Size([10])
'''