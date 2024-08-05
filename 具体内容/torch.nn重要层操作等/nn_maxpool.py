# 开发日期  2024/7/26
import torch
import torchvision
from torch import  nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])   #,dtype=torch.float32

input=torch.reshape(input,(-1,1,5,5))
print(input.shape)

class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output=self.maxpool(input)
        return output

nn_maxpool=OneNet()
writer=SummaryWriter("nn_maxpool-logs")
step=0

dataset=torchvision.datasets.CIFAR10("./nn_data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

for data in dataloader:
    imgs,tags=data
    writer.add_images("imgs",imgs,step)
    output=nn_maxpool(imgs)
    writer.add_images("output",output,step)
    step=step+1

'''
testnet=OneNet()
output=testnet(input)
print(output)
'''