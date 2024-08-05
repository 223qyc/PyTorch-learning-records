# 开发日期  2024/7/26
import  torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,0.5],
                    [-1,3]])

input=torch.reshape(input,(-1,1,2,2))
print(input.shape)

class OneNet(nn.Module):
    def __init__(self):
        super(OneNet,self).__init__()
        self.relu=ReLU()
        self.sigmoid=Sigmoid()

    def forward(self,input):
        output=self.sigmoid(input)
        return output


nn_sigmoid=OneNet()
output=nn_sigmoid(input)
print(output)

dataset=torchvision.datasets.CIFAR10("./nn_data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)
step=1
writer=SummaryWriter("nn_sigmoid-logs")
for data in dataloader:
    imgs,tags=data
    writer.add_images("imgs",imgs,step)
    output=nn_sigmoid(imgs)
    writer.add_images("nn_relu",output,step)
    step=step+1

