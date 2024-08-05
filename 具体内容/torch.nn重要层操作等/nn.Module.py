# 开发日期  2024/7/24

import torch
from torch import  nn

class OneNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+2
        return output

onenet=OneNet()
x=torch.tensor([1,2,3])
output=onenet(x)
print(output)