# 开发日期  2024/7/28
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.vgg import VGG16_Weights

'''
vgg16_false=torchvision.models.vgg16(pretrained=False)
vgg16_true=torchvision.models.vgg16(pretrained=True)
'''
vgg16_false = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print(vgg16_false)
print("\n")
print(vgg16_true)

dataset=torchvision.datasets.CIFAR10("./nn_data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=1,drop_last=True)

# vgg16_true.add_module('add_linear',nn.Linear(1000,10))

vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
vgg16_false.classifier[6]=nn.Linear(4096,10)

print(vgg16_false)
print("\n")
print(vgg16_true)

#del vgg16_true.features[29]删除