# 开发日期  2024/7/24
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


train_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=False,transform=dataset_transform,download=True)

'''
print(test_set[0])
print(test_set.classes)
img,tag=test_set[0]
print(img)
print(tag)
print(test_set.classes[tag])
'''

writer=SummaryWriter("torchvision-logs")
for i in range(10):
    img,tag=test_set[i]
    writer.add_image("test_set",img,i)

writer.close()



