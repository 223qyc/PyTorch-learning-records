# 开发日期  2024/7/24
import  torchvision
from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=False,transform=torchvision.transforms.ToTensor())
test_lodar1=DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
test_lodar2=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
img,tag=test_data[0]                      #要注意函数中各个参数的意义
print(tag)

for data in test_lodar1:
    imgs,tags=data
    print(imgs.shape)
    print(tags)


writer=SummaryWriter("datalodar-logs")
step=0
for data in test_lodar2:
    imgs,tags=data
    writer.add_images("test_data",imgs,step)  #注意是images
    step=step+1

for epoch in range(2):
    step = 0
    for data in test_lodar2:
        imgs, tags = data
        writer.add_images("Epoch__{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
