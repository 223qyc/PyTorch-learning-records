# 开发日期  2024/8/2
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import  time

# from model import *

#1.准备数据集
train_data=torchvision.datasets.CIFAR10("../The complete process&&GPU acceleration/data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("../The complete process&&GPU acceleration/data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#可以检验一下数据集的长度
train_data_size=len(train_data)
print("训练数据集的长度为：{}".format(train_data_size))
test_data_size=len(test_data)
print("测试数据集的长度为：{}".format(test_data_size))

#定义训练的设备
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
# device=torch.device("cuda")

#2.使用dataloader来加载数据集
train_dataloader=DataLoader(train_data,batch_size=64,drop_last=True)
test_dataloader=DataLoader(test_data,batch_size=64,drop_last=True)

#3.搭建神经网络
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

#4.创建网络模型
mynet=MyNet()
if torch.cuda.is_available():
    mynet.to(device)


#5.损失函数与优化器
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn.to(device)
learning_rate=1e-2
optimizer=torch.optim.SGD(mynet.parameters(),lr=learning_rate)

#6.设置相关训练参数
total_train_step=0
total_test_step=0
total_accuracy=0
epoch=30

#7.添加tensorboard
writer=SummaryWriter("../The complete process&&GPU acceleration/mynet-logs")


#8.训练并测试
start_time=time.time()  #开始计时
for i in range(epoch):
    print("------------------第{}轮训练开始------------".format(i+1))

    mynet.train()

    for data in train_dataloader:
        imgs,tags=data
        if torch.cuda.is_available():
            imgs=imgs.to(device)
            tags=tags.to(device)
        output=mynet(imgs)
        loss=loss_fn(output,tags)

        #优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print("训练次数：{},损失：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    #测试步骤开始
    mynet.eval()

    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,tags=data
            if torch.cuda.is_available():
                imgs = imgs.to(device)
                tags = tags.to(device)
            output=mynet(imgs)
            loss=loss_fn(output,tags)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(output.argmax(1)==tags).sum()
            total_accuracy=accuracy+total_accuracy

    print("整体测试集上loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step=total_test_step+1

    torch.save(mynet,"../The complete process&&GPU acceleration/model_save/mynet_{}.pth".format(i))
    # torch.save(mynet.state_dict(),"../The complete process&&GPU acceleration/model_save/mynet_{}.pth".format(i))
    print("模型已保存")


writer.close()