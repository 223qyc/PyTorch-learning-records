# 开发日期  2024/7/31
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#1.准备数据集
train_data=torchvision.datasets.CIFAR10("../The complete process&&GPU acceleration/data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10("../The complete process&&GPU acceleration/data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#可以检验一下数据集的长度
train_data_size=len(train_data)
print("训练数据集的长度为：{}".format(train_data_size))
test_data_size=len(test_data)
print("测试数据集的长度为：{}".format(test_data_size))

#2.使用dataloader来加载数据集
train_dataloader=DataLoader(train_data,batch_size=64,drop_last=True)
test_dataloader=DataLoader(test_data,batch_size=64,drop_last=True)

#3.搭建神经网络，在model.py中完成

#4.创建网络模型
mynet=MyNet()


#5.损失函数与优化器
loss_fn=nn.CrossEntropyLoss()

learning_rate=1e-2
optimizer=torch.optim.SGD(mynet.parameters(),lr=learning_rate)

#6.设置相关训练参数
total_train_step=0
total_test_step=0
total_accuracy=0
epoch=30

#7.添加tensorboard
writer=SummaryWriter("../The complete process&&GPU acceleration/mynet-logs")


start_time=time.time()
#8.训练并测试
for i in range(epoch):
    print("------------------第{}轮训练开始------------".format(i+1))

    mynet.train()

    for data in train_dataloader:
        imgs,tags=data
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
