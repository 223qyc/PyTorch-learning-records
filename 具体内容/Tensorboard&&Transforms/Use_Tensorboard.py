# 开发日期  2024/7/22
#是一个数据可视化的工具
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import  Image


writer=SummaryWriter("Tensorboard-logs")   #括号中的是要画图的文件夹所在的目录的名字
for i in range(100):
    writer.add_scalar("y=x^2-2x+1",i*i-2*i+1,i)#标题，纵坐标，横坐标
writer.close()

img_path="data/train/ants/6240338_93729615ec.jpg"
img_PIL=Image.open(img_path)
img_Array=np.array(img_PIL)
print(type(img_Array))
print(img_Array.shape)
writer.add_image("test1",img_Array,1,dataformats="HWC")
writer.close()

#通过tensorboard --logdir=(所在文件夹名)打开浏览器看到图像
#也可以通过修改端口的办法 tensorboard --logdir=(所在文件夹名) --port=指定端口号
#要想办法避免同一图像上多次拟合，同一标题下step不一样可以来回切换
