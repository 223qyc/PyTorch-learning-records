# 开发日期  2024/7/16
from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):


    def __init__(self,root_dir,lable_dir):
        self.root_dir=root_dir       #存储根目录的路径
        self.label_dir=lable_dir     #存储根目录下标签目录的路径
        self.path=os.path.join(self.root_dir,self.label_dir)   #将路径拼接
        self.img_path=os.listdir(self.path)     #读取该路径下所有的内容，并将内容存储在列表中


    #获得数据集中某个项目内容
    def __getitem__(self, idx):
        img_name=self.img_path[idx]  #从上述定义的函数中获取图片名称
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)   #获取该项目的完整相对路径
        img=Image.open(img_item_path)     #打开
        label=self.label_dir       #记录标签
        return img,label

    def __len__(self):
        return len(self.img_path)


root_dir= "data/train"
ants_label_dir="ants"
bees_lable_dir="bees"
ants_dataset=Mydata(root_dir,ants_label_dir)
bees_dateset=Mydata(root_dir,bees_lable_dir)
train_dateset=ants_dataset+bees_dateset

img,label=train_dateset[126]
print(len(train_dateset))
print(label)
img.show()
