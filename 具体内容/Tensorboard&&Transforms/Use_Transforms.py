# 开发日期  2024/7/23
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#总的来说，transforms像是一个大的package，包含了很多对图片处理的方法，包括格式转换，重塑大小等
from PIL import  Image
import cv2


#tensor的数据类型
#通过transforms.ToTensor

img_path="data/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img=Image.open(img_path)
print(img)

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
print(tensor_img)

cv_img=cv2.imread(img_path)
print(cv_img)   #nunmpy打开形式

writer=SummaryWriter("Transforms-logs")
writer.add_image("Tensor picture",tensor_img)
writer.close()

