# 开发日期  2024/7/23
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path="Image/2019060319562153553.jpg"
img=Image.open(img_path)
print(img)

#Tosenor
writer=SummaryWriter("Transforms-logs")
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("Totensor",img_tensor)

#Normalize 归一化
trans_norm=transforms.Normalize([0.5,0.5,0.5],[1,1,1])
img_norm=trans_norm(img_tensor)
print(img_tensor)
print(img_norm)
writer.add_image("Normalize",img_norm,4)

#Resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)    #注意，处理对象是PIL，是变换之前的(但是现版本的reszie可以直接处理Tensor数据)
print(img_resize.size)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize)

#Compose
#PIL->PIL->Tensor
trans_resize_2=transforms.Resize((1000,700))
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop
trans_random=transforms.RandomCrop(700)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop2",img_crop,i)
writer.close()

'''
总结：查阅函数使用方法可以直接查看官方文档
其次是了解参数，要看清楚什么参数可以被接收，比如上述resize在旧版本中仅支持PIL但现在Tensor也是支持的
要了解compose是一个指令执行合集，初始参数的传入要正确

基本所有方法都以一个类定义
使用的是自己根据官方文档类所定义的对象，而不是直接使用官方文档的函数

尤其在以后要注意，图像处理的像素参数是一个数还是两个数，不同的处理规则是怎样的，要根据官方文档和资料查阅得知
'''
