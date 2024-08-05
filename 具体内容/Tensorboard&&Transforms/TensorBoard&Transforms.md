# TensorBoard&Transforms
## Tensorboard
TensorBoard是一个可视化工具，它可以用来展示网络图、张量的指标变化、张量的分布情况等。

特别是在训练网络的时候，我们可以设置不同的参数（比如：权重W、偏置B、卷积层数、全连接层数等），使用TensorBoader可以很直观的帮我们进行参数的选择。

它通过运行一个本地服务器，来监听6006端口。在浏览器发出请求时，分析训练时记录的数据，绘制训练过程中的图像。

通过语句:

```from torch.utils.tensorboard import SummaryWriter```

来使用这一模块

```对象名=SummaryWriter("文件名")```
### 图像绘制
```python
for i in range(100):
    writer.add_scalar("y=x^2-2x+1",i*i-2*i+1,i)#标题，纵坐标，横坐标
writer.close()
```

**对象.add_sclar(tag,y,x)**

### 图像处理
图像打开的数据类型是Tensor和numpy
```python
img_path="data/train/ants/6240338_93729615ec.jpg"
img_PIL=Image.open(img_path)
img_Array=np.array(img_PIL)
print(type(img_Array))
print(img_Array.shape)
writer.add_image("test2",img_Array,1,dataformats="HWC")
writer.close()
```
>img_Array=np.array(img_PIL) 是将图像转化为numpy形式的办法[^注释]
[^注释]:但仅仅是基于PIL的处理，cv2读取的无需转化就是numpy形式

**对象.add_image()**
### 打开方式
```
tensorboard --logdir=(所在文件夹名) --port=指定端口号
```

名称一样可能会重复拟合出现错误，image中同一标题下step不同可以来回切换

### 注意事宜
- 注意名称相同的多次拟合，出现错误删除文件夹内容，终止程序再次运行即可
- 使用诸如add_sclar()和add_image()时,可以通过在pycharm中按ctrl点击函数查看参数内容

 eg:通过查看add_image()函数得知

```python
def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW")
img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
#将数据转化为numpy后shape为(181, 500, 3)
Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
#因此添加一句dataformats="HWC"
```

## Transforms
总的来说，transforms像是一个大的package，包含了很多对图片处理的方法，包括格式转换，重塑大小等
### 认识各种数据类型
#### PIL
python的自带的图像处理数据类型

通过指定path，通过open，show，可以实现图像的打开，展示

诸如代码片段：

```python
img_path="data/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img=Image.open(img_path)
print(img)
```
#### Tensor数据类型
通过ToTensor实现对PIL的类型转换

诸如代码片段：

```python
tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)
print(tensor_img)
```
#### numpy数据类型
默认使用cv2获取为nunmpy，也可以通过np进行转换

诸如代码片段：

```python
cv_img=cv2.imread(img_path)
print(cv_img)   #nunmpy打开形式
```

但是要调用对应cv2

```import cv2```

### 常见的Transforms
##### 1.Tosenor
数据类型的转换

eg：
```python
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("Totensor",img_tensor)
```
##### 2.Normalize
归一化手段，对应通道，output[channel] = (input[channel] - mean[channel]) / std[channel]

eg：
```pyhton
trans_norm=transforms.Normalize([0.5,0.5,0.5],[1,1,1])
img_norm=trans_norm(img_tensor)
print(img_tensor)
print(img_norm)
writer.add_image("Normalize",img_norm,4)
```

#### 3.Resize
重塑大小，参数取决于一个int还是一个tuple，实际效果取决于参数决定，详情参见官方文档

eg：

```pyhton
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)    #注意，处理对象是PIL，是变换之前的
print(img_resize.size)
img_resize=trans_totensor(img_resize)
writer.add_image("Resize",img_resize)
```

#### 4.Compose
可以将若干个指令执行顺序共同存放在一个列表中执行，但顺序是否正确十分重要

例如旧版本中resize只能处理PIL，因此ToTensor只能在其后

但现在resize可以直接处理Tensor数据类型，值得注意

eg：
```python
trans_resize_2=transforms.Resize((1000,700))
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)
```

#### 5.RandomCrop
提供随机裁剪

同样，传递的参数要求，实际效果要参考官方文档

eg：
```python
trans_random=transforms.RandomCrop(700)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop2",img_crop,i)
```

### 对Transforms的总结
1.想要了解对应功能的使用方法直接在pycharm中查阅官方文档是最好的解决办法

2.其次要注意参数，注意参数的接收类型，避免传参数据类型错误。同时，在诸如resize和crop中，注意参数的个数所影响的实际效果。对参数的选择，也是以参考官方文档为主。

3.对每一个方法的使用本质上是先创建一个类，在类中进行调用

eg:
```pyhton
trans_random=transforms.RandomCrop(700)
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop=trans_compose_2(img)
    writer.add_image("RandomCrop2",img_crop,i)
```

先根据官方文档中的类自定义一个对象，对对象进行函数的调用


```python
trans_compose_2=transforms.Compose([trans_random,trans_totensor])
```

这是一个综合的体现，Compose中的参数是自己定义的对象，最终赋值的是自己定义的对象，定义的过程是借助了官方所定义的类



