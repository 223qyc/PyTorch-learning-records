# 关于数据的处理
## 认识处理数据的两个类
### 1.Dataset
翻译：数据集

实际上是通过该类获取数据信息，包括从数据集就中读出数据及其label
- 获取每一个数据及其label
- 得知所有数据的总数

### 2.Dataloader
翻译：数据装载器

对即将处理的数据提供处理，比如形式的处理，batch_size的打包，为网络提供不同的数据形式

## 关于Dataset
```python
from torch.utils.data import Dataset
```

这是调用该类代码语句
在上传的代码中，还有语句：

```python
from PIL import Image
import os
```

在该代码中，创建了一个自定义的类，继承了Dateset

在__init__函数中，传入根目录和标签目录，并将标签目录下的地址全部转化为链表形式

在__getitem__函数中，通过传入索引获得图片的名称，并通过地址的拼接得出完整的相对地址，将图片打开并返回其label值

在代码的实际运行中，将ants和bees两个数据获得后，还可以利用+将其相加

**新习得的操作：**
```python
img_item_path=os.path.join(self.root_dir,self.label_dir,img_name) #路径的拼接
self.img_path=os.listdir(self.path)     #读取该路径下所有的内容，并将内容存储在列表中
train_dateset=ants_dataset+bees_dateset #类的直接相加
```

**python控制台关于图像的处理:**
```python
from PIL import Image
img_path="xxxxxx"
img=Image.open(img_path)
img.show()
```

## 关于DataLoader
### 引用
```pyhton
from torch.utils.data import  DataLoader
```
### DataLoader的使用原理
相当于一个用于规范批处理数据，比如指定批大小，随机性等

**主要方法还是参考官方文档，阅读各个参数代表含义**,[官方文档](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

- dataset（数据集）：需要提取数据的数据集，Dataset对象
- batch_size（批大小）：每一次装载样本的个数，int型
- shuffle（洗牌）：进行新一轮epoch时是否要重新洗牌，Boolean型
- num_workers：是否多进程读取机制
- drop_last：当样本数不能被batchsize整除时,是否舍弃最后一批数据

### DataLoader中返回数据到底是怎样的类型
eg:
```python
for data in test_lodar1:
    imgs,tags=data
    print(imgs.shape)
    print(tags)

```

我们假设一个数据集中第一个对象包含img和tag，则dataloader中会将batch_size中的数据所有img和tag打包为一个imgs和tags

### 关于在Tensorboard中进行图像的绘制


特别注意：批处理下的绘图使用add_images而非add_image




### dataloader中参数对图像的影响
eg：
```pyhton
for epoch in range(2):
    step = 0
    for data in test_lodar2:
        imgs, tags = data
        writer.add_images("Epoch__{}".format(epoch), imgs, step)
        step = step + 1
```

首先drop_last设置未False表示最后不会舍去，因此我们可以在图像中看到最后step中数量不足batch_size

shuffle为Ture表示数据抽取是随机的，因此在epoch_0和epoch_1中是两个完全不同的图像呈现






