# torchvision
## torchvision的引用
```pyhton
import torchvision
```

## torchvision是如何使用的
首先，我们可以通过其直接从pytorch官网上直接下载数据集，并通过设定相关参数和格式

详情参照官方文档：[官方文档](https://pytorch.org/pytorch-domains)

## 一个举例
eg：

```pyhton
dataset_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=True,transform=dataset_transform,download=True)
test_set=torchvision.datasets.CIFAR10(root="./torchvision_dataset",train=False,transform=dataset_transform,download=True)
```
**CIFAR10**是torchvisino中所包含的一个数据集，是一个32*32的图像集

其中各个参数对应着不同的要求：

1.root定义了下载后文件的所处地址

2.train为True表示下载的是训练集，反之表示为测试集合

3.transform表示了对文件的处理格式，在上述代码中我们通过类定义了一个对象，将数据结构变为Tensor

4.download为True表示下载


