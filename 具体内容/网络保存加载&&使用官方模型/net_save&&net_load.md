# 网络的保存与加载
## 网络的保存
torch.save 是 PyTorch 中用于序列化和保存模型参数或整个模型状态的函数。这个函数通常用于保存训练过程中的模型参数，以便之后可以重新加载并继续训练，或者用于模型的部署和共享。
### 基本用法
```python
torch.save(object, filename)
```
object: 要保存的对象。它可以是模型的状态字典（state_dict），整个模型（model），或者其他 Python 对象。
filename: 保存对象的文件路径。

#### 第一种方法，保存整个模型
```python
#1.模型结构+参数
torch.save(vgg16,"./net/vgg16_method1.pth")
```

#### 第二种方法，以字典状态保存模型
```python
#2.模型参数
torch.save(vgg16.state_dict(),"./net/vgg16_method2.pth")
```


## 网络的加载

torch.load 是 PyTorch 中用于加载之前使用 torch.save 保存的序列化对象的函数。它通常用于加载训练好的模型参数、整个模型状态或者任何其他使用 torch.save 保存的 Python 对象。
### 基本用法
```python
torch.load(filename, map_location=None, pickle_module=None, **pickle_load_args)
```

- filename: 要加载的文件路径。
- map_location: 用于指定将保存的张量映射到哪个设备。例如，如果模型是在 GPU 上训练的，可以使用 map_location='cpu' 来将模型加载到 CPU 上。
- pickle_module: 用于反序列化的对象，通常不需要指定，因为 PyTorch 会自动识别。
- pickle_load_args: 传递给 pickle.load 函数的额外参数。

  ### 对整个模型的加载
  ```python
  # 加载整个模型，包括其参数和架构
model = torch.load('complete_model.pth')
  ```
  ### 对字典状态保存的模型的加载
  ```pyhton
  import torch
import torch.nn as nn

# 假设有一个与保存时相同架构的模型
model = nn.Linear(10, 5)

# 加载保存的状态字典
state_dict = torch.load('model_state.pth')
model.load_state_dict(state_dict)
  ```
### 加载模型到指定设备
```python
# 假设模型是在 GPU 上保存的，但我们现在想在 CPU 上使用它
model = torch.load('model.pth', map_location='cpu')
```
  
