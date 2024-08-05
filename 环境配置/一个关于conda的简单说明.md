# anaconda终端启动与jupyter的打开

## 启动anconda 

### 怎么知道自己打开了？

地址左侧出现(base)或者()中是自己配置的环境

### 打开方式

1.直接点击Anaconda Prompt或者Anaconda PowerShell进入即可。

2.在cmd中输入**activate base**进入base环境.

3.在已进入的base环境下再通过**conda avtivate XXX**进入自己要进入的环境。

## 启动Jupyer notebook

### 须知

从哪里进入了Jupyter noteboook，则将来的文件会存储在哪里

### 启动的最好方法

这里最好使用cmd，因为cmd可以进入base环境从而进入其他自定义环境

所以具体的操作是

先从cmd进入要存放文件的位置

再进入base环境从而进入自定义环境

然后输入jupyter notebook打开

### cmd打开指定路径

可以选择在cmd中通过指令逐步打开

也可以直接在要打开的文件夹或文件处上方路径栏直接输入cmd进入

### cmd和PowerShell的区别

提供参考如下：

[参考1：两者的区别](https://blog.csdn.net/u013589130/article/details/129573557)

[参考2：两者的选择](https://blog.csdn.net/Dream_Weave/article/details/86791463)

