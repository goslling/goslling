周末和周一在跑Mask-Rcnn的代码。

github的地址是：https://github.com/goslling/Mask_RCNN-master

代码结构如下：

![pic_1.png](https://github.com/goslling/goslling/blob/master/pic_1.png?raw=true)

assets和images文件夹中存放着图片，assets存放的是处理过程的图片，images存的是测试集。该模型使用COCO数据集进行预训练，预训练权重存在mask_rcnn_coco.h5文件中，单独下载约300M。

mrcnn文件夹中存的是Mask-rcnn的模型：

![pic_2.png](https://github.com/goslling/goslling/blob/master/pic_2.png?raw=true)

模型在model.py中，parallel_model实现了多GPU同时训练的功能，默认两张卡上一起跑（可更改），最后统一计算loss，反向更新网络。utils,py存放通用的函数和类库，config.py存放网络中的默认参数，visualize.py实现了训练过程和测试过程中中间图像的输出功能。

samples文件夹中存放的是mask-rcnn应用的案例：

![pic_3.png](https://github.com/goslling/goslling/blob/master/pic_3.png?raw=true)

balloon文件夹以一个追踪气球的示例说明了maskrcnn的优越。我也是借鉴了这个实验的思路，重新训练海马的数据集。

![pic_5.png](https://github.com/goslling/goslling/blob/master/pic_5.png?raw=true)

![img](https://cdn-images-1.medium.com/max/1200/1*OKE6wyZFfh2f_aZ3rd9BRw.png)

运行程序需要下载预训练的权重和气球数据集，本次是要在自己的图像上做实验，这个实验就不实现了。

nucleus文件夹也是一个示例，寻找图像中的原子核。此处略过。

直接跳到训练自己的数据集部分：

![pic_6.png](https://github.com/goslling/goslling/blob/master/pic_6.png?raw=true)

首先我去参考了github给出的训练自己数据集的参考：https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

首先要处理自己的数据集，需要按照github上给的格式处理，数据集处理了50张国旗的，先看看运行效果。说一下为什么只需要50张，因为直接使用coco的预训练权重进行迁移学习，原coco数据集里有了很多种类识别，加上这次对精度要求不高，先制作50张。

目前正在训练（配环境神马的刚搞定），预计后面几天都会继续做这件事。

在此期间，刷了3道LeetCode的题目。

总结了这次跑实验的经验：https://github.com/goslling/goslling/blob/master/跑实验的经验.md

总结了yaml的知识点。

