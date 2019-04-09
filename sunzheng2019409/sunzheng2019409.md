### 4.8日试验进度

#### 目标：使用mask-rcnn进行青天白日旗的识别（效果如下）

![pic_1.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019409/pic_1.PNG?raw=true)

#### 截止4.8实验进展

##### 数据集

首先说明mask-rcnn的数据标注要稍微复杂一些，需要标出检测目标的轮廓：

![pic_2.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019409/pic_2.PNG?raw=true)

使用的标注工具是VGG Image Annotator，生成的标注文件是一个jason文件。该标注工具的使用教程如下：

https://blog.csdn.net/heiheiya/article/details/81530952

https://www.robots.ox.ac.uk/~vgg/software/via

目前已标注90张图像，其中70张用作训练集，20张用作验证集。

![pic_3.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019409/pic_3.PNG?raw=true)

##### 实验代码

本次实验代码是在原实验代码的基础上修改得到。使用了COCO数据集上的预训练模型，只重新训练“head”部分的参数，进行30个epoches。实验部分参考了链接：

https://blog.csdn.net/heiheiya/article/details/81532914

https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46

![pic_4.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019409/pic_4.PNG?raw=true)

![pic_5.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019409/pic_5.PNG?raw=true)

目前已经开始训练，明天能跑出结果