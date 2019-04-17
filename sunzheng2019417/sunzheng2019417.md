总结下最近的工作。

周一和张老师谈了下自己最近的工作和状态。对接下来的有如下安排：

1.最紧要的事情是评估mask-rcnn的性能，得出在测试集上的mAP值。看一下最近一直在做的mask-rcnn的实验究竟能不能用在项目中。

2.跟做证件识别的学弟对接进度和数据集，做一下证照识别（仅许可证编号）的工作。

3.调研SED的方向，跟师兄交流。

4.保证每周阅读至少两篇论文

1）目前完成了对mask-rcnn的性能评估：

首先看评价指标：

![pic1.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019417/pic1.PNG?raw=true)

precisions是一个不同实体类别的分类精准度列表。

recalls是一个不同实体类别分类recall值得列表。

overlaps是预测bounding box和ground truth b box的IoU列表。

mAP是平均的精准度

在测试集一共20张图片上得到了一个结果，其中第一列是每张图片的AP值，第二三列是precision和recall，第四列是overlaps。可以看到AP值大多数为1，一部分AP值偏低。

![pic2.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019417/pic2.PNG?raw=true)

可以看到mAP值为0.715。这个mAP值并不高，分析一下不高的原因。我把AP值低于0.9的图像保存了下来

图1：

![0.99869895.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.99869895.png?raw=true)

这张图中的青天白日旗比较多，最终只识别出了一个青天白日旗，另外还误识别了一个领带。

![0.99976987.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.99976987.png?raw=true)

这张图中要识别的青天白日旗更多，而且待识别的实体很小。

![0.9998115.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.9998115.png?raw=true)

这张上可以看到只识别出了一张旗帜，还把讲台误识别成了旗帜。

![0.9998331.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.9998331.png?raw=true)

这张图像把三张重叠的旗帜识别成了一张旗帜，还另外将船体误识别为了旗帜

![0.99986887.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.99986887.png?raw=true)

这张倒是识别出了所有的旗帜，但是它把非青天白日旗的旗帜也分成了青天白日旗，属于误分类

![0.99999213.png](https://github.com/goslling/goslling/blob/master/sunzheng2019417/0.99999213.png?raw=true)

可以看到这张图里只识别出了最明显的一张青天白日旗，剩余的两幅旗帜没有识别出

综合以上的误识别的结果，可以总结一下：

误识别：长得形状像青天白日旗的其他图案会被误识别，背景色为红色的图案会被误识别

漏识别：待识别体积过小的实体容易被漏识别

误识别和漏识别两种情况导致了整体的mAP值不高

解决方法：针对误识别情况，可以增加数据集，增加迭代次数。毕竟得到这个效果只用了58张训练图像，30次迭代（在coco预训练模型的基础上重新训练）

​                  针对漏识别，可以在训练前将proposal框设得更小一些。有很多参数都可以调整。

有了调整方案，可以将maskrcnn放到证照识别上试试效果

对于证照识别，我已经跟张宸赫对接过了，目前等他把 去水印的程序接口发过来，整体处理一下数据集，就可以进行证照识别的工作。

SED方面的话，打算周四讲一下maskrcnn的内容。