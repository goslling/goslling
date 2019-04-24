# yolo V1阅读笔记

yolo的全称是you look only once，很有内涵的一个名字。意味着看一眼就能监测出图像中的目标物体。这是一篇CVPR2016的一篇文章。

之前的目标检测模型都是two-stage的监测模型，首先进行 region proposal，再对推荐的框进行fine tune。yolo是一个one-stage的监测模型，将监测变为一个regression problem。yolo从输入图像，仅仅经过一个深层卷积神经网络，直接得到bbox和所属类别及其概率。正因为整个的检测过程仅仅有一个网络，所以它可以直接end-to-end的优化。

yolo的特点是非常快，标准yolo每秒可以实时处理45帧图像。fast yolo可以每秒处理155帧图像。同时yolo在与其他同时期的检测模型（faster rcnn）相比，在框的 坐标上可能误差更大一些，但是其背景误差更小一些（将背景噪点识别为前景目标的概率）。

最后，yolo可以学到物体更加泛化的特征，在将yolo应用到其他领域的图像时（如artwork 图像），其检测效果会好于rcnn。



![aaa](https://img-blog.csdn.net/20160317163739691)

yolo对于RCNN之类的two-stage模型有以下两个好处：

- yolo的检测速度很快，在titan x上他可以达到45fps。（多讲一句，这个可算不上是实时。因为要落地到实际检测设备上，不可能配上titan X显卡给你inference的。所以后续的目标检测方向我认为是降低对设备的算力要求的，即更加简洁的网络，更少的运算步骤）
- yolo在做inference时，使用的是全局图像，与region proposal相比，yolo一次看一整张图像，所以它可以将物体的上下文的信息考虑进去。fast-rcnn比较容易误将图像中的背景噪点看成是物体，因为它看的范围要小很多。
- yolo学到物体更加泛化的特征表示。当在自然图像上训练yolo，再在artwork图像上去测试yolo时，yolo的表现甩rcnn好几条街。

下面是yolo的原理，和ssd一样使用了分格的思想，首先将图像分为sxs个grid。如果一个物体的中心落在一个grid内，那么这个grid就负责检测这个物体。每一个grid预测b个bbox，以及这些bbox的得分情况：score。这个score的计算方法如下：

![](https://img-blog.csdn.net/20160317154752063)

如果有object落在一个grid里，第一项取1，否则取2.第二项是预测的 bbox和gt之间的IoU值。每个bbox要预测（x，y，w，h）和score五个值，每个网格还要预测一个class信息，一共有c类。则sxs个网格要输出的就是s*s *（5 *b+c）的tensor。需要注意的是score是针对每个每个bbox的，而class信息是针对每个网格 的（意思是每个网格只能预测一种物体，对于物体密集的图像来说，这不太好。）

举例说明：在PASCAL VOC中，图像输入为448x448，取S=7，B=2，一共有20个类别(C=20)。则输出就是7x7x30的一个tensor。整个yolo的网络结构如下：

![](https://img-blog.csdn.net/20160317155955917)

在进行图像测试的时候，每个网格预测 的class信息和bbox的score相乘，就得到了每个bbox的class confidence：

![](https://img-blog.csdn.net/20160317161412042)

等式左边第一项就是每个网格预测的类别信息，二三项就是每个bbox的score。这样就可以得到每个bbox的class score，然后设置阈值，滤掉得分低的boxes。对保留的boxes进行NMS处理，就得到了最终的监测结果。

