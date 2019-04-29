1.完成了ImageNet_ILSVRC2012数据集的下载，准备做一个图像数据集的库放在服务器上，目前该数据集放在了

T630_5上，具体路径是/mnt/data/imagedata$ ，有需要的同学可以使用该数据集，预计以后会添加更多的图像数据集

2.完成了之前SSD数据集的格式改动，maskrcnn需要json格式标签，在代码里改了响应的接口，改成了读取xml格式的标签。明天就能跑起来

3。阅读了文章CornerNet: Detecting Objects as Paired Keypoints（ECCV2018），这是一种新型的one-stage目标检测算法，具体的阅读笔记在：https://github.com/goslling/goslling/blob/master/CornerNet%20Detecting%20Objects%20as%20Paired%20Keypoints%EF%BC%88ECCV2018%EF%BC%89.md

