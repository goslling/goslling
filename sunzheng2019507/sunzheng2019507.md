1.由于带毕设的学生时间节点有点紧张，就帮他跑了下程序。暂时搁置了mask-rcnn在文字定位上的实验，但是帮助学生跑的实验是CRNN文字识别的实验，也还算不亏。帮助带的学生跑程序，首先是使用纯CRNN（CNN+GRU+CTC）进行文字识别。现有一个pytorch版本的CRNN，有预训练权重。贴一下github地址：https://github.com/xiaofengShi/CHINESE-OCR。这个模型主要是进行中文字符及数字的识别，正好和证照的识别字符串类型相符，直接用在证照识别上效果不太好。然后考虑使用证照的2000多张图像在预训练权重上进行迁移训练。得到了还不错的效果。下面是tensoprboard的loss和acc的展示，具体的批量测试马上进行。

![1.png](https://github.com/goslling/goslling/blob/master/sunzheng2019507/1.png?raw=true)

![2.png](https://github.com/goslling/goslling/blob/master/sunzheng2019507/2.png?raw=true)

![3.png](https://github.com/goslling/goslling/blob/master/sunzheng2019507/3.png?raw=true)

![4.png](https://github.com/goslling/goslling/blob/master/sunzheng2019507/4.png?raw=true)

![5.png](https://github.com/goslling/goslling/blob/master/sunzheng2019507/5.png?raw=true)



2.阅读了文章CornerNet: Detecting Objects as Paired Keypoints，感觉这篇文章提出的模型很有吸引力。这是个object detection的文章，这篇文章在COCO数据集上实验效果吊打

YOLO v3.具体的阅读笔记如下：

https://github.com/goslling/goslling/blob/master/sunzheng2019507/CornerNet%20Detecting%20Objects%20as%20Paired%20Keypoints%EF%BC%88ECCV2018%EF%BC%89.md

3.学习了pytorch的使用， 上手还算快，比tensorflow容易入门 

