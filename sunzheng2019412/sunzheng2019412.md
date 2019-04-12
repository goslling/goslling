更新下mask-rcnn的进度：

目前使用小规模的训练集实现了不错的效果（two-stage确实比one-stage的效果好很多），后续加大训练集可以取得更好的效果是毋庸置疑的。也就是海马的数据后续马上就能开跑。

先说说目前的实验：

使用56张训练集，20张验证集。

![pic.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019412/pic.PNG?raw=true)

训练过程中的loss：

![pic1.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019412/pic1.PNG?raw=true)

训练30个epoches，每个epoch处理100张图片。30个epoch后得到分类loss为0.0340，bounding-box的loss为0.0523，mask的loss为0.1254.

得到的训练权重：

![pic2.PNG](https://github.com/goslling/goslling/blob/master/sunzheng2019412/pic2.PNG?raw=true)

使用训练得到的权重对测试集图像进行测试得到的结果：

![pic3.png](https://github.com/goslling/goslling/blob/master/sunzheng2019412/pic3.png?raw=true)