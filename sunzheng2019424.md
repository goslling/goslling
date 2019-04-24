# YOLO

最近也是想了很多，打算最近的重点是跑海马的那边的证件代码。首先是使用Mask rcnn跑一下日期和编号定位，其次是字符串识别方面，使用attention-ocr模型进行字符串识别。

胡老师那边有个沙龙，最近主要在讲目标检测，上周我讲了一下two-stage的目标检测算法。从RCNN到mask-rcnn讲了下其中的技术细节，这周要讲的是yolo。正好最近在看的一篇论文[CornerNet-Lite: Efﬁcient Keypoint Based Object Detection](https://arxiv.org/abs/1904.08900),这篇论文讲的是一个keypoint based的目标检测框架，在文中作者将自己的监测效果和yolo v3的进行对比，但是之前还没有详细了解过yolo 三个版本的具体结构和识别精准度，这两天正在补这个知识。

首先阅读了yolo v1的论文，这是yolo的基础，阅读笔记在：

齐次阅读了csdn上的yolo v2和v3的简介，其论文后边补上，然后他们的实现代码也得跑起来：[v2参考链接](<https://blog.csdn.net/jesse_mx/article/details/53925356>)，[v3链接](<https://www.jiqizhixin.com/articles/2018-05-14-4>)









