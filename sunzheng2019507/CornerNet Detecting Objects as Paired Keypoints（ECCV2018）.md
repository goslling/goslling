# **CornerNet: Detecting Objects as Paired Keypoints**（ECCV2018）

论文链接：<https://arxiv.org/abs/1808.01244>

代码链接：<https://github.com/umich-vl/CornerNet>

## Abstract：

文中提出了一种新的目标检测方法，**使用单个卷积神经网络将目标边界框检测为一对关键点（即边界框的左上角和右下角）**。通过将目标检测为成对关键点，消除了现有的one stage检测器设计中对一组anchors的需要。除了上述新颖的构想，文章还引入了corner pooling，这是一种新型的池化层，可以帮助网络更好地定位边界框的角点。CornerNet在MS COCO上实现了42.1％的AP，优于所有现有的one stage检测器。

## Introducion：

基本上基于卷积神经网络的目标检测器已经在各种数据集上实现了state-of-art的水准。现有的优秀的目标检测器的一个共同的组成部分是anchor boxes（就是各种尺寸和长宽比的矩形框），是用作检测的候选框。anchor boxes广泛的用于one-stage和two-stage检测器上，实际证明anchor-boxes对目标检测有很好的提升。

但其有两个缺点。首先，一般在训练中通常需要很大数量的候选框，在ssd中超过4万，在retinaNet中超过10万。通过检测所有的候选框与GT的IoU来筛选出正负样本，但是正样本只占一小部分，巨大的正负样本差距使得训练速度很慢。

其次，anchor boxes的使用引入了很多超参数选择和设计的难题，这包括box的数量，大小，长宽比等等。**这些选择主要是通过ad-hoc启发式方法进行的**，当与图像的多尺度架构相结合时，会更加复杂。

在本文中，我们介绍了CornerNet，这是一种新的one stage目标检测方法，可以消除anchor boxes。 **文中将一个目标物体检测为一对关键点——边界框的左上角和右下角**。 我们使用单个卷积网络来预测同一物体类别的所有实例的左上角的热图，所有右下角的热图，以及每个检测到的角点的嵌入向量。 嵌入用于对属于同一目标的一对角点进行分组——训练网络以预测它们的类似嵌入。 文中的方法极大地简化了网络的输出，并且无需设计anchor boxes。 下图说明了该结构的整体流程。

![img](https://img-blog.csdn.net/20180904164356274?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

CornerNet提出了corner pooling，这是一种新型的池化层，可帮助卷积网络更好地定位边界框的角点。 边界框的一角通常在目标之外，在这种情况下，角点不能根据当前的信息进行定位。相反，为了确定像素位置是否有左上角，我们需要水平地向右看目标的最上面边界，垂直地向底部看物体的最左边边界。 这激发了我们的corner pooling layer：它包含两个特征图; 在每个像素位置，它最大池化从第一个特征映射到右侧的所有特征向量，最大池化从第二个特征映射下面的所有特征向量，然后将两个池化结果一起添加，如下图所示。

![img](https://img-blog.csdn.net/2018090417553845?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

为什么角点比检测边界框中心点或者proposals更好呢，文中假设了两个原因。首先，盒子的中心可能更难以定位，因为它取决于目标的所有4个边，而定位角取决于2边，因此更容易，甚至更多的corner pooling，它编码一些明确的关于角点定义的先验信息。 其次，角点提供了一种更有效的方式来密集地离散边界框的空间：只需要用O(wh)角点来表示O(w2h2)可能的anchor boxes。

## related work

### Two-stage目标检测器

Two-stage目标检测由R-CNN首次引入并推广[12]。Two-stage检测器生成一组稀疏的感兴趣区域(RoIs)，并通过网络对每个区域进行分类。R-CNN使用低层次视觉算法生成(RoIs)[41,47]。然后从图像中提取每个区域，由ConvNet独立处理，这将导致大量计算冗余。后来，SPP-Net[14]和Fast R-CNN[11]改进了R-CNN，设计了一个特殊的池化层(金字塔池化)，将每个区域从feature map中池化。然而，两者仍然依赖于单独的proposals算法，不能进行端到端训练。Faster-RCNN[32]通过引入区域生成网络(RPN)来去除低层次的proposals算法，RPN从一组预先确定的候选框(通常称为anchor boxes)中生成proposals。这不仅使检测器更加高效，通过RPN与检测网络的联合训练，可实现端到端训练。R-FCN[6]将全连接子检测网络替换为完全卷积子检测网络，进一步提高了Faster R-CNN的检测效率。其他的工作主要集中在结合子类别信息[42]，用更多的上下文信息在多个尺度上生成目标的proposals[1,3,35,22]，选择更好的特征[44]，提高速度[21]，并行处理和更好的训练过程[37]。

### one-stage目标检测器

另一方面，YOLO[30]和SSD[25]推广了one-stage方法，该方法消除了RoI池化步骤，并在单个网络中检测目标。One-stage检测器通常比two-stage检测器计算效率更高，同时在不同的具有挑战性的基准上保持着具有竞争性的性能。
SSD算法将anchor boxes密集地放置在多个尺度的feature maps之上，直接对每个anchor boxes进行分类和细化。YOLO直接从图像中预测边界框坐标，后来在YOLO9000[31]中，通过使用anchor boxes进行了改进。DSSD[10]和RON[19]采用了类似沙漏的网络[28]，使它们能够通过跳跃连接将低级和高级特性结合起来，从而更准确地预测边界框。然而，在RetinaNet[23]出现之前，这些one-stage检测器的检测精度仍然落后于two-stage检测器。在RetinaNet[23]中，作者认为密集的anchor boxes在训练中造成了正样本和负样本之间的巨大不平衡。这种不平衡导致训练效率低下，从而导致结果不佳。他们提出了一种新的loss，Focal Loss，来动态调整每个anchor boxes的权重，并说明了他们的one-stage检测器检测性能优于two-stage检测器。RefineDet[45]建议对anchor boxes进行过滤，以减少负样本的数量，并对anchor boxes进行粗略的调整。
DeNet[39]是一种two-stage检测器，不使用anchor boxes就能生成RoIs。它首先确定每个位置属于边界框的左上角、右上角、左下角或右下角的可能性。然后，它通过列举所有可能的角点组合来生成RoI，并遵循标准的two-stage方法对每个RoI进行分类。本文提出的方法和DeNet很不一样。首先，DeNet不识别两个角是否来自同一目标，并依赖子检测网络来拒绝糟糕的RoI。相比之下，我们的方法是一种one-stage方法，使用单个卷积网络来检测和分组角点。其次，DeNet在人工确定的位置上的区域提取特征进行分类，而我们的方法不需要任何特征选择步骤。第三，引入corner pooling，一种新型的用于增强角点检测的layer。

我们的方法受到Newell等人在多人姿态估计上下文中关联嵌入的启发[27]。Newell等人提出了一种在单个网络中检测和分组人类关节的方法。在他们的方法中，每个检测到的人类关节都有一个嵌入向量。这些关节是根据它们嵌入的距离来分组的。本文是第一个将目标检测任务定义为同时检测和分组角点的任务。我们的另一个新颖之处在于corner pooling layer，它有助于更好定位角点。我们还对沙漏结构进行了显著地修改，并添加了新的focal loss[23]的变体，以帮助更好地训练网络。

## CornerNet

### 3.1概述

在CornerNet中，我们将物体边界框检测为一对关键点（即边界框的左上角和右下角）。卷积网络通过预测两组热图来表示不同物体类别的角的位置，一组用于左上角，另一组用于右下角。 网络还预测每个检测到的角的嵌入向量[27]，使得来自同一目标的两个角的嵌入之间的距离很小。 为了产生更紧密的边界框，网络还预测偏移以稍微调整角的位置。 通过预测的热图，嵌入和偏移，我们应用一个简单的后处理算法来获得最终的边界框。

下图提供了CornerNet的概述。 我们使用沙漏网络[28]作为CornerNet的骨干网络。 沙漏网络之后是两个预测模块。 一个模块用于左上角，而另一个模块用于右下角。 每个模块都有自己的corner pooling模块，在预测热图、嵌入和偏移之前，池化来自沙漏网络的特征。 与许多其他物体探测器不同，我们不使用不同尺度的特征来检测不同大小的物体。 我们只将两个模块应用于沙漏网络的输出。

![图4](https://img-blog.csdn.net/20180905110230344?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 3.2检测角点

我们预测两组热图，一组用于左上角，另一组用于右下角。 每组热图具有C个通道，其中C是分类的数量，并且大小为H×W。 没有背景通道。 每个通道都是一个二进制掩码，用于表示该类的角点位置。

对于每个角点，有一个ground-truth正位置，其他所有的位置都是负值。 在训练期间，我们没有同等地惩罚负位置，而是减少对正位置半径内的负位置给予的惩罚。 这是因为如果一对假角点检测器靠近它们各自的ground-truth位置，它仍然可以产生一个与ground-truth充分重叠的边界框。我们通过确保半径内的一对点生成的边界框与ground-truth的IoU ≥ t（我们在所有实验中将t设置为0.7）来确定物体的大小，从而确定半径。 给定半径，惩罚的减少量由非标准化的2D高斯 给出，其中心位于正位置，其σ是半径的1/3。

![图5](https://img-blog.csdn.net/20180905175904493?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


### 3.3分组角点

图像中可能出现多个目标，因此可能检测到多个左上角和右下角。我们需要确定左上角和右下角的一对角点是否来自同一个目标边界框。我们的方法受到Newell等人提出的用于多人姿态估计任务的关联嵌入方法的启发[27]。Newell等人检测所有人类关节，并为每个检测到的关节生成嵌入。他们根据嵌入之间的距离将节点进行分组。
关联嵌入的思想也适用于我们的任务。 网络预测每个检测到的角点的嵌入向量，使得如果左上角和右下角属于同一个边界框，则它们的嵌入之间的距离应该小。 然后，我们可以根据左上角和右下角嵌入之间的距离对角点进行分组。 嵌入的实际值并不重要。 仅使用嵌入之间的距离来对角点进行分组。

我们关注Newell等人[27]并使用1维嵌入。etk成为对象k的左上的嵌入，ebk为右下角的的嵌入。 如[26]中所述，我们使用“pull”损失来训练网络对角点进行分组，并且用“push”损失来分离角点。

### 3.4Corner Pooling

如图2所示，通常没有局部视觉证据表明存在角点。要确定像素是否为左上角，我们需要水平地向右看目标的最上面边界，垂直地向底部看物体的最左边边界。因此，我们提出*corner Pooling*通过编码显式先验知识来更好地定位角点。

假设我们要确定(i,j)位置是左上角。设ft和fl为左上角池化层的输入特征映射，ftij和fli分别为(i,j)位置中ft和fl的向量。H×W的特征映射，corner pooling层首先最大池化ft中在(i,j)与(i,H)之间所有的特征向量，使之成为特征向量tij，还有，最大池化fl中在(i,j)与(W,j)之间所有的特征向量，使之成为特征向量lij。最后，把tij和lijl加在一起。

图

在这里，我们应用了一个elementwise最大化操作。动态规划可以有效地计算*t**i**j**和*l**i**j，如下图所示。

![img](https://img-blog.csdn.net/20180906115428132?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 实验

![这里写图片描述](https://img-blog.csdn.net/20180907095247615?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

该实验证明了corner pooling对目标检测影响明显 

![这里写图片描述](https://img-blog.csdn.net/20180907133124887?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQxNDI2Nw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

该实验将CornerNet与其他先进的目标检测模型进行了对比，可以发现在MS COCO test-dev上，CornerNet优于其他所有one-stage检测器，可实现与two-stage探测器相媲美的结果。