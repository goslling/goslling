# tensorflow概念

***本篇总结参考链接[tensorflow中文社区](http://www.tensorfly.cn/)***

tensorflow是一个采用**数据流图**，用于数值计算的开源软件库。**节点**在图中表示数学操作，图中的**线**表示在节点间相互联系的多维数据数组，即**tensor**。

数据流图用“**结点**”（nodes）和“**线**”(edges)的有向图来描述数学计算。“**节点**” 一般用来表示施加的数学操作，但也可以表示数据输入（feed in）的起点/输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。**“线”表示“节点”之间的输入/输出关系。这些数据“线”可以输运“size可动态调整”的多维数据数组，即“张量”（tensor）。张量从图中流过的直观图像是这个工具取名为“Tensorflow”的原因。**一旦输入端的所有张量准备好，节点将被分配到各种计算设备完成异步并行地执行运算。

---

使用tensorflow，你必须明白tensorflow：

- 使用图（graph）来表示计算任务
- 在被称之为会话（session）的上下文中执行图
- 使用tensor表示数据
- 通过变量（variable）维护状态
- 使用feed和fetch可以为任意的操作赋值或者从其中获取数据

#### 综述

TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 *op*  (operation 的缩写). 一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor`. 每个 Tensor 是一个类型化的多维数组.  例如, 你可以将一小组图像集表示为一个四维浮点数数组,  这四个维度分别是 `[batch, height, width, channels]`.

一个 TensorFlow 图*描述*了计算的过程. 为了进行计算, 图必须在 `会话` 里被启动. `会话` 将图的 op 分发到诸如 CPU 或 GPU 之类的 `设备` 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回. **在 Python 语言中, 返回的 tensor 是 [numpy](http://www.numpy.org/) `ndarray` 对象; 在 C 和 C++ 语言中, 返回的 tensor 是  `tensorflow::Tensor` 实例.**

#### 计算图

TensorFlow 程序通常被组织成一个**构建阶段**和一个**执行阶段**. 在构建阶段, op 的执行步骤 被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op.

例如, ***通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.***

TensorFlow 支持 C, C++, Python 编程语言. 目前, **TensorFlow 的 Python 库更加易用**, 它提供了大量的辅助函数来简化构建图的工作, 这些函数尚未被 C 和 C++ 库支持.

三种语言的会话库 (session libraries) 是一致的.

### 构建图

构建图的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 `常量 (Constant)`. 源 op 的输出被传递给其它 op 做运算.Python 库中, op 构造器的返回值代表被构造出的 op 的输出, 这些返回值可以传递给其它 op 构造器作为输入.

TensorFlow Python 库有一个*默认图 (default graph)*, op 构造器可以为其增加节点. 这个默认图对 许多程序来说已经足够用了. 阅读 [Graph 类](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Graph) 文档 来了解如何管理多个图.

### 在一个会话中启动图

构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 `Session` 对象, 如果无任何创建参数, 
会话构造器将启动默认图.

`Session` 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块
来自动完成关闭动作.

在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU). 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow  会尽可能地利用找到的第一个 GPU 来执行操作.

如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow  使用这些 GPU, 你必须将 op 明确指派给它们执行. `with...Device` 语句用来指派特定的 CPU 或 GPU 执行操作

### 交互式使用

文档中的 Python 示例使用一个会话 [`Session`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#Session) 来 启动图, 并调用 [`Session.run()`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#Session.run) 方法执行操作.  

为了便于使用诸如 [IPython](http://ipython.org/) 之类的 Python 交互环境, 可以使用 [`InteractiveSession`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#InteractiveSession) 代替 `Session` 类, 使用 [`Tensor.eval()`](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Tensor.eval) 和 [`Operation.run()`](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Operation.run) 方法代替 `Session.run()`. 这样可以避免使用一个变量来持有会话.

### Tensor

TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor.
你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和
一个 shape. 想了解 TensorFlow 是如何处理这些概念的, 参见
[Rank, Shape, 和 Type](http://www.tensorfly.cn/tfdoc/resources/dims_types.html).

### 变量

[Variables](http://www.tensorfly.cn/tfdoc/how_tos/variables.html) for more details.变量维护图执行过程中的状态信息. 下面的例子演示了如何使用变量实现一个简单的计数器. 参见
[变量](http://www.tensorfly.cn/tfdoc/how_tos/variables.html) 章节了解更多细节

代码中 `assign()` 操作是图所描绘的表达式的一部分, 正如 `add()` 操作一样. 所以在调用 `run()`  执行表达式之前, 它并不会真正执行赋值操作.

**通常会将一个统计模型中的参数表示为一组变量. **例如, 你可以将一个**神经网络的权重**作为某个变量存储在一个 tensor 中. 在训练过程中, 通过重复运行训练图, 更新这个 tensor.

### Fetch

为了取回操作的输出内容, 可以在使用 `Session` 对象的 `run()` 调用 执行图时, 传入一些 tensor,这些 tensor 会帮助你取回结果. 在之前的例子里, 我们只取回了单个节点 `state`, 但是你也可以取回多个tensor。

### Feed

上述示例在计算图中引入了 tensor, 以常量或变量的形式存储. TensorFlow 还提供了 feed 机制, 该机制 可以临时替代图中的任意操作中的 tensor    可以对图中任何操作提交补丁, 直接插入一个 tensor.

feed 使用一个 tensor 值临时替换一个操作的输出结果. 你可以提供 feed 数据作为 `run()` 调用的参数. feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 最常见的用例是将某些特殊的操作指定为 "feed" 操作, 标记的方法是使用 tf.placeholder() 为这些操作创建占位符. 

 

```python

```

### 关键函数

#### 1. tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)

是tensorflow里面实现卷积的函数

除去name参数用以指定该操作的name，与方法有关的一共五个参数：

1. 第一个参数**input**：指需要做卷积的输入图像，它要求是一个Tensor，具有**[batch, in_height, in_width, in_channels]**这样的shape，具体含义是***[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]***，注意这是一个4维的Tensor，要求类型为**float32**和**float64**其中之一
2. 第二个参数filter：相当于CNN中的卷积核，`它要求是一个Tensor，具有`**[filter_height, filter_width, in_channels, out_channels]**`这样的shape`，具体含义是**`[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数`]**，要求类型与参数input相同，有一个地方需要注意，第三维`in_channels`，就是参数input的第四维
3. 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4。[1,n,n,1],n表示步长
4. 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，"SAME"表示feature与原图同尺寸。
5. 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true

结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是`[batch, height, width, channels]`这种形式。

#### 2. tf.name_scope('scope_name')

主要与tf.Variable搭配使用；当传入字符串时，用以给变量名添加前缀，类似于目录，如case1所示；当传入已存在的name_scope对象时，则其范围内变量的前缀只与当前传入的对象有关，与更上层的name_scope无关，如case2所示。

import tensorflow as tf

##### case1：

with tf.name_scope('l1'):
	with tf.name_scope('l2'):
		wgt1 = tf.Variable([1,2,3], name='wgts')
		bias1 = tf.Variable([0.1], name='biases')

print wgt1.name, bias1.name

##### >>> l1/l2/wgts:0 l1/l2/biases:0

##### case2：

with tf.name_scope('l1') as  l1_scp:
	with tf.name_scope('l2'):
		wgt0 = tf.Variable([1,2,3], name='wgts')
		bias0 = tf.Variable([0.1], name='biases')
		with tf.name_scope(l1_scp):
			wgt1 = tf.Variable([1,2,3], name='wgts')
			bias1 = tf.Variable([0.1], name='biases')

print wgt0.name, bias0.name, wgt1.name, bias1.name
##### >>> l1_1/l2/wgts:0 l1_1/l2/biases:0 l1_1/wgts:0 l1_1/biases:0
#### 3. tf.variable_scope(named_scope)

与name_scope一样：当传入字符串时，用以给变量名添加前缀，类似于目录；当传入已存在的variable_scope对象时，则其范围内变量的前缀只与当前传入的对象有关，与更上层的variable_scope无关。常于get_variable搭配使用，多用于变量共享；其中 reuse 参数可设为 None、tf.AUTO_REUSE、True、False；

**当 reuse=None（默认情况）时，与上层variable_scope的reuse参数一样。**

##### case 1
with tf.variable_scope('lv1'):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name
##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0



***

**当 reuse=tf.AUTO_REUSE 时，自动复用，如果变量存在则复用，不存在则创建。这是最安全的用法。**

with tf.variable_scope('lv1'):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])
print wgt1.name, bias1.name

##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0

with tf.variable_scope('lv1', reuse=tf.AUTO_REUSE):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt2 = tf.get_variable('wgts', [2,2])
		bias2 = tf.get_variable('biases', [2,2])
print wgt2.name, bias2.name

##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0
---------------------
with tf.variable_scope('lv1', reuse=tf.AUTO_REUSE):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt2 = tf.get_variable('wgts', [2,2])
		bias2 = tf.get_variable('biases', [2,2])

print wgt2.name, bias2.name
##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0
---------------------
**当 reuse=True 时，tf.get_variable会查找该命名变量，如果没有找到，则会报错；所以设置reuse=True之前，要保证该命名变量已存在。**

with tf.variable_scope('lv1', reuse=True):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name
>>> **ValueError: Variable lv1/lv2/wgts does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?**

---------------------
**命名变量已存在：**

with tf.variable_scope('lv1'):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name
##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0

##### case 2
with tf.variable_scope('lv1', reuse=True):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name
##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0
---------------------
**当 reuse=False 时，tf.get_variable会调用tf.Variable来创建变量，并检查创建的变量是否以存在，如果已存在，则报错；**

with tf.variable_scope('lv1'):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name
##### >>> lv1/lv2/wgts:0 lv1/lv2/biases:0

##### case 2
with tf.variable_scope('lv1', reuse=False):
	with tf.variable_scope('lv2'):
		init = tf.constant_initializer(0.1)
		wgt1 = tf.get_variable('wgts', [2,2])
		bias1 = tf.get_variable('biases', [2,2])

print wgt1.name, bias1.name

**ValueError: Variable lv1/lv2/wgts already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? **

---------------------
