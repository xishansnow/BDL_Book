# 贝叶斯神经网络最新综述

【原文】Goan, E., & Fookes, C. (2020). Bayesian Neural Networks: An Introduction and Survey.  https://arxiv.org/abs/2006.12024

【摘要】神经网络已经为许多机器学习任务提供了最先进的结果，例如计算机视觉、语音识别和自然语言处理领域的检测、回归和分类任务等。尽管取得了成功，但它们通常是在频率学派框架内实施的，这意味着其无法对预测中的不确定性进行推断。本文介绍了贝叶斯神经网络及一些开创性研究，对不同近似推断方法进行了比较，并提出未来改进的一些方向。

---
# **1 引言**

下一代神经网络的演化方向是什么？最近两年在北京举行的智源大会都谈到了这个问题，可能性的一个答案是贝叶斯神经网络，因为它可以对已有的知识进行推断。逻辑推断作用就是可以对已有的知识进行延伸扩展。

举个例子，如果询问训练完善的 AI 模型的一个问题，“在乌克兰，新西兰，新加坡，阿尔及利亚这四个国家里，哪一个国家位于中国的最西边”，这个问题的难点就在于那个“最”字，如果是传统的 AI 模型可能会蒙圈，因为乌克兰和阿尔及利亚都是在中国的西边，因为现有的训练的知识并不足以告诉它哪个是最西边，经过 BNN（贝叶斯神经网络）训练的模型可能会从经纬度，气温等其他信息进行推断得出一个阿尔及利亚在中国的最西边这个答案。

BNN 的最新进展值得每个 AI 研究者紧密关注，**本文就是一篇新鲜出炉的关于 BNN 的综述**，为了方便读者的阅读，我按照自己的节奏和想法重新梳理了一下这篇文章。

# 2  文献调研

## **2.1  神经网络**

先回顾一下传统神经网络，论文限于篇幅的原因有一些重要的细节没有展开，而且我一直觉得神经网络中一个完善的形式应该是通过矩阵的形式表现出来，同理矩阵形式 BP 反向传播原理也能一目了然。

### **2.1 标量形式的神经网络**

下图为标量形式的神经网络，并且为了说明方便不考虑偏置项。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy92SGhnam9FaEtHWThPeHVjaWN3UGF0ejA4cmtlSFRSZWlhS3dMdnc3a3FSc3dXc1NQTDJRbVFwY0RhekswNFY0cFdJWFhGOUw4eGNKTXlqZGo1SmRnR1d3LzY0MA?x-oss-process=image/format,png)

给定一个训练样本 ，假设模型输出为 ，则均方误差为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1pY2xiMFlJcGZDM3VpYWlibFN1V3ZJd1pOOEUzSWlhaWJqVmlhNzQ1WWFWYWxmNG5ybnVRZkpHYlVJRFEvNjQw?x-oss-process=image/format,png)

根据梯度下降法更新模型的参数，则各个参数的更新公式为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1QalZQS2FFdE5VZHBRVG5qMFdvdlBHbkN0WkZCZ0E2aGlhS3d1OE9XbDFOVXRkNW03M0FNODlnLzY0MA?x-oss-process=image/format,png)

链式法则求解 会有如下推导形式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1MQ21aV1dLckRoMFBTNUdBTVNIaWFDamljcURSRm8zMndkOUI2V3V4Z2V4UTNQM1ZNcjVTbXV2Zy82NDA?x-oss-process=image/format,png)

链式法则求解 会有如下推导形式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1LdVdyVldkcHh3aWJGVENydXdXd25VZGFSa0JZWTd1Q2c2ZEdKS25ISEZYSng5NUtVcWlhaE53Zy82NDA?x-oss-process=image/format,png)

可以发现标量视角下的神经网络更新参数求解梯度会给人一种很混乱的感觉。

### **2.2 矩阵形式的神经网络**

下图为 3 层不考虑偏置项的全连接神经网络示意图：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek14MmxRdndhT3gwTGJuSXVWNXZuNWpwTEY3a1h5a2ZaNkVlaWNzNWRnU0dPZFBwOXpMRFhsblRBLzY0MA?x-oss-process=image/format,png)

上图可以描述为如下公式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1rTWJkUGljbzJpYjFjU0dFZ013bUJFQWRsUmV1cHNDYmY1Z2ljSDdkQVZObkhLVzZGY0F2NGpyZ1EvNjQw?x-oss-process=image/format,png)

损失函数如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek01VDhIcmFCSGtYOXVSMUREaWJLNGt4VVAzRFVLVWhoYjA0N0lNRWpxT2xyNE95SEdpY29ybFltUS82NDA?x-oss-process=image/format,png)

优化的目标函数为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek10TU1jYThldkJ0YW9kWjBUamM3TTB4dkNXYUppYjR2SlVQVmZlMmFpYWM3OHRVVjJ0VjZpYW9YcUEvNjQw?x-oss-process=image/format,png)

其中， 表示的权重矩阵， 为隐层向量。

**2.2.1 随机梯度**

采用随机梯度下降法求解优化深度神经网络的问题，如下式所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek14YXJnRlozSURMaWN5cVBKemRJUElmbnBlcHl2alY5UThFUnppYmIxOUtndWFIRUd0dm9CNGNKdy82NDA?x-oss-process=image/format,png)

上式中，主要的问题是在于计算 ，通常采用的方法是链式法则求导。而反向传播就是一种很特殊的链式法则的方法。反向传播非常有效的避免大量的重复性的计算。

**2.2.2 无激活函数的神经网络**

L 层神经网络的无激活函数的目标函数定义为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1ZR1hpYVJ3U0ZLRXJmWGVzTXhXSzhod3F0U250d1BLODFIVlE3VDc3TXpvaWNEMzBUVmJob1g0Zy82NDA?x-oss-process=image/format,png)

则各个层的梯度有如下形式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1ZNGliQnRpYjR0cWFsZlNnMkcxOFdqNThXbUZpYTBCem5LRlFzNzdzbUI3eThRdXJHZXdCdm96SHcvNjQw?x-oss-process=image/format,png)

其中， 。

**2.2.3 含有激活函数的神经网络**

首先，考虑 2 层的有激活函数的神经网络，目标函数定义为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1iRDJmMlRDZ0NnbzZBekEwbEFVQTE2aWNXUVJmd0VpYzh5SmxtRXlrWlBqdTdsQ2tJYkpTUjY5dy82NDA?x-oss-process=image/format,png)

各个层参数的梯度为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek0wb21zaWFzejRseFc3RlRkaWFDa2ZBWlA5VFVXcFFVTEZIWllUUjg4b0FEMXgybndKVkN4RTBzZy82NDA?x-oss-process=image/format,png)

其中， ， ， 是 导数。再考虑 L 层有激活函数的神经网络，目标函数定义为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek13czNqMUl0dVhVYjJVaWJscllySXRpYk1ZTmc1MTJpYzdJRkpkTkg5Q0VTOEV5ejR2NktBMWliQVVBLzY0MA?x-oss-process=image/format,png)

其中，

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1YNGtHd2VGZU1XQm4yWlhweWI4djg1c2NlOHhoNHpUVFlKbUlKSjIzSTM2OTdENmlhdElVa1FRLzY0MA?x-oss-process=image/format,png)

并且  。

我们可以发现矩阵形式的求解参数梯度感官上更加的简便明了（公式推导会让人头大，不过推导过程是严格的）。

**2.3 激活函数**

神经网络中激活函数的作用是用来加入非线性因素以此来提高模型的表达能力，因为没有激活函数神经网络训练出来的模型是一种线性模型，这样对于回归和分类任务来说其表达能力不够。

下图为神经网络中常用的激活函数示例，其中蓝色线条为激活函数图像，红色线条为激活函数的导数图像。这些函数分别是 Sigmoid(x)，Tanh(x)，ReLU(x)，Leaky-ReLU(x)。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1UNVpScmV2N1ZuSHdpY2liQkN2Z0JPd2JYbzZkbzMzZEFFV3pFTzRtR0h3OUd6MUFaaWE0SG1ZdVEvNjQw?x-oss-process=image/format,png)

Sigmod 函数定义域为 ，值域为 ，过（0,1）点，单调递增，其函数和导数形式分别为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek11WGdpY2pYaWExTzRqYjgzZWh2TzNpYmVsUmdHNVdLTG9sOHZoaWE0UzlOajJsMmVTa0RaRDhxUXNnLzY0MA?x-oss-process=image/format,png)

Tanh 函数是一种双曲正切函数，定义域为 R，值域为 ，函数图像为过原点严格单调递增，其函数和导数形式分别为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1pYXU2Q3JhOWtSQXBRUkpLY1dvVkRCWDRkRE5PSkFpYUpuZGtHbzFUdlJodlhweUdyZmQ1QjVHQS82NDA?x-oss-process=image/format,png)

ReLU 函数又称线性整流函数其，定义域为 R，值域为 ，其函数和导数形式为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1KTVdtUnYySHhURlpLR0dLYXpjUjlEY0kxSXZpYVVnWWVkMUtNNEJXYkhVOXVkVHRoZWFDczRnLzY0MA?x-oss-process=image/format,png)

Leak-ReLU 函数是 ReLU 函数的改进版本，定义域为 R，值域为 R，其函数和导数形式为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1hNGhDOURKbFJtaFJ2UmE2Mm42Y09LbGdEOW93VUlGT3MzWU5STXVqOFdqdVk1QlVxRU9pYXRBLzY0MA?x-oss-process=image/format,png)

## 2.2 贝叶斯神经网络

在上述频率学派框架内，模型权重不被视为随机变量，而是假设权重是一个确切的值，只是其值尚未知，同时将数据视为随机变量。这似乎有些违反解决问题的原始直觉，因为我们想知道根据手头信息，未知模型的权重是多少。与频率学派不同，贝叶斯统计建模视数据为可获得的信息，而将未知的权重视为随机变量。其基本逻辑是；未知（或潜在）参数被视为随机变量，希望在可观测的训练数据支持下，了解参数的分布。

在 BNN 的“学习”过程中，未知的模型权重可以在已知信息和所观测到的信息基础上推断出来。这是个逆概率问题，可以贝叶斯定理来解决。我们的模型中权重 $ω$ 是隐藏或潜在的变量，通常无法立即观察到其真实分布；而贝叶斯定理允许利用可观测的概率来表示权重的分布，形成以观测数据为条件的权重分布 $p(ω|D)$，我们称之为后验分布。

在训练前，可以观察权重和数据之间的联合分布 $p(ω，D)$ 。该联合分布由我们对潜在变量的先验信念 $p(ω)$ 和我们对模型/似然的选择 $p(D|ω)$ 来定义，
$$
p(\boldsymbol{\omega}, \mathcal{D})=p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega}) \tag{6}
$$
我们对网络结构和损失函数的选择用于定义式 6 中的似然项。例如，对于具有均方误差损失和已知噪声方差的一维同方差回归问题，似然是网络输出平均值的高斯分布。
$$
p(\mathcal{D} \mid \boldsymbol{\omega})=\mathcal{N}\left(\mathbf{f}^{\omega}(\mathcal{D}), \sigma^{2}\right)
$$
在该建模方案下，通常假设来自 $D$ 的所有样本都是独立同分布的（I.I.D.），这意味着可将似然可以被写成数据集中 $N$ 个单独贡献项的乘积，
$$
p(\mathcal{D} \mid \boldsymbol{\omega})=\prod_{i=1}^{N} \mathcal{N}\left(\mathbf{f}^{\omega}\left(\mathbf{x}_{i}\right), \sigma^{2}\right) \tag{7}
$$
在查看任何数据前，应该指定先验分布，以包含关于权重应该如何分布的信念。由于神经网络的黑箱性质，指定一个有意义的先验具有挑战性。在许多实际的频率学派网络训练中，训练后的网络权值较低，且大致集中在零附近。在此经验观察基础上，可以使用具有较小方差的零均值高斯分布作为先验，或使用以零为中心的 spike-slab 先验来鼓励模型中的稀疏性。

在指定先验和似然后，应用贝叶斯定理来得到模型权重上的后验分布：
$$
\pi(\boldsymbol{\omega} \mid \mathcal{D})=\frac{p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega})}{\int p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega}) d \boldsymbol{\omega}}=\frac{p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega})}{p(\mathcal{D})} \tag{8}
$$
后验分布中的分母项称为边际似然（或证据）。该量是关于未知模型权重的常量，对后验进行归一化，以确保后验是有效分布。

根据该后验分布，我们可以预测任何感兴趣的量。预测的形式是关于后验分布的期望
$$
\mathbb{E}_{\pi}[f]=\int f(\boldsymbol{\omega}) \pi(\boldsymbol{\omega} \mid \mathcal{D}) d \boldsymbol{\omega} \tag{9}
$$
所有感兴趣的预测量都是上述形式的期望。无论预测均值、方差还是区间，预测量都是后验的期望值。唯一的不同是应用期望的函数 $f(ω)$ 。然后，预测可被视为函数 $f$  经后验 $π(ω)$ 加权后的平均值。

贝叶斯推断过程围绕着对未知模型权重的边缘化(集成)展开。通过边缘化方法，能够了解模型的生成过程，而不是在频率主义框架中使用的优化方案。通过使用生成模型，预测将以有效的条件概率形式表示。

在上例中假设诸如噪声方差 $σ$ 等参数或者所有先验参数是已知的，但实践中很少会出现此情况，因此需要对这些未知变量进行推断。贝叶斯框架允许对其进行推断，类似于对权重的推断方式；将额外的变量视为潜在变量，分配一个先验分布（有时称为 `超先验分布`），然后对其进行边缘化处理，以找到后验分布。有关如何对 BNN 执行此操作的更多说明，请参考[33，38]。

对于许多感兴趣的模型，后验(式8)的计算仍然很困难。这在很大程度上是由边际似然的计算造成的。对于非共轭模型或存在潜在变量的非线性模型（如前馈神经网络），边际似然几乎没有解析解，对于高维模型的计算更为困难。因此，一般会对后验进行近似。以下部分详细说明了如何在 BNN 中实现近似贝叶斯推断。

## 2.3 贝叶斯神经网络的起源

### 2.3.1 BNN 的起源

根据本调研和之前的调查报告 `[39]`，可以被认为是 BNN 的第一个实例是在 `[40]` 中开发的。该论文通过对神经网络损失函数的统计解释，强调了其主要统计特性，证明了平方误差最小化等价于求高斯函数的最大似然估计（MLE）。更重要的是，通过在网络权重上指定先验，可用贝叶斯定理获得适当的后验。虽然此工作提供了对神经网络贝叶斯观点的关键见解，但没有提供计算边际似然的方法，也就意味着没有提出任何实用的推断方法。`Denker` 和 `LeCun[41]` 对该工作进行了扩展，提供了一种使用拉普拉斯近似进行近似推断的实用方法。

神经网络是一种通用函数逼近器。当单个隐层网络中的参数数目接近无穷大时，可以表示任意函数 `[42，43，44]` 。这意味着只要模型中有足够的参数，就可以用单层神经网络来逼近任何训练数据。与高次多项式回归类似，虽然可以表示任何函数，甚至可以精确匹配训练数据，但神经网络中参数数量的增加会导致过拟合问题。这是神经网络设计中面临的一个基本挑战：模型应该有多复杂？

在 `Gull` 和 `Skilling` `[45]` 的工作基础上，`MacKay` 于 1992 年发表的文章`《Bayesian interpolation》` 展示了如何自然地使用贝叶斯框架处理模型设计和模型比较任务`[46]`。该工作描述了两个层次推断：拟合模型的推断、评估模型适用性的推断。第一层推断是贝叶斯规则用于模型参数更新的典型应用：
$$
P\left(\boldsymbol{\omega} \mid \mathcal{D}, \mathcal{H}_{i}\right)=\frac{P\left(\mathcal{D} \mid \boldsymbol{\omega}, \mathcal{H}_{i}\right) P\left(\boldsymbol{\omega} \mid \mathcal{H}_{i}\right)}{P\left(\mathcal{D} \mid \mathcal{H}_{i}\right)} \tag{10}
$$
其中 $\omega$  是统计模型中的参数， $\mathcal{D}$ 是训练数据， $\mathcal{H}_i$ 是用于此层次推断的第 $i$ 个模型。可以将其描述为：
$$
\text{Posterior}=\frac{\text { Likelihood } \times \text { Prior }}{\text { Evidence }} \notag
$$


注意式 10 中的归一化常数也被称为模型 $\mathcal{H}_i$ 的证据。对于大多数模型，后验的计算仍然很困难，只能采用近似的方法。在该论文中使用了拉普拉斯近似。

虽然需要计算后验超参数，但该论文的另一层次目的是展示模型假设  $\mathcal{H}_i$ 上的后验评估方法。其中后验模型可以简化为：
$$
P\left(\mathcal{H}_{i} \mid \mathcal{D}\right) \propto P\left(\mathcal{D} \mid \mathcal{H}_{i}\right) P\left(\mathcal{H}_{i}\right) \tag{11}
$$
该公式可以理解为：
$$
\text{Model Posterior} \propto \text{Evidence} \times \text{Model Prior} \notag
$$
式 11 中的数据依赖项是该模型的证据。尽管对后验归一化常数有很好的解释，但对于大多数贝叶斯神经网络来说，求证据的分布非常困难。如果假设其为高斯分布，则可以得到证据的拉普拉斯近似：

$$
\begin{align*} 
P\left(\mathcal{D} \mid \mathcal{H}_{i}\right) &=\int P\left(\mathcal{D} \mid \boldsymbol{\omega}, \mathcal{H}_{i}\right) P\left(\boldsymbol{\omega} \mid \mathcal{H}_{i}\right) d \boldsymbol{\omega} \tag{11} \\ 
& \approx P\left(\mathcal{D} \mid \boldsymbol{\omega}_{\mathrm{MAP}}, \mathcal{H}_{i}\right)\left[P\left(\boldsymbol{\omega}_{\mathrm{MAP}} \mid \mathcal{H}_{i}\right) \Delta \omega\right]  \tag{12} \\ 
&=P\left(\mathcal{D} \mid \boldsymbol{\omega}_{\mathrm{MAP}}, \mathcal{H}_{i}\right)\left[P\left(\boldsymbol{\omega}_{\mathrm{MAP}} \mid \mathcal{H}_{i}\right)(2 \pi)^{\frac{k}{2}} \mathrm{det}^{-\frac{1}{2}} \mathbf{A}\right] \tag{13}\\ 
&=\text { Best Likelihood Fit } \times \text { Occam Factor. } 
\end{align*}
$$



这可以解释为对模型证据的一种黎曼近似，具有代表证据峰值的 `最佳似然拟合（Best Likelihood Fit）`，并且 `奥卡姆因子（Occam Factor）` 是由高斯峰值周围的曲率表征的宽度，可以解释为给定模型 $\mathcal{H}_i$ 的后验宽度 $∆ω$ 与先验宽度 $∆ω_0$ 之比，计算公式为：
$$
\text{Occam Factor}  =\frac{\Delta \omega}{\Delta \omega_{0}} \tag{15}
$$
这意味着奥卡姆因子是参数空间中前后变化的比率，下图展示了此概念。有了此表示，一个能够代表大范围数据的复杂模型将拥有更宽的证据，因此具有更大的奥卡姆因子。而简单模型捕获复杂生成过程的能力较弱，但较小范围的数据能够更确定地建模，从而产生较低的奥卡姆因子。这导致了模型复杂性的天然正规化：不必要的复杂模型通常会导致较宽的后验分布，从而导致较大奥卡姆因子和对给定模型较低的证据。同样，信息量较大或信息量较少的先验信息将导致奥卡姆因子降低，从而进一步直观地了解正则化的贝叶斯设置（此即所谓 “贝叶斯推断内置奥卡姆剃刀”）。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021052615491522.webp)

图4：证据在评估不同模型中发挥的作用。简单模型 $\mathcal{H}_1$ 能够以更大的强度预测较小范围的数据，而较复杂的模型 $\mathcal{H}_2$ 能够表示较大范围的数据，尽管概率较低。改编自 `[46，47]`。

使用这种证据框架需要计算边际似然，这是贝叶斯建模中最关键的挑战。考虑到接近边际可能性所需的大量的计算成本，比较许多不同的体系结构可能是不可行的。

尽管如此，证据框架的使用可以用来评估 BNN 的解决方案。对于大多数感兴趣的神经网络结构，目标函数是非凸的，具有许多局部极小值。每一个局部极小都可以看作是推断问题的一个可能解。`MacKay` 以此为动机，使用相应的证据函数来比较来自每个局部最小值的解 `[48]` ，这允许在没有大量计算的情况下评估每个模型方案的复杂性。

### 2.3.2 早期 BNN 的变分推断

机器学习社区在基于优化的问题上一直表现出色。许多最大似然模型（如支持向量机和线性高斯模型）的目标函数都是凸函数，而神经网络的目标函数是高度非凸的，具有许多局部极小值。难以定位全局最小值催生了基于梯度的优化方法，如反向传播 `[3]` 。这种优化方法可以通过 `变分推断（Variational Inference ）` 的方式在贝叶斯上下文中进行实践。

VI是一种近似推理方法，它将贝叶斯推理过程中所需的边际化定义为优化问题[49，50，51]。这是通过假设后验分布的形式并执行优化以找到最接近真实后验的假设密度来实现的。这一假设简化了计算，并提供了一定程度的可操作性。

变分推断一种近似推断方法，它将贝叶斯推断过程中所需的边缘化定义为一个优化问题 `[49，50，51]` 。变分推断首先假设后验分布的形式，然后通过优化方法找到最接近真实后验分布的解。这种假设简化了计算，并提供了可操作性。

假设后验分布 $q_θ(ω)$ 是参数集 $ω$ 上的概率密度函数（被称为 `变分分布`），但被限制为由 $θ$ 参数化的某一分布族。然后调整参数 $\theta$ 以减小变分分布与真实后验$p(ω|D)$ 之间的不相似性。而度量变分分布与真实分布之间之间相似性的手段是正向 `KL 散度`：
$$
K L\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega} \mid \mathcal{D})\right)=\int q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega} \mid \mathcal{D})} d \boldsymbol{\omega} \tag{16}
$$
对于变分推断，式 16 用作相对于参数 $θ$ 的最小化目标函数，可将其扩展为：

$$
\begin{align*} 
\mathrm{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega} \mid \mathcal{D})\right)&= \mathbb{E}_{q}\left[\log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega})}-\log p(\mathcal{D} \mid \boldsymbol{\omega})\right]+\log p(\mathcal{D}) \tag{17}\\ &=\mathrm{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega})\right)-\mathbb{E}_{q}[\log p(\mathcal{D} \mid \boldsymbol{\omega})]+\log p(\mathcal{D}) \tag{18} \\
&=-\mathcal{F}\left[q_{\theta}\right]+\log p(\mathcal{D}) \tag{19}
\end{align*}
$$
其中，$ \mathcal{F}\left[q_{\theta}\right]=-\mathrm{KL}\left(q_{\theta}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega})\right)+\mathbb{E}_{q}[\log p(\mathcal{D} \mid \boldsymbol{\omega})] $。该组合是为了将易处理的项从难处理的对数边际似然中分离出来。现在可以使用反向传播来优化该函数，并且由于对数边缘似然不依赖于变化参数 $θ$ ，因此其关于 $\theta$ 的导数为零。这就只剩下包含变分参数的项，即 $\mathcal{F}[q_θ]$。

式 19 中的符号，特别是包含 $\mathcal{F}[q_θ]$ 的负值，是为强调“与同一个结果不相同但等价的导数”，并与文献保持一致。这一结果不是通过最小化真实分布和近似分布之间的 KL 散度，而是通过近似难以处理的对数边缘似然来获得的。

通过应用Jensen不等式，可以发现 $\mathcal{F}[q_θ]$ 形成了边际对数似然的下界 `[49,52]` 。通过公式 19 注意到： KL 散度严格 ≥ 0 且仅当两个分布相等时才等于零。边缘对数似然 $\log p(\mathcal{D})$ 等于近似后验与真实后验之间的 KL 散度与  $\mathcal{F}[q_θ]$  之和。通过最小化近似后验和真实后验之间的 KL 散度， $\mathcal{F}[q_θ]$  将越来越接近边缘对数似然。因此， $\mathcal{F}[q_θ]$  通常被称为证据下界（ELBO），图 5 可视化地说明了这一点。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021052616535332.webp)

图5：近似后验和真实后验之间的 KL 散度最小化导致的证据下界最大化。当近似后验和真实后验之间的 KL 散度被最小化时，证据下界 $\mathcal{F}[q_θ]$ 收紧到对数证据。因此，最大化证据下界 `ELBO` 等同于最小化 KL 散度。改编自 `[53]` 。

`Hinton` 和 `Van Camp` `[54]` 首次将变分推断应用于 BNN ，试图解决神经网络中的过拟合问题。他们认为，通过使用模型权重的概率观点，其可包含的信息量会减少，并将简化网络。该问题的表述是通过信息论基础，特别是最小描述长度原则，尽管其应用导致了一个相当于变分推断的框架。正如变分推断中常见的那样，使用平均场方法。`平均场变分贝叶斯(MFVB)` 假设后验分布分解感兴趣的参数。对于 `[54]` 中的工作，假设模型权重上的后验分布是独立高斯的因子分解，
$$
q_{\boldsymbol{\theta}}(\boldsymbol{\omega})=\prod_{i=1}^{P} \mathcal{N}\left(w_{i} \mid \mu_{i}, \sigma_{i}^{2}\right) \tag{20}
$$
其中 $P$ 是网络中权重的数量。对于只有一个隐层的回归网络，这个后验的解析解是可用的。能够获得近似解析解是一种理想特性，因为解析解极大减少了执行推断的时间。

这项工作存在几个问题，其中最突出的问题是假设后验因子分解了单个网络权重。众所周知，神经网络中的参数之间存在强相关性。因子分解分布通过牺牲参数之间丰富的相关性信息来简化计算。`Mackay` 在对BNN的早期调研中强调了这一局限性 `[32]`，并提供了对隐藏层输入的预处理阶段如何允许更全面的近似后验分布的洞察力。

`Barber` 和 `Bishop` `[53]` 再次强调了这一局限性，并提出了一种基于变分推断的方法，该方法扩展了 `[54]` 中的工作，允许通过使用 `全秩高斯近似后验` 来捕获参数之间的完全相关性。对于利用 Sigmoid 激活的单隐层回归网络，提供了用于评估`ELBO` 的解析表达式。这通过用适当缩放的误差函数替换 Sigmoid 来实现。

此建模方案的一个问题是参数数量的增加。对于完全协方差模型，参数的数量与网络中的权重数量成二次函数关系。为纠正这一点，`Barber` 和 `Bishop` 对因子分析中经常使用的协方差提出了一种限制形式，
$$
\mathbf{C}=\operatorname{diag}\left(d_{1}^{2}, \ldots, d_{n}^{2}\right)+\sum_{i=1}^{s} \mathbf{s}_{i} \mathbf{s}_{i}^{T} \tag{21}
$$
其中，$diag$ 运算符从大小为 $n$ 的向量 $d$ 创建对角线矩阵，其中 $n$ 是模型中权重的数量。然后，该表单随网络中隐藏单元的数量线性扩展。

这些工作为如何将显著的反向传播方法应用于挑战贝叶斯问题提供了重要的见解。使得这两个研究领域的特性可以合并，并提供名义上单独看到的好处。现在可以使用NNS在概率意义上处理大量数据集的复杂回归任务。

尽管这些方法提供了洞察力，但这些方法也有局限性。`Hinton` 、 `Van Camp` 、 `Barber` 和 `Bishop` 的工作都集中在发展一种封闭形式的网络表示。这种分析处理能力对网络施加了许多限制。如前面所讨论的，`[54]` 假设后验因子分解超过单个权重，它不能捕获参数中的任何相关性。在 `[53]` 中捕获了协方差结构，尽管作者将他们的分析限制在使用 Sigmoid 激活函数(该函数由误差函数很好地近似)的情况下，该函数很少在现代网络中使用，因为梯度的幅度较低。这两种方法共有的一个关键限制是单个隐含层网络的限制。

如前所述，NN可以通过添加额外的隐藏单元来任意地逼近任何函数。对于现代网络，经验结果表明，通过增加网络中的隐含层数，可以用更少的隐含单元来表示类似的复杂函数。这就产生了“深度学习”一词，其中深度指的是隐藏层的数量。当试图近似各层之间的全部协方差结构时，减少权重变量的数量尤其重要。例如，可以捕获单个层内的隐藏单元之间的相关性，同时假设不同层之间的参数是独立的。这样的假设可以显著减少相关参数的数量。随着现代网络在许多层上拥有数以亿计的权重(这些网络只能提供点估计)，开发超出单层的实用概率解释的需求是必不可少的。

### 2.3.3 BNN 的混合蒙特卡罗方法

在这一点上，值得反思一下实际感兴趣的数量。到目前为止，重点一直放在寻找后验的良好近似上，尽管后验的准确表示通常不是最终设计要求。感兴趣的主要数量是预测时刻和间隔。我们希望在提供信心信息的同时做出良好的预测。我们强调后验的原因是预测矩和区间都是根据后验π(ω|D)13的期望值来计算的。该期望值列在方程式9中，为方便起见在这里重复

贝叶斯神经网络的重点是寻找良好的后验分布近似值上，预测值和区间都是作为后验 的期望值来计算的，其中精确的预测依赖于对难以处理的后验概率的精确近似。具体的计算公式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1OdnFqUzNDWGMyVEw2MXhoTU1pY09nTFlEb2s5a0dyYzEyRXFwVER2VVZDU0tiNU1PeDZkeVRRLzY0MA?x-oss-process=image/format,png)

上面的积分公式的求解很困难，以前的方法是通过基于优化的方案，但优化方法中设置的限制通常会导致预测值不准确，所以基于优化的方案可以提供不准确的预测量。为了在有限的计算资源下做出准确的预测，通过使用马尔可夫链蒙特卡罗（MCMC）方法来求解上积分。

MCMC 是一种可以从从任意和难以处理的分布中进行采样的通用方法，会有如下公式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1KNEpQR1hKQnVEREswaWNMSWxVdDN4UlBId0VFeVlmdGV0SXVKSXIyaWNpYUFOMW5sR094T1ZsdmcvNjQw?x-oss-process=image/format,png)

传统的 MCMC 方法表现出一种随机游走行为，即序列是随机产生的。由于 BNNs 后验函数的复杂性和高维性，这种随机游走行为使得这些方法不适合在任何合理的时间内进行推断。为了避免随机行走行为，本文采用混合蒙特卡洛（HMC）方法用于将梯度信息合并到迭代行为中。

对于 BNNs，首先引入超先验分布 来模拟先验参数精度，其中先验参数服从高斯先验。 上的先验服从伽马分布，并且它是条件共轭的，这就使得吉布斯抽样可以用于对超参数进行推断，然后可以使用 HMC 来更新后验参数，最后从联合后验分布 进行取样。

## 2.4 现代贝叶斯神经网络

在Neal，MacKay和Bishop于90年代提出的早期工作之后，对BNN进行的研究要少得多。这种相对停滞在大多数神经网络研究中都可以看到，这在很大程度上是由于训练神经网络的计算需求很高。神经网络是能够以任意精度捕获任何函数的参数模型，但是准确地捕获复杂函数需要具有许多参数的大型网络。即使从传统的频域观点来看，训练如此庞大的网络也变得不可行，而为了研究信息量更大的贝叶斯网络，计算需求也大大增加。

考虑到网络的大规模性，强大的推断能力通常需要建立在大数据集上。对于大数据集，完全对数似然的评估在训练目的上变得不可行。为了解决这一问题，作者采用了随机梯度下降（SGD）方法，利用小批量的数据来近似似然项，这样变分目标就变成：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek00Q2JnN0ZFQWFjTElpYUFiaWE3bVUwNG5FVUZlRldyVURvWXAzb0NIaWNhdFNLdzRpYzI0ZE9FY2hBLzY0MA?x-oss-process=image/format,png)

其中 ，每个子集的大小为 。这为在训练期间使用大型数据集提供了一种有效的方法。在通过一个子集 后，应用反向传播来更新模型参数。SGD 是使用变分推断方法训练神经网络和贝叶斯神经网络的最常用方法。

Graves 在 2011 年发表了 BNN 研究的一篇重要论文《Practical variational inference for neural networks》。这项工作提出了一个 MFVB 处理使用因子高斯近似后验分布。这项工作的主要贡献是导数的计算。变分推断的目标可以看作是两个期望值的总和如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1xdmdjdUhYd2I0VTJYam95TjA4d3hTQ1RWVjluVHJPaWFEZTlJT1lJWWYyMjZpYlNjTTYzcTRkZy82NDA?x-oss-process=image/format,png)

Opper 在 2009 年发表的《The variational gaussian approximation revisited Neural computation》中提出了利用高斯梯度特性来对参数进行更新操作，具体的如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1NczJtSlBYaWFyN0dDbUJxUWIxTzR0RWppYnJ5YlluQzI1c2c1VUZFNU8xeVdKdWljdFA1NVE5T1EvNjQw?x-oss-process=image/format,png)

上面两个公式用于近似平均参数和方差参数的梯度，并且该框架允许对 ELBO 进行优化，并且可以推广到任何的对数损失参数模型中。

已知分数函数估计依赖于对数导数性质，具体公式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1FNkIxMUtNb2gzcUhMNDRTMXpscDA2d2JlbXppYVZaaEdCWU1TMTBXTWNybmFpYWlhNG0wWnY5b2cvNjQw?x-oss-process=image/format,png)

利用这个性质，可以对一个期望的导数形成蒙特卡罗估计，这在变分推断中是经常被使用，具体的推导过程如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek11ZmhTaWM4Um5RdmliRUJNZ0FKVTFDbUVSMXNaajZqdUhpYzdDZ2VHVkRWUFZNTXB0aWN1aG5wb2N3LzY0MA?x-oss-process=image/format,png)

变分推断的第二种梯度估计方法是通过路径导数估计值法。这项工作建立在“重新参数化技巧”的基础上，其中一个随机变量被表示为一个确定性和可微的表达式，具体形式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1IdHYxb1R4RVhHYXNQT2tXdEFHaWJxbTYzaWJCcldmS2JSWkFpYWxzcjRJUEdwaWFFdXphY0RDbG93LzY0MA?x-oss-process=image/format,png)

其中 和 表示的是哈达玛积（Hadamard product）。使用这种方法可以有效地对期望值进行蒙特卡罗估计，具体的计算公式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1pYUxpYkRqc1ViYnZaU2ljWTBlVG9nd1BYaWNDZldBMHNpYUpKd2ljYmpIUVNNc3hpYVlXV25HNjRpYm9SUS82NDA?x-oss-process=image/format,png)

由于上式是可微的，所以可以使用梯度下降法来优化这种期望近似。这是变分推断中的一个重要属性，因为变分推断的目标的对数似然的期望值的求解困难很大。与分数函数估计量相比，路径估计值法更有利于降低方差。

Blundell 等人在论文《Bayes by Backprop》中提出了一种在 BNNs 中进行近似推断的方法。该方法利用重参数化技巧来说明如何找到期望导数的无偏估计。其期望导数的具体形式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1VSTRZajhCSzFnRnFYUWljaWJuanhFQmlheENqZnNKWjNZcHFHbWd5amlhTUU1cXBROWVFd3UwV29RLzY0MA?x-oss-process=image/format,png)

在贝叶斯的反向传播的算法中，函数 设为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek0yaFdJZ3RINjFzU2tHNWM2TjMwWTRpY3pRWG9UTkJNZ2lhaGxoR0xFZkZ6SFRyaWF2Um0yZllpYmZBLzY0MA?x-oss-process=image/format,png)

其中 可以看作是期望值的自变量，它是下界的一部分。

假设全因子高斯后验函数 ，其中 用于确保标准差参数为正。由此，将网络中的权重 的分布重新参数化为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek11aFAyRTc3cklieXVLNFJKbTZOaWJjQ0RKeUNvN2tCd3NERmR2WWdiaWNTMkxtVDVYS3ZiaWMwM1EvNjQw?x-oss-process=image/format,png)

在该 BNN 中，可训练参数为 和 。由于采用全因子分布，则近似后验概率的对数可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek0xMDZNRzVSV1M0eUd1a21sZFVLeXhRaGVqNW1MRnZpY2lhNVVVdWtqaWJaQ2hSTG9LMnpvMmVsU2cvNjQw?x-oss-process=image/format,png)

综合上面提到的贝叶斯的反向传播算法的细节，会有如下完整的算法流程图。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9DeFBpY2lhcE8yVVdiWDdXMDVmSGQ0UGlhWTBvaFFCTVNvaWJlMTRPektiQnBnaWFuQTdhUlFGMlJZUktNMjgyUWljdW5Sc3NXY3FkcWNwR3lscDBOQ2licDJUOEEvNjQw?x-oss-process=image/format,png)

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ESGVmbFdUZFRJQkVpYVBNQmFLTXFrUnJWOURjZGFGUDNWeXhTY0hDdWFCeVo1OHZ3OUljNHMzaWFjVWljOFhHbmtZODZsSDg2MmJaWmVpY2cvNjQw?x-oss-process=image/format,png)



## 2.5 贝叶斯神经网络的高斯过程特性

Neal[38]还给出了推导和实验结果，以说明对于只有一个隐层的网络，当隐藏单元的数量接近无穷大时，会出现高于网络输出的高斯过程(GP)，并且将高斯先验置于参数22之上。图6说明了这一结果。



下图为当在参数上放置高斯先验时，随着网络规模的增加，先验在输出上诱导的图示。其中图中的每个点对应于一个网络的输出，参数从先验分布中进行采样。对于每个网络，隐藏单元的数量是图（a）对应着 1 个单元，图（b）对应着 3 个单元，图（c）对应着 10 个单元，图（d）对应着 100 个单元。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek12aWJpYmoxRGUzSmVvRVpOZGpyMWh3aWJYdUkycTRVamliRUxNUU5Qem9GMk9wblpKV1hzaWJqelpLZy82NDA?x-oss-process=image/format,png)

由此可知在中心极限定理下，随着隐层数 N 逐渐增大，输出服从高斯分布。由于输出被描述为基函数的无穷和，因此可以将输出看作高斯过程。高斯过程和具有单个隐藏层工作的无限宽网络之间的关系最近已扩展到深度神经网络的当中，这一联系的识别激发了 BNNs 的许多研究工作。

高斯过程提供了可靠的不确定性估计、可解释性和鲁棒性。但是高斯过程提供这些好处的代价是预测性能和随着数据集大小的增加所需的大量计算资源。高斯过程和 BNNs 之间的这种联系促使了两种建模方案的合并；既能维持神经网络的预测性能和灵活性，同时结合了高斯过程的的鲁棒性和概率性。

最近的研究已经确定了高斯过程属性不限于 MLP-BNN，而且在卷积中也可以应用该属性。因为 CNN 可以实现为 MLP，其结构在权重中被强制执行。Vander Wilk 等人在 2003 年发表的论文《Convolutional gaussian processes》中提出了卷积高斯过程，它实现了一种类似于 CNN 中的基于 patch 的操作来定义 GP 先验函数。

如下图所示，分析显示了高斯过程在预测偏差和方差方面的对比性能。用 Backprop 和一个因式高斯近似后验概率对 Bayes 模型进行训练，在训练数据分布的情况下，虽然训练数据区域外的方差与高斯过程相比显著低估，但预测结果是合理的。具有标度伯努利近似后验的 MC-Dropout 通常表现出更大的方差，尽管在训练数据的分布中保持不必要的高方差。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9XcnNrRXlJTmZIMUtYd1g0R0pMYlpPN0NTcHdneTJuUmZpY3ZkSTd2T2JUZnJjMzR5aWF2Njg2cnoxRENab29pYUpWSlZpYW1EYU9yaE1EeUtPZnQxQ2FoQmcvNjQw?x-oss-process=image/format,png)

## 2.6 当前贝叶斯神经网络的局限性

虽然人们已经付出了很大的努力来开发在NNS中执行推理的贝叶斯方法，但这些方法有很大的局限性，文献中还存在许多空白。一个关键的限制是严重依赖VI方法。在VI框架内，最常用的方法是平均场方法。MFVB通过强制参数之间独立的强假设，提供了一种表示近似后验分布的方便方法。这一假设允许使用因式分解分布来近似后验分布。这种独立假设大大降低了近似推理的计算复杂度，但代价是概率精度。

# 3 现代贝叶斯神经网络的比较

## 3.1 通用贝叶斯神经网络

从文献综述来看，BNN中两种重要的近似推理方法是Backprop的Bayes[76]和MC Dropout[85]。这些方法被认为是BNN中最有前途、影响最大的近似推理方法。这两种VI方法都足够灵活，可以使用SGD，从而使部署到大型、实用的数据集成为可能。鉴于这些方法的突出之处，有必要对这些方法进行比较，看看它们的表现如何。

为了比较这些方法，进行了一系列简单的同方差回归任务。对于这些回归模型，概率用高斯表示。有了这个，我们就可以写出未归一化的后验是，

## 3.2 卷积贝叶斯神经网络

虽然 MLP 是神经网络的基础，但最突出的神经网络架构是卷积神经网络。这些网络在具有挑战性的图像分类任务方面表现出色，其预测性能远远超过先前基于核或特征工程的方法。CNN 不同于典型的 MLP，它的应用是一个卷积型的算子，单个卷积层的输出可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1GWEFQN2N3QTdjSlhrOE90VHpuVzlyOG9kdG9MUWJXWHNiS2FpYUdpYmozWktwc25TaWFsQnFzUWcvNjQw?x-oss-process=image/format,png)

其中 是非线性激活， 表示类似卷积的运算。这里输入 X 和权重矩阵 W 不再局限于向量或矩阵，而是可以是多维数组。可以得到证明的是 cnn 可以编写成具有等效 MLP 模型，允许使用优化的线性代数包进行反向传播训练。

在现有研究方法的基础上，发展了一种新型的贝叶斯卷积神经网络（BCNN）。假设卷积层中的每个权重是独立的，允许对每个单独的参数进行因子分解，其中 BCNN 输出概率向量由 Softmax 函数表示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1KRGJMVTVLckRJeWNKdjhVa01oU3ZpYlVlZ0g1NTdmZFFyMGliOFpaWlFRQ2lhYUQ5UUZBektzSWcvNjQw?x-oss-process=image/format,png)

非标准化的后验分布可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek15NDRONzI3cW1TZHJCbTdmbzVOaWFTREdHVWN3eG5BQUEyV3dTZmx5RVlKMEdJcWliRWljMVpod2cvNjQw?x-oss-process=image/format,png)

作者做了相关的对比实验，BCNN 使用了流行的 LeNet 架构，并利用 BCNN 的平均输出进行分类，并使用可信区间来评估模型的不确定性。

在 MNIST 数据集中的 10000 个测试图像上，两个网络的总体预测性能显示出了比较好的性能。BCNN 的测试预测精度为 98.99%，香草网络的预测精度略有提高，预测精度为 99.92%。

虽然竞争性的预测性能是必不可少的，但 BCNN 的主要优点是可以提供有关预测不确定性的有价值的信息。从这些例子中，可以看到这些具有挑战性的图像存在大量的预测不确定性，这些不确定性可以用于在实际场景中做出更明智的决策。

## 3.3 循环贝叶斯神经网络



# 4 结论

在这份报告中，典型的NNS和特别模型设计中过度自信的预测所产生的问题已经被阐明。贝叶斯分析已经被证明提供了丰富的理论来解决这些挑战，尽管对于任何感兴趣的BNN来说，精确的计算仍然是分析和计算上的难题。在实践中，必须依靠近似推理才能得到精确的后验近似。

BNN中的许多近似推理方法都围绕着MFVB方法。这为优化W.r.t变分参数提供了一个易于处理的下限。这些方法因其相对易用性、预测均值的准确性和可接受的诱导参数数目而具有吸引力。尽管如此，文献调查和实验结果表明，在完全分解的MFVB方法中所做的假设导致了过度自信的预测。结果表明，这些MFVB方法可以推广到更复杂的模型，如CNN。实验结果表明，对于图像分类任务，预测性能与点估计CNN相当。贝叶斯CNN能够为预测提供可信的区间，这些预测被发现是对难以分类的数据点的高度信息性和直观性的不确定性度量

这篇综述和这些实验突出了贝叶斯分析解决机器学习社区中常见挑战的能力。这些结果还突显了当前用于BNN的近似推理方法的不足，并且可能提供不准确的方差信息。还需要进一步研究，不仅要确定这些网络是如何运行的，而且要确定现代大型网络如何才能实现准确的推理。将MCMC等精确推理方法扩展到大数据集的方法将允许使用更有原则性的方法来执行推理。MCMC提供了评估收敛和推理质量的诊断方法。对VI的类似诊断将允许研究人员和实践者评估他们假设的后验的质量，并告知他们改进这一假设的方法。实现这些目标将使我们能够获得精确的后验近似。由此，我们将能够充分确定我们的模型知道什么，也可以确定他们不知道什么。

