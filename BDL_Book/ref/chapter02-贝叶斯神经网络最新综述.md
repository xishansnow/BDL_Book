---
layout:     post
title:      "   贝叶斯神经网络概述与现状调研  "
description:   "  贝叶斯深度学习 "
date:       2021-05-25 10:00:00
author:     "西山晴雪"
mathjax:    true
categories: 
    - [贝叶斯统计, 贝叶斯深度学习]
    - [GeoAI, 贝叶斯方法]
tags:
    - 贝叶斯统计
    - 贝叶斯深度学习
    - 感知组件
    - 任务组件
    - 铰链组件
    - 综述
    - 概率图模型
    - 深度学习
---

# 贝叶斯神经网络概述与现状调研（Bayesian Neural Networks: An Introduction and Survey）

【作者】Ethan Goan, Clinton Fookes， Queensland University of Technology
【原文】https://arxiv.org/abs/2006.12024
【摘要】神经网络已经为许多具有挑战性的机器学习任务提供了最先进的结果，例如计算机视觉、语音识别和自然语言处理领域的检测、回归和分类等。尽管其取得了成功，但大多是在频率学派框架内实施的，这意味着它们无法对预测中的不确定性进行推断。本文介绍了贝叶斯神经网络及其实现的开创性研究。对不同的近似推断方法进行了比较，并强调了未来如何在现有方法基础上进行改进。

**1 概述**


长期以来，仿生学一直是技术发展的基础。科学家和工程师反复使用物理世界的知识来模仿自然界对经过数十亿年演变而来的复杂问题的优雅解决方案。生物仿生学在统计学和机器学习中的一个重要例子是感知器的发展[1]，它提出了一个基于神经元生理学的数学模型。机器学习团体已经使用这一概念来开发高度互连的神经元阵列的统计模型，以创建神经网络(NNS)。

虽然NNS的概念早在几十年前就已为人所知，但这些网络的应用直到最近才开始显现出来。神经网络研究和开发的停滞很大程度上是由于三个关键因素：缺乏足够的算法来训练这些网络，训练复杂网络所需的大量数据，以及训练过程中所需的大量计算资源。1986年，[3]引入了反向传播算法来解决这些网络的有效训练问题。虽然有了一种有效的培训手段，但新网络的规模不断扩大，仍然需要相当多的计算资源。这个问题在[4，5，6]中得到了解决，其中表明通用GPU可以有效地执行培训所需的许多操作。随着数字硬件的不断进步，能够捕捉和存储真实世界数据的传感器数量不断增加。通过高效的训练方法、改进的计算资源和庞大的数据集，复杂神经网络的训练变得真正可行。

在绝大多数情况下，神经网络都是在频域内使用的；用户使用可用数据定义网络结构和成本函数，然后对其进行优化，以允许我们获得点估计预测。这种对NNS的解释产生了问题。增加参数的数量(在机器学习文献中通常称为权重)或模型的深度会增加网络的容量，使其能够表示具有更强非线性的函数。这种容量的增加允许使用NNS处理更复杂的任务，尽管当应用频域方法时，NNS很容易与训练数据过度拟合。使用大型数据集和正则化方法(如寻找MAP估计)可以限制网络学习的函数的复杂性，并有助于避免过度拟合。

神经网络已经为许多机器学习和人工智能(AI)应用提供了最先进的结果，例如图像分类[6，7，8]，目标检测[9，10，11]和语音识别[12，13，14，15]。其他网络，如DeepMind[16]开发的AlphaGo模型，强调了NNS在开发人工智能系统方面的潜力，吸引了对这些网络的开发感兴趣的广泛受众。随着NNS性能的不断提高，某些行业对NNS的开发和采用的兴趣变得更加突出。NN目前用于制造[17]、资产管理[18]和人机交互技术[19，20]。

自从NNS在工业中部署以来，已经发生了许多事件，这些系统中的故障导致模型行为不道德和不安全。这包括对边缘化群体表现出相当大的性别和种族偏见的模型[21，22，23]，或者对导致生命损失的更极端的案例[24，25]。神经网络是一种统计黑盒模型，这意味着决策过程不是基于定义良好和直观的协议。相反，决策是以一种无法解释的方式做出的，希望合理的决策将基于训练数据2中提供的先前证据做出。因此，在社会和安全关键环境中实施这些系统会引起相当大的伦理关注。欧盟发布了一项新的规定3，实际上规定用户有权对人工智能系统所做的决定作出“解释”[26，27]。由于不清楚它们的操作或设计的原则方法，其他领域的专家仍然对采用当前技术感到担忧[28，29，30]。这些限制激发了对可解释人工智能领域的研究努力[31]。

适当设计NNS需要充分了解它们的能力和局限性；在部署之前找出它们的不足之处，而不是在这些悲剧之后调查这些局限性的现行做法。由于NNS是一个统计黑匣子，对决策过程的解释和解释是当前理论所不能理解的。普通神经网络的频域观点提供的这种缺乏解释和过度自信的估计，使得它们不适合于诸如医疗诊断和自动驾驶汽车等高风险领域。贝叶斯统计提供了一种自然的方式来推断预测中的不确定性，并可以洞察这些决策是如何做出的。

图1比较了用于执行回归的贝叶斯方法和简单神经网络的方法，并说明了测量不确定性的重要性。虽然这两种方法都在需要外推的训练数据的范围内执行得很好，但是概率方法提供了函数输出的完全分布，而不是由神经网络提供的点估计。概率方法提供的输出分布允许开发可靠的模型，因为它们可以识别预测中的不确定性。鉴于神经网络是生成人工智能系统最有前途的模型，我们同样可以信任他们的预测是很重要的。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526093343_31.webp)

图1：在紫色区域没有训练数据的回归任务中，神经网络与传统概率方法的比较。(A)使用具有2个隐藏层的神经网络的回归输出；(B)使用高斯过程框架的回归，灰色条表示±2标准。从期望值开始。

贝叶斯观点使我们能够解决NNS目前面临的许多挑战。为此，在网络参数上放置一个分布，然后将得到的网络称为贝叶斯神经网络(BNN)。BNN的目标是拥有一个高容量的模型，该模型展示了贝叶斯分析的重要理论好处。最近的研究已经调查了如何将贝叶斯近似应用于实际中的神经网络。这些方法的挑战在于部署在合理的计算约束内提供准确预测的模型4。

这份文件的目的是提供一个容易理解的BNN的介绍，伴随着对该领域的开创性工作和实验的调查，以激发对当前方法的能力和局限性的讨论。对贝叶斯和机器学习文献中与BNN相关的所有研究项目的调查可能会填满多本教科书。因此，这项调查中的项目只是为了让读者了解激励他们进行研究的主要叙述。同样，许多关键结果的推导都被省略了，最后的结果被列出，并附有对原始来源的引用。我们鼓励受这一激动人心的研究领域启发的读者参考以前的调查：[32]调查了BNN的早期发展，[33]讨论了对NNS进行全面贝叶斯处理的细节，以及[34]调查了近似贝叶斯推理在现代网络结构中的应用。


本文档应该适合统计领域的所有人，尽管感兴趣的主要读者是那些更熟悉机器学习概念的人。尽管新机器学习学者的开创性参考几乎等同于贝叶斯文本[2，35]，但在实践中，许多现代机器学习和贝叶斯统计研究之间存在分歧。希望这项调查将有助于突出BNN中的一些现代研究与统计学之间的相似之处，强调概率观点在机器学习中的重要性，并鼓励机器学习和统计学领域未来的合作/协调。

## 2 现状调查





下一代神经网络的演化方向是什么？最近两年在北京举行的智源大会都谈到了这个问题，可能性的一个答案是贝叶斯神经网络，因为它可以对已有的知识进行推断。逻辑推理作用就是可以对已有的知识进行延伸扩展。

举个例子，如果询问训练完善的 AI 模型的一个问题，“在乌克兰，新西兰，新加坡，阿尔及利亚这四个国家里，哪一个国家位于中国的最西边”，这个问题的难点就在于那个“最”字，如果是传统的 AI 模型可能会蒙圈，因为乌克兰和阿尔及利亚都是在中国的西边，因为现有的训练的知识并不足以告诉它哪个是最西边，经过 BNN（贝叶斯神经网络）训练的模型可能会从经纬度，气温等其他信息进行推断得出一个阿尔及利亚在中国的最西边这个答案。

BNN 的最新进展值得每个 AI 研究者紧密关注，**本文就是一篇新鲜出炉的关于 BNN 的综述**，为了方便读者的阅读，我按照自己的节奏和想法重新梳理了一下这篇文章。


**神经网络**

先回顾一下传统神经网络，论文限于篇幅的原因有一些重要的细节没有展开，而且我一直觉得神经网络中一个完善的形式应该是通过矩阵的形式表现出来，同理矩阵形式 BP 反向传播原理也能一目了然。

**2.1 标量形式的神经网络**

下图为标量形式的神经网络，并且为了说明方便不考虑偏置项。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094051_d8.webp)

给定一个训练样本 ，假设模型输出为 ，则均方误差为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094055_a8.webp)

根据梯度下降法更新模型的参数，则各个参数的更新公式为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094059_5c.webp)

链式法则求解 会有如下推导形式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094102_97.webp)

链式法则求解 会有如下推导形式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094106_95.webp)

可以发现标量视角下的神经网络更新参数求解梯度会给人一种很混乱的感觉。

### **2.2 矩阵形式的神经网络**

下图为 3 层不考虑偏置项的全连接神经网络示意图：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094110_a7.webp)

上图可以描述为如下公式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094115_32.webp)

损失函数如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094120_81.webp)

优化的目标函数为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094124_b5.webp)

其中， 表示的权重矩阵， 为隐层向量。

**2.2.1 随机梯度**

采用随机梯度下降法求解优化深度神经网络的问题，如下式所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094127_8b.webp)

上式中，主要的问题是在于计算 ，通常采用的方法是链式法则求导。而反向传播就是一种很特殊的链式法则的方法。反向传播非常有效的避免大量的重复性的计算。

**2.2.2 无激活函数的神经网络**

L 层神经网络的无激活函数的目标函数定义为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094131_b8.webp)

则各个层的梯度有如下形式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094134_49.webp)

其中， 。

**2.2.3 含有激活函数的神经网络**

首先，考虑 2 层的有激活函数的神经网络，目标函数定义为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094142_0c.webp)

各个层参数的梯度为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094147_2d.webp)

其中， ， ， 是 导数。再考虑 L 层有激活函数的神经网络，目标函数定义为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094150_26.webp)

其中，

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094155_02.webp)

并且  。

我们可以发现矩阵形式的求解参数梯度感官上更加的简便明了（公式推导会让人头大，不过推导过程是严格的）。

**2.3 激活函数**

神经网络中激活函数的作用是用来加入非线性因素以此来提高模型的表达能力，因为没有激活函数神经网络训练出来的模型是一种线性模型，这样对于回归和分类任务来说其表达能力不够。

下图为神经网络中常用的激活函数示例，其中蓝色线条为激活函数图像，红色线条为激活函数的导数图像。这些函数分别是 Sigmoid(x)，Tanh(x)，ReLU(x)，Leaky-ReLU(x)。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094159_e3.webp)

Sigmod 函数定义域为 ，值域为 ，过（0,1）点，单调递增，其函数和导数形式分别为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094203_fb.webp)

Tanh 函数是一种双曲正切函数，定义域为 R，值域为 ，函数图像为过原点严格单调递增，其函数和导数形式分别为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094207_89.webp)

ReLU 函数又称线性整流函数其，定义域为 R，值域为 ，其函数和导数形式为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094210_73.webp)

Leak-ReLU 函数是 ReLU 函数的改进版本，定义域为 R，值域为 R，其函数和导数形式为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094215_ce.webp)



## 3 DNN和BNN的区别

BNN 跟 DNN 的不同之处在于，其权重参数是随机变量，而非确定的值，它是通过概率建模和神经网络结合起来，并能够给出预测结果的置信度。其先验用来描述关键参数，并作为神经网络的输入。

神经网络的输出用来描述特定的概率分布的似然。通过采样或者变分推断来计算后验分布。这对于很多问题来说非常关键，由于 BNN 具有不确定性量化能力，所以具有非常强的鲁棒性。如下图所示为 DNN 和 BNN 的之间的差异：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094316_90.webp)

如下图所示贝叶斯神经网络的回归与简单的神经网络方法的回归的进行了比较，并说明了测量不确定度的重要性。

虽然这两种方法在训练数据的范围内都表现良好，在需要外推法的情况下，概率方法提供了函数输出的完整分布，而不是由神经网络提供的点估计。概率方法提供的输出分布允许开发可信的模型，因为它们可以识别预测中的不确定性。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094321_6c.webp)

贝叶斯模型可以通过预测器的集合来捕捉数据驱动模型的内在认知下的不确定性；它通过将算法参数（以及相应的预测）转化为随机变量来实现。

在神经网络中，对于具有输入 和网络权重参数 的神经网络 ，则从网络权重 的先验度量开始。通过似然 评估权重为 的网络与数据 的拟合度。

贝叶斯推理通过 Bayes 定理将似然和先验相结合，得到权重空间 的后验测度。神经网络的标准训练可以看作是贝叶斯推理的一种近似。

对于 NNs 这样的非线性/非共轭模型来说，精确地获得后验分布是不可能的。后验分布的渐近精确样本可以通过蒙特卡洛模拟来获得，对于一个新输入的样本 贝叶斯预测都是从 n 个神经网络的集合中获得的，每个神经网络的权重都来自于其后验分布 ：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094329_c5.webp)

论文中这这部分作者没有详细展开说明，不过可以从公式可以推测出来 表示的是已知训练数据集的情况下，贝叶斯神经网络给出的样本 的预测， 表示是不同权重参数的给出预测的期望，然后用蒙特卡洛模拟将期望形式转化成离散的平均加和的形式。

## **4 BNN的起源**

MacKay 在 1992 年公布的文章《Bayesian interpolation》展示了贝叶斯框架如何自然地处理模型设计和通用统计模型比较的任务。在该工作中，描述了两个层次的推理：拟合模型的推理和评估模型适用性的推理。第一层推理是贝叶斯规则用于模型参数更新的典型应用。如下公式

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094353_dc.webp)

其中 是一般统计模型中的一组参数， 是训练数据， 是用于这一水平推断的第 i 个模型。然后将其描述为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094359_41.webp)

其中后验模型可以简化为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094403_66.webp)

对这个公式可以简单的理解为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094407_58.webp)

通过拉普拉斯近似可以得到如下推导：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094411_20.webp)

这可以解释一种黎曼近似，具有代表证据峰值的最佳似然拟合，并且 Occam 因子是由高斯峰值周围的曲率表征的宽度。其中 Occam 因子的计算公式为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094423_8f.webp)

这意味着 Occam 因子是似然参数空间中前后变化的比率。下图以图形方式演示了这个概念。有了这种表示，一个能够表示大范围数据的复杂模型将具有更广泛的证据，从而具有更大的 Occam 因子。

一个简单的模型在捕捉复杂的生成过程方面的能力会较低，但是较小范围的数据将能够以更大的确定性进行建模，从而降低 Occam 因子。这导致了模型复杂性的自然正则化。

一个不必要的复杂模型通常会导致较大的后验概率，从而导致较大的 Occam 因子和较低的证据。类似地，一个广泛或信息量较少的先验将导致 Occam 因子的减少，从而为正则化的贝叶斯设置提供了进一步的直觉。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094429_16.webp)

使用这种证据框架需要计算边际似然，这是贝叶斯建模中最关键的挑战。考虑到接近边际可能性所需的大量的计算成本，比较许多不同的体系结构可能是不可行的。

尽管如此，证据框架的使用可以用来评估 BNN 的解决方案。对于大多数感兴趣的神经网络结构，目标函数是非凸的，具有许多局部极小值。每一个局部极小都可以看作是推理问题的一个可能解。

## **5 BNN的早期变分推理**

变分推理一种近似推理方法，它将贝叶斯推理过程中所需的边缘化作为一个优化问题。这是通过假设后验分布的形式来实现的，并进行优化以找到最接近真实的后验分布。这种假设简化了计算，并提供了一定程度的可操作性。

假定的后验分布 是参数 集上的一个合适的密度，它仅限于由 参数化的某一类分布。然后调整此变分分布的参数，以减少变分分布与真实后验分布 之间的差异。

度量变分推理相似性的方法通常是变分分布和真分布之间的正向 KL 散度为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094454_bb.webp)

对于变分推理可以将 KL 散度扩展为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094459_3f.webp)

如图所示说明了如何将近似分布和真实后验之间的关系示意图，由此可知通过近似对数似然逼近近似分布与真实分布之间的 KL 散度。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094502_cd.webp)

## 6 BNN的蒙特卡洛方法

贝叶斯神经网络的重点是寻找良好的后验分布近似值上，预测值和区间都是作为后验 的期望值来计算的，其中精确的预测依赖于对难以处理的后验概率的精确近似。具体的计算公式如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094522_1a.webp)

上面的积分公式的求解很困难，以前的方法是通过基于优化的方案，但优化方法中设置的限制通常会导致预测值不准确，所以基于优化的方案可以提供不准确的预测量。为了在有限的计算资源下做出准确的预测，通过使用马尔可夫链蒙特卡罗（MCMC）方法来求解上积分。

MCMC 是一种可以从从任意和难以处理的分布中进行采样的通用方法，会有如下公式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094526_46.webp)

传统的 MCMC 方法表现出一种随机游走行为，即序列是随机产生的。由于 BNNs 后验函数的复杂性和高维性，这种随机游走行为使得这些方法不适合在任何合理的时间内进行推理。为了避免随机行走行为，本文采用混合蒙特卡洛（HMC）方法用于将梯度信息合并到迭代行为中。

对于 BNNs，首先引入超先验分布 来模拟先验参数精度，其中先验参数服从高斯先验。 上的先验服从伽马分布，并且它是条件共轭的，这就使得吉布斯抽样可以用于对超参数进行推理，然后可以使用 HMC 来更新后验参数，最后从联合后验分布 进行取样。

## **7 现代BNN模型**

考虑到网络的大规模性，强大的推理能力通常需要建立在大数据集上。对于大数据集，完全对数似然的评估在训练目的上变得不可行。为了解决这一问题，作者采用了随机梯度下降（SGD）方法，利用小批量的数据来近似似然项，这样变分目标就变成：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094539_8d.webp)

其中 ，每个子集的大小为 。这为在训练期间使用大型数据集提供了一种有效的方法。在通过一个子集 后，应用反向传播来更新模型参数。SGD 是使用变分推理方法训练神经网络和贝叶斯神经网络的最常用方法。

Graves 在 2011 年发表了 BNN 研究的一篇重要论文《Practical variational inference for neural networks》。这项工作提出了一个 MFVB 处理使用因子高斯近似后验分布。这项工作的主要贡献是导数的计算。变分推理的目标可以看作是两个期望值的总和如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094544_2b.webp)

Opper 在 2009 年发表的《The variational gaussian approximation revisited Neural computation》中提出了利用高斯梯度特性来对参数进行更新操作，具体的如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094548_f8.webp)

上面两个公式用于近似平均参数和方差参数的梯度，并且该框架允许对 ELBO 进行优化，并且可以推广到任何的对数损失参数模型中。

已知分数函数估计依赖于对数导数性质，具体公式如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094553_c0.webp)

利用这个性质，可以对一个期望的导数形成蒙特卡罗估计，这在变分推理中是经常被使用，具体的推导过程如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094559_43.webp)

变分推断的第二种梯度估计方法是通过路径导数估计值法。这项工作建立在“重新参数化技巧”的基础上，其中一个随机变量被表示为一个确定性和可微的表达式，具体形式如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094602_b1.webp)

其中 和 表示的是哈达玛积（Hadamard product）。使用这种方法可以有效地对期望值进行蒙特卡罗估计，具体的计算公式如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094607_ca.webp)

由于上式是可微的，所以可以使用梯度下降法来优化这种期望近似。这是变分推断中的一个重要属性，因为变分推断的目标的对数似然的期望值的求解困难很大。与分数函数估计量相比，路径估计值法更有利于降低方差。

Blundell 等人在论文《Bayes by Backprop》中提出了一种在 BNNs 中进行近似推理的方法。该方法利用重参数化技巧来说明如何找到期望导数的无偏估计。其期望导数的具体形式如下所示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094614_51.webp)

在贝叶斯的反向传播的算法中，函数 设为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094618_1b.webp)

其中 可以看作是期望值的自变量，它是下界的一部分。

假设全因子高斯后验函数 ，其中 用于确保标准差参数为正。由此，将网络中的权重 的分布重新参数化为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094623_63.webp)

在该 BNN 中，可训练参数为 和 。由于采用全因子分布，则近似后验概率的对数可以表示为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094626_c4.webp)

综合上面提到的贝叶斯的反向传播算法的细节，会有如下完整的算法流程图。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094629_12.webp)

## 8 **BNN的高斯过程特性**

下图为当在参数上放置高斯先验时，随着网络规模的增加，先验在输出上诱导的图示。其中图中的每个点对应于一个网络的输出，参数从先验分布中进行采样。对于每个网络，隐藏单元的数量是图（a）对应着 1 个单元，图（b）对应着 3 个单元，图（c）对应着 10 个单元，图（d）对应着 100 个单元。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094656_f7.webp)

由此可知在中心极限定理下，随着隐层数 N 逐渐增大，输出服从高斯分布。由于输出被描述为基函数的无穷和，因此可以将输出看作高斯过程。高斯过程和具有单个隐藏层工作的无限宽网络之间的关系最近已扩展到深度神经网络的当中，这一联系的识别激发了 BNNs 的许多研究工作。

高斯过程提供了可靠的不确定性估计、可解释性和鲁棒性。但是高斯过程提供这些好处的代价是预测性能和随着数据集大小的增加所需的大量计算资源。高斯过程和 BNNs 之间的这种联系促使了两种建模方案的合并；既能维持神经网络的预测性能和灵活性，同时结合了高斯过程的的鲁棒性和概率性。

最近的研究已经确定了高斯过程属性不限于 MLP-BNN，而且在卷积中也可以应用该属性。因为 CNN 可以实现为 MLP，其结构在权重中被强制执行。Vander Wilk 等人在 2003 年发表的论文《Convolutional gaussian processes》中提出了卷积高斯过程，它实现了一种类似于 CNN 中的基于 patch 的操作来定义 GP 先验函数。

如下图所示，分析显示了高斯过程在预测偏差和方差方面的对比性能。用 Backprop 和一个因式高斯近似后验概率对 Bayes 模型进行训练，在训练数据分布的情况下，虽然训练数据区域外的方差与高斯过程相比显著低估，但预测结果是合理的。具有标度伯努利近似后验的 MC-Dropout 通常表现出更大的方差，尽管在训练数据的分布中保持不必要的高方差。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094701_c6.webp)

## 9 **卷积BNN**

虽然 MLP 是神经网络的基础，但最突出的神经网络架构是卷积神经网络。这些网络在具有挑战性的图像分类任务方面表现出色，其预测性能远远超过先前基于核或特征工程的方法。CNN 不同于典型的 MLP，它的应用是一个卷积型的算子，单个卷积层的输出可以表示为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094724_4f.webp)

其中 是非线性激活， 表示类似卷积的运算。这里输入 X 和权重矩阵 W 不再局限于向量或矩阵，而是可以是多维数组。可以得到证明的是 cnn 可以编写成具有等效 MLP 模型，允许使用优化的线性代数包进行反向传播训练。

在现有研究方法的基础上，发展了一种新型的贝叶斯卷积神经网络（BCNN）。假设卷积层中的每个权重是独立的，允许对每个单独的参数进行因子分解，其中 BCNN 输出概率向量由 Softmax 函数表示：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094728_d1.webp)

非标准化的后验分布可以表示为：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526094732_8a.webp)

作者做了相关的对比实验，BCNN 使用了流行的 LeNet 架构，并利用 BCNN 的平均输出进行分类，并使用可信区间来评估模型的不确定性。

在 MNIST 数据集中的 10000 个测试图像上，两个网络的总体预测性能显示出了比较好的性能。BCNN 的测试预测精度为 98.99%，香草网络的预测精度略有提高，预测精度为 99.92%。

虽然竞争性的预测性能是必不可少的，但 BCNN 的主要优点是可以提供有关预测不确定性的有价值的信息。从这些例子中，可以看到这些具有挑战性的图像存在大量的预测不确定性，这些不确定性可以用于在实际场景中做出更明智的决策。






1. F. Rosenblatt, “The perceptron: A probabilistic model for information storage and
organization in the brain.” Psychological Review, vol. 65, no. 6, pp. 386 – 408, 1958.
2. C. Bishop, Pattern recognition and machine learning. New York: Springer, 2006.
3. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by
back-propagating errors,” nature, vol. 323, no. 6088, p. 533, 1986.
4. K.-S. Oh and K. Jung, “Gpu implementation of neural networks,” Pattern Recogni-
tion, vol. 37, no. 6, pp. 1311–1314, 2004.
5. D. C. Ciresan, U. Meier, L. M. Gambardella, and J. Schmidhuber, “Deep big simple
neural nets excel on handwritten digit recognition,” CoRR, 2010.
6. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep con-
volutional neural networks,” in Advances in neural information processing systems,
2012, pp. 1097–1105.
7. K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale
image recognition,” CoRR, 2014.
8. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Van-
houcke, A. Rabinovich et al., “Going deeper with convolutions,” in CVPR, 2015.
9. R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for ac-
curate object detection and semantic segmentation,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2014, pp. 580–587.
10. S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object de-
tection with region proposal networks,” in Advances in neural information processing
systems, 2015, pp. 91–99.
11. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified,
real-time object detection,” in Proceedings of the IEEE conference on computer vision
and pattern recognition, 2016, pp. 779–788.
12. A. Mohamed, G. E. Dahl, and G. Hinton, “Acoustic modeling using deep belief
networks,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 20,
no. 1, pp. 14–22, 2012.
13. G. E. Dahl, D. Yu, L. Deng, and A. Acero, “Context-dependent pre-trained deep neu-
ral networks for large-vocabulary speech recognition,” IEEE Transactions on audio,
speech, and language processing, vol. 20, no. 1, pp. 30–42, 2012.
14. G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Van-
houcke, P. Nguyen, T. N. Sainath et al., “Deep neural networks for acoustic modeling
in speech recognition: The shared views of four research groups,” IEEE Signal Pro-
cessing Magazine, vol. 29, no. 6, pp. 82–97, 2012.
15. D. Amodei, S. Ananthanarayanan, R. Anubhai, J. Bai, E. Battenberg, C. Case,
J. Casper, B. Catanzaro, Q. Cheng, G. Chen, J. Chen, J. Chen, Z. Chen,
M. Chrzanowski, A. Coates, G. Diamos, K. Ding, N. Du, E. Elsen, J. Engel, W. Fang,
L. Fan, C. Fougner, L. Gao, C. Gong, A. Hannun, T. Han, L. Johannes, B. Jiang,
C. Ju, B. Jun, P. LeGresley, L. Lin, J. Liu, Y. Liu, W. Li, X. Li, D. Ma, S. Narang,
A. Ng, S. Ozair, Y. Peng, R. Prenger, S. Qian, Z. Quan, J. Raiman, V. Rao,
S. Satheesh, D. Seetapun, S. Sengupta, K. Srinet, A. Sriram, H. Tang, L. Tang,
C. Wang, J. Wang, K. Wang, Y. Wang, Z. Wang, Z. Wang, S. Wu, L. Wei, B. Xiao,
W. Xie, Y. Xie, D. Yogatama, B. Yuan, J. Zhan, and Z. Zhu, “Deep speech 2 :
End-to-end speech recognition in english and mandarin,” in Proceedings of The 33rd
International Conference on Machine Learning, ser. Proceedings of Machine Learn-
ing Research, M. F. Balcan and K. Q. Weinberger, Eds., vol. 48.  New York, New
York, USA: PMLR, 20–22 Jun 2016, pp. 173–182.
16. D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hu-
bert, L. Baker, M. Lai, A. Bolton et al., “Mastering the game of go without human
knowledge,” Nature, vol. 550, no. 7676, p. 354, 2017.
17. “Smartening up with artificial intelligence (ai) - what’s in it for germany and
its industrial sector?” McKinsey & Company, Inc, Tech. Rep., 4 2017. [Online].
Available: https://www.mckinsey.de/files/170419 mckinsey ki final m.pdf
18. E.  V.  T.  V.  Serooskerken,  “Artificial intelligence in wealth and as-
set management,” Pictet on Robot Advisors, Tech. Rep., 1 2017.
[Online]. Available: https://perspectives.pictet.com/wp-content/uploads/2016/12/
Edgar-van-Tuyll-van-Serooskerken-Pictet-Report-winter-2016-2.pdf
19. A.  van  den  Oord,  T.  Walters,  and  T.  Strohman,  “Wavenet  launches
in  the  google  assistant.”  [Online].  Available:  https://deepmind.com/blog/
wavenet-launches-google-assistant/
20. Siri Team, “Deep learning for siri’s voice: On-device deep mixture density
networks for hybrid unit selection synthesis,” 8 2017. [Online]. Available:
https://machinelearning.apple.com/2017/08/06/siri-voices.html
21. J. Wakefield, “Microsoft chatbot is taught to swear on twitter.” [Online]. Available:
www.bbc.com/news/technology-35890188
22. J.  Guynn,  “Google  photos  labeled  black  people  ’goril-
las’.”  [Online].  Available:  https://www.usatoday.com/story/tech/2015/07/01/
google-apologizes-after-photos-identify-black-people-as-gorillas/29567465/
23. J. Buolamwini and T. Gebru, “Gender shades: Intersectional accuracy disparities
in commercial gender classification,” in Conference on fairness, accountability and
transparency, 2018, pp. 77–91.
24. Tesla Team, “A tragic loss.” [Online]. Available: https://www.tesla.com/en GB/
blog/tragic-loss
25. ABC News, “Uber suspends self-driving car tests after vehicle hits and kills woman
crossing the street in arizona,” 2018. [Online]. Available: http://www.abc.net.au/
news/2018-03-20/uber-suspends-self-driving-car-tests-after-fatal-crash/9565586
26. Council of European Union, “Regulation (eu) 2016/679 of the european parliment
and of the council,” 2016.
27. B. Goodman and S. Flaxman, “European union regulations on algorithmic decision-
making and a right to explanation,” AI magazine, vol. 38, no. 3, pp. 50–57, 2017.
28. M. Vu, T. Adali, D. Ba, G. Buzsaki, D. Carlson, K. Heller, C. Liston, C. Rudin,
V. Sohal, A. Widge, H. Mayberg, G. Sapiro, and K. Dzirasa, “A shared vision for
machine learning in neuroscience,” JOURNAL OF NEUROSCIENCE, vol. 38, no. 7,
pp. 1601–1607, 2018.
29. A. Holzinger, C. Biemann, C. S. Pattichis, and D. B. Kell, “What do we need to build
explainable ai systems for the medical domain?” arXiv preprint arXiv:1712.09923,
2017.
30. R. Caruana, Y. Lou, J. Gehrke, P. Koch, M. Sturm, and N. Elhadad, “Intelligible
models for healthcare: Predicting pneumonia risk and hospital 30-day readmission,”
in Proceedings of the 21th ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining. ACM, 2015, pp. 1721–1730.
31. D. Gunning, “Explainable artificial intelligence (xai),” Defense Advanced Research
Projects Agency (DARPA), nd Web, 2017.
32. D. J. MacKay, “Probable networks and plausible predictionsa review of practical
bayesian methods for supervised neural networks,” Network: computation in neural
systems, vol. 6, no. 3, pp. 469–505, 1995.
33. J. Lampinen and A. Vehtari, “Bayesian approach for neural networksreview and case
studies,” Neural networks, vol. 14, no. 3, pp. 257–274, 2001.
34. H. Wang and D.-Y. Yeung, “Towards bayesian deep learning: A survey,” arXiv
preprint arXiv:1604.01662, 2016.
35. K. Murphey, Machine learning, a probabilistic perspective.  Cambridge, MA: MIT
Press, 2012.
36. X. Glorot, A. Bordes, and Y. Bengio, “Deep sparse rectifier neural networks,” in
AISTATS, 2011, pp. 315–323.
37. A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier nonlinearities improve neural
network acoustic models,” in ICML, vol. 30, 2013, p. 3.
38. R. M. Neal, Bayesian learning for neural networks.  Springer Science & Business
Media, 1996, vol. 118.
39. Y. Gal, “Uncertainty in deep learning,” University of Cambridge, 2016.
40. N. Tishby, E. Levin, and S. A. Solla, “Consistent inference of probabilities in layered
networks: predictions and generalizations,” in International 1989 Joint Conference
on Neural Networks, 1989, pp. 403–409 vol.2.
41. J. S. Denker and Y. Lecun, “Transforming neural-net output levels to probability
distributions,” in NeurIPS, 1991, pp. 853–859.
42. G. Cybenko, “Approximation by superpositions of a sigmoidal function,” Mathemat-
ics of control, signals and systems, vol. 2, no. 4, pp. 303–314, 1989.
43. K.-I. Funahashi, “On the approximate realization of continuous mappings by neural
networks,” Neural networks, vol. 2, no. 3, pp. 183–192, 1989.
44. K. Hornik, “Approximation capabilities of multilayer feedforward networks,” Neural
networks, vol. 4, no. 2, pp. 251–257, 1991.
45. S. F. Gull and J. Skilling, “Quantified maximum entropy memsys5 users manual,”
Maximum Entropy Data Consultants Ltd, vol. 33, 1991.
46. D. J. MacKay, “Bayesian interpolation,” Neural computation, vol. 4, no. 3, pp. 415–
447, 1992.
47. ——, “Bayesian methods for adaptive models,” Ph.D. dissertation, California Insti-
tute of Technology, 1992.
48. ——, “A practical bayesian framework for backpropagation networks,” Neural com-
putation, vol. 4, no. 3, pp. 448–472, 1992.
49. M. I. Jordan, Z. Ghahramani, T. S. Jaakkola, and L. K. Saul, “An introduction
to variational methods for graphical models,” Machine learning, vol. 37, no. 2, pp.
183–233, 1999.
50. M. J. Wainwright, M. I. Jordan et al., “Graphical models, exponential families, and
variational inference,” Foundations and Trends R ? in Machine Learning, vol. 1, no.
1–2, pp. 1–305, 2008.
51. D. M. Blei, A. Kucukelbir, and J. D. McAuliffe, “Variational inference: A review for
statisticians,” Journal of the American Statistical Association, vol. 112, no. 518, pp.
859–877, 2017.
52. M. D. Hoffman, D. M. Blei, C. Wang, and J. Paisley, “Stochastic variational infer-
ence,” The Journal of Machine Learning Research, vol. 14, no. 1, pp. 1303–1347,
2013.
53. D. Barber and C. M. Bishop, “Ensemble learning in bayesian neural networks,” NATO
ASI SERIES F COMPUTER AND SYSTEMS SCIENCES, vol. 168, pp. 215–238,
1998.
54. G. E. Hinton and D. Van Camp, “Keeping the neural networks simple by minimizing
the description length of the weights,” in Proceedings of the sixth annual conference
on Computational learning theory. ACM, 1993, pp. 5–13.
55. M. Betancourt, “A conceptual introduction to hamiltonian monte carlo,” arXiv
preprint arXiv:1701.02434, 2017.
56. M. Betancourt, S. Byrne, S. Livingstone, M. Girolami et al., “The geometric founda-
tions of hamiltonian monte carlo,” Bernoulli, vol. 23, no. 4A, pp. 2257–2298, 2017.
57. G. Madey, X. Xiang, S. E. Cabaniss, and Y. Huang, “Agent-based scientific simula-
tion,” Computing in Science & Engineering, vol. 2, no. 01, pp. 22–29, jan 2005.
58. S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, “Hybrid monte carlo,”
Physics letters B, vol. 195, no. 2, pp. 216–222, 1987.
59. R. M. Neal et al., “Mcmc using hamiltonian dynamics,” Handbook of markov chain
monte carlo, vol. 2, no. 11, p. 2, 2011.
60. S. Brooks, A. Gelman, G. Jones, and X.-L. Meng, Handbook of markov chain monte
carlo. CRC press, 2011.
61. M. Welling and Y. Teh, “Bayesian learning via stochastic gradient langevin dynam-
ics,” Proceedings of the 28th International Conference on Machine Learning, ICML
2011, pp. 681–688, 2011.
62. A. Graves, “Practical variational inference for neural networks,” in Advances in Neu-
ral Information Processing Systems 24, J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett,
F. Pereira, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2011, pp. 2348–2356.
63. M. Opper and C. Archambeau, “The variational gaussian approximation revisited,”
Neural computation, vol. 21, no. 3, pp. 786–792, 2009.
64. J. M. Hern´  andez-Lobato and R. Adams, “Probabilistic backpropagation for scal-
able learning of bayesian neural networks,” in International Conference on Machine
Learning, 2015, pp. 1861–1869.
65. J. Paisley, D. Blei, and M. Jordan, “Variational bayesian inference with stochastic
search,” arXiv preprint arXiv:1206.6430, 2012.
66. J. R. Wilson, “Variance reduction techniques for digital simulation,” American Jour-
nal of Mathematical and Management Sciences, vol. 4, no. 3-4, pp. 277–312, 1984.
67. M. Opper and C. Archambeau, “The variational gaussian approximation revisited,”
Neural computation, vol. 21 3, pp. 786–92, 2009.
68. D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint
arXiv:1312.6114, 2013.
69. D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic backpropagation and ap-
proximate inference in deep generative models,” in Proceedings of the 31st Interna-
tional Conference on Machine Learning (ICML), 2014, pp. 1278–1286.
70. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov,
“Dropout: A simple way to prevent neural networks from overfitting,” The Jour-
nal of Machine Learning Research, vol. 15, no. 1, pp. 1929–1958, 2014.
71. D. P. Kingma, T. Salimans, and M. Welling, “Variational dropout and the local
reparameterization trick,” in Advances in Neural Information Processing Systems,
2015, pp. 2575–2583.
72. S. Wang and C. Manning, “Fast dropout training,” in international conference on
machine learning, 2013, pp. 118–126.
73. A. Livnat, C. Papadimitriou, N. Pippenger, and M. W. Feldman, “Sex, mixability,
and modularity,” Proceedings of the National Academy of Sciences, vol. 107, no. 4,
pp. 1452–1457, 2010.
74. M. Opper and O. Winther, “A bayesian approach to on-line learning,” On-line learn-
ing in neural networks, pp. 363–378, 1998.
75. T. P. Minka, “A family of algorithms for approximate bayesian inference,” Ph.D.
dissertation, Massachusetts Institute of Technology, 2001.
76. C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, “Weight uncertainty in
neural networks,” arXiv preprint arXiv:1505.05424, 2015.
77. C. K. Williams, “Computing with infinite networks,” in Advances in neural informa-
tion processing systems, 1997, pp. 295–301.
78. J. Lee, J. Sohl-dickstein, J. Pennington, R. Novak, S. Schoenholz, and Y. Bahri, “Deep
neural networks as gaussian processes,” in International Conference on Learning
Representations, 2018.
79. A. Damianou and N. Lawrence, “Deep gaussian processes,” in AISTATS, 2013, pp.
207–215.
80. A. Damianou, “Deep gaussian processes and variational propagation of uncertainty,”
Ph.D. dissertation, University of Sheffield, 2015.
81. N. Lawrence, “Deep gaussian processes,” 2019. [Online]. Available: http:
//inverseprobability.com/talks/notes/deep-gaussian-processes.html
82. A. Damianou, M. K. Titsias, and N. D. Lawrence, “Variational gaussian process
dynamical systems,” in NeurIPS, 2011, pp. 2510–2518.
83. M. Titsias, “Variational learning of inducing variables in sparse gaussian processes,”
in Proceedings of the Twelth International Conference on Artificial Intelligence
and Statistics, ser. Proceedings of Machine Learning Research, D. van Dyk and
M. Welling, Eds., vol. 5. Hilton Clearwater Beach Resort, Clearwater Beach, Florida
USA: PMLR, 16–18 Apr 2009, pp. 567–574.
84. Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Insights and
applications,” in Deep Learning Workshop, ICML, vol. 1, 2015, p. 2.
85. ——, “Dropout as a bayesian approximation: Representing model uncertainty in deep
learning,” in ICML, 2016, pp. 1050–1059.
86. ——, “Dropout as a bayesian approximation: Appendix,” arXiv preprint
arXiv:1506.02157, 2015.
87. A. Garriga-Alonso, L. Aitchison, and C. E. Rasmussen, “Deep convolutional networks
as shallow gaussian processes,” arXiv preprint arXiv:1808.05587, 2018.
88. R. Novak, L. Xiao, Y. Bahri, J. Lee, G. Yang, D. A. Abolafia, J. Pennington, and
J. Sohl-dickstein, “Bayesian deep convolutional networks with many channels are
gaussian processes,” in International Conference on Learning Representations, 2019.
89. M. Van der Wilk, C. E. Rasmussen, and J. Hensman, “Convolutional gaussian pro-
cesses,” in Advances in Neural Information Processing Systems, 2017, pp. 2849–2858.
90. D. J. MacKay and D. J. Mac Kay, Information theory, inference and learning algo-
rithms. Cambridge university press, 2003.
91. B. Wang and D. Titterington, “Inadequacy of interval estimates corresponding to
variational bayesian approximations.” in AISTATS. Barbados, 2005.
92. R. E. Turner and M. Sahani, “Two problems with variational expectation maximi-
sation for time-series models,” in Bayesian Time Series Models, D. Barber, A. T.
Cemgil, and S. Chiappa, Eds. Cambridge University Press, 2011.
93. R. Giordano, T. Broderick, and M. I. Jordan, “Covariances, robustness, and vari-
ational bayes,” Journal of Machine Learning Research, vol. 19, no. 51, pp. 1–49,
2018.
94. D. Hafner, D. Tran, A. Irpan, T. Lillicrap, and J. Davidson, “Reliable uncertainty
estimates in deep neural networks using noise contrastive priors,” arXiv preprint
arXiv:1807.09289, 2018.
95. V. Kuleshov, N. Fenner, and S. Ermon, “Accurate uncertainties for deep learning
using calibrated regression,” arXiv preprint arXiv:1807.00263, 2018.
96. Y. Gal, J. Hron, and A. Kendall, “Concrete dropout,” in Advances in Neural Infor-
mation Processing Systems, 2017, pp. 3581–3590.
97. C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distribution: A continuous
relaxation of discrete random variables,” arXiv preprint arXiv:1611.00712, 2016.
98. T. S. Jaakkola and M. I. Jordan, “Improving the mean field approximation via the
use of mixture distributions,” in Learning in graphical models.  Springer, 1998, pp.
163–173.
99. C. Louizos and M. Welling, “Structured and efficient variational deep learning with
matrix gaussian posteriors,” in International Conference on Machine Learning, 2016,
pp. 1708–1716.
100. E. G. Tabak and E. Vanden-Eijnden, “Density estimation by dual ascent of the log-
likelihood,” Commun. Math. Sci., vol. 8, no. 1, pp. 217–233, 03 2010.
101. E. G. Tabak and C. V. Turner, “A family of nonparametric density estimation al-
gorithms,” Communications on Pure and Applied Mathematics, vol. 66, no. 2, pp.
145–164, 2013.
102. D. J. Rezende and S. Mohamed, “Variational inference with normalizing flows,” arXiv
preprint arXiv:1505.05770, 2015.
103. C. Louizos and M. Welling, “Multiplicative normalizing flows for variational bayesian
neural networks,” in Proceedings of the 34th International Conference on Machine
Learning - Volume 70, ser. ICML’17. JMLR.org, 2017, pp. 2218–2227.
104. L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density estimation using real NVP,”
CoRR, vol. abs/1605.08803, 2016.
105. C. Cremer, X. Li, and D. K. Duvenaud, “Inference suboptimality in variational au-
toencoders,” CoRR, vol. abs/1801.03558, 2018.
106. S.-i. Amari, Differential-geometrical methods in statistics. Springer Science & Busi-
ness Media, 2012, vol. 28.
107. T. Minka et al., “Divergence measures and message passing,” Technical report, Mi-
crosoft Research, Tech. Rep., 2005.
108. Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama,
and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” arXiv
preprint arXiv:1408.5093, 2014.
109. F. Chollet, “keras,” https://github.com/fchollet/keras, 2015.
110. M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado,
A. Davis, J. Dean, M. Devin, S. Ghemawat, I. J. Goodfellow, A. Harp, G. Irving,
M. Isard, Y. Jia, R. J´  ozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man´ e,
R. Monga, S. Moore, D. G. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner,
I. Sutskever, K. Talwar, P. A. Tucker, V. Vanhoucke, V. Vasudevan, F. B. Vi´ egas,
O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, “Tensor-
flow: Large-scale machine learning on heterogeneous distributed systems,” CoRR,
vol. abs/1603.04467, 2016.
111. J. V. Dillon, I. Langmore, D. Tran, E. Brevdo, S. Vasudevan, D. Moore, B. Patton,
A. Alemi, M. D. Hoffman, and R. A. Saurous, “Tensorflow distributions,” CoRR, vol.
abs/1711.10604, 2017.
112. P. Adam, G. Sam, C. Soumith, C. Gregory, Y. Edward, D. Zachary, L. Zeming, D. Al-
ban, A. Luca, and L. Adam, “Automatic differentiation in pytorch,” in Proceedings
of Neural Information Processing Systems, 2017.
113. T. Chen, M. Li, Y. Li, M. Lin, N. Wang, M. Wang, T. Xiao, B. Xu, C. Zhang, and
Z. Zhang, “Mxnet: A flexible and efficient machine learning library for heterogeneous
distributed systems,” CoRR, vol. abs/1512.01274, 2015.
114. A. Kucukelbir, D. Tran, R. Ranganath, A. Gelman, and D. M. Blei, “Automatic
Differentiation Variational Inference,” arXiv e-prints, p. arXiv:1603.00788, Mar 2016.
115. S. Patterson and Y. W. Teh, “Stochastic gradient riemannian langevin dynamics on
the probability simplex,” in Advances in Neural Information Processing Systems 26,
C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, Eds.
Curran Associates, Inc., 2013, pp. 3102–3110.
116. T. Chen, E. Fox, and C. Guestrin, “Stochastic gradient hamiltonian monte carlo,” in
Proceedings of the 31st International Conference on Machine Learning, ser. Proceed-
ings of Machine Learning Research, E. P. Xing and T. Jebara, Eds., vol. 32. PMLR,
22–24 Jun 2014, pp. 1683–1691.
117. C. Li, C. Chen, D. Carlson, and L. Carin, “Preconditioned Stochastic Gradient
Langevin Dynamics for Deep Neural Networks,” arXiv e-prints, Dec. 2015.
118. M. Betancourt, “The fundamental incompatibility of scalable hamiltonian monte
carlo and naive data subsampling,” in Proceedings of the 32Nd International Confer-
ence on International Conference on Machine Learning - Volume 37, ser. ICML’15.
JMLR.org, 2015, pp. 533–540.
119. I. Osband, C. Blundell, A. Pritzel, and B. V. Roy, “Deep exploration via bootstrapped
DQN,” CoRR, vol. abs/1602.04621, 2016.
120. A. G. d. G. Matthews, M. van der Wilk, T. Nickson, K. Fujii, A. Boukouvalas,
P. Le´  on-Villagr´  a, Z. Ghahramani, and J. Hensman, “GPflow: A Gaussian process
library using TensorFlow,” Journal of Machine Learning Research, vol. 18, no. 40,
pp. 1–6, 4 2017.
121. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and
L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural
computation, vol. 1, no. 4, pp. 541–551, 1989.
122. I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT Press, 2016.
123. Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to
document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov
1998.