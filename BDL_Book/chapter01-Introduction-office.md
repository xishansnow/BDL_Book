# 贝叶斯神经网络最新综述

【原文】Goan, E., & Fookes, C. (2020). Bayesian Neural Networks: An Introduction and Survey.  https://arxiv.org/abs/2006.12024

【摘要】神经网络已经为许多机器学习任务提供了最先进的结果，例如计算机视觉、语音识别和自然语言处理领域的检测、回归和分类任务等。尽管取得了成功，但它们通常是在频率学派框架内实施的，这意味着其无法对预测中的不确定性进行推断。本文介绍了贝叶斯神经网络及一些开创性研究，对不同近似推断方法进行了比较，并提出未来改进的一些方向。

---

# 1 引言

长期以来，仿生学一直是技术发展的基础。科学家和工程师反复使用物理世界的知识来模仿自然界对经过数十亿年演变而来的复杂问题的优雅解决方案。生物仿生学在统计学和机器学习中的重要例子是感知器的发展[1]，它提出了一个基于神经元生理学的数学模型。机器学习团体已使用该概念开发高度互连的神经元阵列统计模型，以创建神经网络。

虽然神经网络的概念早在几十年前就已为人所知，但其应用直到最近才显现出来。神经网络研究的停滞很大程度上由三个关键因素造成：

- 缺乏足够的算法来训练这些网络。
- 训练复杂网络所需的大量数据。
- 训练过程所需大量计算资源。

1986年，[3]引入了反向传播算法解决了网络的有效训练问题。虽然有了有效的训练手段，但网络规模不断扩大，仍然需要相当多计算资源。该问题在[4，5，6]中得到了解决，其表明通用 GPU 可有效执行训练所需的许多操作。随着硬件不断进步，能够捕获和存储真实世界数据的传感器数量不断增加。通过高效训练方法、改进的计算资源和庞大的数据集，复杂神经网络的训练已经变得真正可行。

在绝大多数情况下，神经网络都是在频率主义框架内使用的；通过使用有效的数据，用户可以定义网络结构和成本函数，然后对其进行优化，以获得模型参数的点估计。增加神经网络参数（权重）的数量或网络深度会增加神经网络的容量，使其能够表示强非线性函数，进而允许神经网络处理更复杂的任务。但频率主义框架也很容易由于参数过多而产生过拟合问题，但使用大型数据集和正则化方法（如寻找最大后验估计），可以限制网络所学习函数的复杂性，并有助于避免过拟合。

神经网络已经为许多机器学习和人工智能应用提供了最先进的结果，例如图像分类[6，7，8]，目标检测[9，10，11]和语音识别[12，13，14，15]。其他网络（如 `DeepMind` [16] 开发的 `AlphaGo` 模型）更加突出了神经网络在开发人工智能系统方面的潜力，吸引了广泛受众。随着神经网络性能的不断提高，某些行业对神经网络的开发和应用越来越显著。神经网络目前已经大量用于制造[17]、资产管理[18]和人机交互技术[19，20]。

自从神经网络在工业中部署以来，发生了许多事故。这些系统的故障导致模型出现不道德或不安全的行为，包括一些对边缘化群体表现出较大（性别和种族）偏见的模型[21，22，23]，或者导致生命损失的极端案例[24，25]。神经网络是一种统计黑盒模型，这意味着决策过程并非基于定义良好而直观的协议。相反，决策以一种无法解释的方式做出。因此，在社会和安全关键环境中使用这些系统会引起相当大的伦理关注。鉴于此，欧盟发布了一项新规定，明确用户拥有对人工智能系统所做决定的“解释权”[26，27]。由于不清楚系统操作或设计的原则方法，其他领域的专家仍然对采用神经网络技术感到担忧[28，29，30]。这激发了对可解释人工智能的研究尝试[31]。

神经网络的充分工程设计需要合理地理解其能力和局限性；尽量在部署前就找出其不足，而避免在悲剧发生后调查其缺陷。由于神经网络是一个统计黑匣子，当前理论尚无法解释和说明其决策过程。普通神经网络的频率学派观点为决策提供了缺乏解释和过度自信的估计，使其不适合于诸如医疗诊断、自动驾驶汽车等高风险领域。贝叶斯统计提供了一种自然的方式来推断预测中的不确定性，并可以洞察决策是如何做出的。

图 1 比较了执行回归任务的贝叶斯方法和简单神经网络方法，并说明了度量不确定性的重要性。虽然两种方法在需要外推的训练数据范围内执行得都很好，但贝叶斯方法提供了函数输出的完全分布，而不是神经网络提供的点估计。贝叶斯方法输出的分布允许开发可靠的模型，因为其可以识别预测中的不确定性。考虑到神经网络是人工智能系统最有前途的技术，让人们信任神经网络的预测结果也就随之变得越加重要了。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210526093343_31.webp)

图1：在紫色区域没有训练数据的回归任务中，神经网络与传统概率方法的比较。(A)使用具有2个隐藏层的神经网络的回归输出；(B)使用高斯过程框架的回归，灰色条表示±2标准。从期望值开始。

贝叶斯观点使我们能够解决神经网络目前面临的许多挑战。为此，在网络参数上放置一个分布，然后将得到的网络被称为贝叶斯神经网络（Bayesian Neural Networks，BNN）。贝叶斯神经网络的目标是拥有一个高容量模型，该模型展示了贝叶斯分析的重要理论优势。最近已经有不少研究致力于将贝叶斯近似应用于实际的神经网络中，而这些研究面临的主要挑战在于：如何在合理计算约束下，部署能够提供准确预测能力的模型。

本文的目的是向读者提供容易理解的贝叶斯神经网络介绍，同时伴随该领域的一些开创性工作和实验的调研，以激发对当前方法的能力和局限性的讨论。涉及贝叶斯神经网络的资料太多，无法一一列举，因此本文仅列出了其中具有里程碑性质的关键项目。同样，许多关键成果的推导也被省略了，仅列出了最后结果，并附有对原始来源的引用。我们也鼓励受相关研究启发的读者参考之前的一些调研报告：[32] 调查了贝叶斯神经网络的早期发展，[33] 讨论了对神经网络进行全面贝叶斯处理的细节，以及 [34] 调查了近似贝叶斯推断在现代网络结构中的应用。

本文应该适合所有统计领域的读者，但更感兴趣的读者可能是那些熟悉机器学习概念的人。尽管机器学习和贝叶斯概率方法都很重要[2，35]，但实践中许多现代机器学习方法和贝叶斯统计研究之间存在分歧。希望这项调研能够有助于突出贝叶斯神经网络的现代研究与统计学之间的相似之处，强调概率观点对机器学习的重要性，并促进机器学习和统计学领域未来的合作。

# 2 文献调研

## 2.1  神经网络

在讨论神经网络的贝叶斯观点之前，简要介绍神经计算的基本原理并定义本章要使用的符号是很重要的。本调查将重点介绍感兴趣的主要网络结构-多层感知器(MLP)网络。MLP是神经网络的基础，现代体系结构如卷积网络具有等价的MLP表示。图2显示了一个简单的MLP，它有一个适合于回归或分类的隐藏层。对于具有维度n1的输入x的该网络，f网络的输出可以被建模为，



## 2.2贝叶斯神经网络

### （1）为神经网络引入贝叶斯

在频率主义框架内，模型权重被视为一个尚未知的确切值，而非随机变量，而将数据视为随机变量。这似乎与直觉相反，因为直觉上应该是根据手头已有的信息，判断未知的模型权重是多少。贝叶斯统计建模就是一种符合直觉的方法，其视数据为可获得的信息，而将未知的权重视为随机变量。其基本逻辑是：将未知（或潜在）参数视为随机变量，希望在可观测的训练数据支持下，掌握这些参数的分布。

在贝叶斯神经网络的“学习”过程中，`未知的权重` 可以在 `已知信息` 和 `观测到的信息` 基础上推断出来。这是个逆概率问题，可以用贝叶斯定理来解决。贝叶斯模型中权重 $ω$ 是隐藏或潜在的变量，通常无法直接观测到其真实分布，而贝叶斯定理允许使用 `可观测到的概率` 来表示权重的分布，形成 `以观测数据为条件` 的权重分布 $p(ω|D)$，我们称之为 `后验分布（Posterior Distribution），简称后验（Posterior）`。

在训练前，先观察和分析下权重和数据之间的联合分布 $p(ω，D)$ 。该联合分布由我们对潜在变量的 `先验(Prior)` 信念 $p(ω)$ 和对模型 / `似然（Likelihood）` 的选择 $p(D|ω)$ 来定义：

$$
p(\boldsymbol{\omega}, \mathcal{D})=p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega}) \tag{6}
$$

在神经网络中，式 6 中的似然项 $p(D|ω)$ 由网络结构和所选择的损失函数来定义。例如，对于具有均方误差损失、噪声方差已知的同方差一元回归问题，似然是以网络输出为平均值的高斯分布。

$$
p(\mathcal{D} \mid \boldsymbol{\omega}) = \mathcal{N}\left(\mathbf{f}^{\omega}(\mathcal{D}), \sigma^{2}\right)
$$

在这个回归模型中，一般假设来自 $\mathcal{D}$ 的所有样本是独立同分布的（I.I.D.），这意味着可将似然写成数据集中 $N$ 个独立项的乘积：

$$
p(\mathcal{D} \mid \boldsymbol{\omega})=\prod_{i=1}^{N} \mathcal{N}\left(\mathbf{f}^{\omega}\left(\mathbf{x}_{i}\right), \sigma^{2}\right) \tag{7}
$$

在查看任何数据前，应该指定先验分布，以包含关于权重应该如何分布的信念。由于神经网络的黑箱性质，指定一个有意义的先验是有挑战性的。不过经验主义告诉我们，在许多频率主义网络中，网络训练后的权值较低，且大致集中在零附近，因此可以使用具有较小方差的 `零均值高斯分布` 作为先验，或使用 `以零为中心的 spike-slab 先验` 来鼓励模型中的稀疏性。

```{note}
贝叶斯神经网络中，常假设随机变量呈高斯分布。
```

在指定先验和似然后，应用贝叶斯定理可计算得到模型权重的后验分布：

$$
\pi(\boldsymbol{\omega} \mid \mathcal{D})=\frac{p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega})}{\int p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega}) d \boldsymbol{\omega}}=\frac{p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega})}{p(\mathcal{D})} \tag{8}
$$

后验分布中的分母项称为边缘似然（或证据），其相对于模型权重而言是一个常量，起到对后验进行归一化的作用，以确保后验是有效分布。

### （2）基于后验分布做预测

根据该后验分布可以预测任何感兴趣的量。预测方法是基于后验分布求期望：

$$
\mathbb{E}_{\pi}[f]=\int f(\boldsymbol{\omega}) \pi(\boldsymbol{\omega} \mid \mathcal{D}) d \boldsymbol{\omega} \tag{9}
$$

所有感兴趣的预测量都是上述形式的期望。无论预测均值、方差还是区间，预测量都是基于后验的期望值，它们之间唯一的不同是求取期望的函数 $f(ω)$ 。通过公式可以直观的看出，预测值可被视为函数 $f$  经后验 $π(ω)$ 加权后的平均值。

### （3）基于后验分布做推断

贝叶斯推断任务就是基于后验分布 $\pi(\boldsymbol{\omega} \mid \mathcal{D})$ ，推断出任一随机变量（或子集）的后验分布。因此，贝叶斯推断过程实际上是围绕着对某一模型权重（或子集）的边缘化展开。与频率主义框架中使用的优化方法不同，贝叶斯推断通过边缘化，使我们能够了解模型的生成过程。而贝叶斯方法中的所有预测，都是在该生成模型基础上，以有效的条件概率形式表达的。

上例假设了噪声方差 $σ$ 等先验分布的参数（其实可以泛化到任一先验参数）是已知的，但实践中较少出现此情况，因此也需要将其视为随机变量进行推断。贝叶斯框架允许对其进行推断，而且推断方式和权重类似，即将其视为潜在随机变量，并为之分配一个先验分配（有时称为 `超先验分布`），然后对其进行边缘化处理，以找到后验分布。有关如何对贝叶斯神经网络执行此操作的更多说明，请参考 [33，38]。

```{note}
对先验的某些参数未知时，通常假设该未知参数也是一个随机变量，且服从某一超先验的分布。该分布也需要从数据中学的，相关知识请参阅 `分层贝叶斯` 。
```

### （4） 后验分布的计算难题

对于许多模型，式 8 的后验计算仍然很困难，这主要由边缘似然的计算造成。对于非共轭模型或存在潜在变量的非线性模型（如前馈神经网络），边缘似然几乎没有解析解，而对高维模型则计算更为困难。因此，一般会对后验进行近似。以下部分详细说明了如何在贝叶斯神经网络中实现近似贝叶斯推断。


## 2.3 贝叶斯神经网络的起源

### 2.3.1 BNN 的起源

根据本调研和之前的调查报告 `[39]`，可认为贝叶斯神经网络的第一个实例是在 1989 年的 `[40]` 中发表的。该论文通过对神经网络损失函数的统计解释，强调了其主要统计学特性，并证明了均方误差最小化（MSE）等价于求高斯分布的最大似然估计（MLE）。重要的是，通过给网络权重指定先验，可用贝叶斯定理获得适当的后验。此工作虽然给出了对神经网络非常关键的贝叶斯见解，但并没有提供计算边缘似然的方法，也就意味着没有提出任何实用的推断方法。`Denker` 和 `LeCun[41]` 1991 年对该工作进行了扩展，提供了一种使用拉普拉斯近似进行近似推断的实用方法。

神经网络是一种通用函数逼近器。当单个隐层网络中的参数数量趋近于无穷大时，可以表示任意函数 `[42，43，44]` 。这意味着只要模型有足够参数，就可以用单层神经网络来逼近任何训练数据。但与高次多项式回归类似，虽然表达任意函数的能力增强了（甚至可以精确匹配训练数据），但参数数量的增加会导致过拟合问题。

在 1991 年 `Gull` 和 `Skilling`  `[45]` 的工作基础上，`MacKay` 于 1992 年发表的文章 `《Bayesian interpolation》` `[46]` 展示了如何自然地使用贝叶斯框架处理模型设计和模型比较任务。该工作描述了两个层次的推断：一是用于拟合模型的推断、二是用于评估模型适用性的推断。

第一层次的推断是贝叶斯规则用于模型参数更新的典型应用：

$$
P\left(\boldsymbol{\omega} \mid \mathcal{D}, \mathcal{H}_{i}\right)=\frac{P\left(\mathcal{D} \mid \boldsymbol{\omega}, \mathcal{H}_{i}\right) P\left(\boldsymbol{\omega} \mid \mathcal{H}_{i}\right)}{P\left(\mathcal{D} \mid \mathcal{H}_{i}\right)} \tag{10}
$$

其中 $\omega$ 是统计模型中的参数， $\mathcal{D}$ 是训练数据， $\mathcal{H}_i$ 是用于此层次推断的第 $i$ 个模型（在本层次推断任务中可视为确定的）。上式可以描述为：

$$
\text{Posterior}=\frac{\text { Likelihood } \times \text { Prior }}{\text { Evidence }} \notag
$$

注意式 10 中的归一化常数也被称为模型 $\mathcal{H}_i$ 的证据。对于大多数模型，后验的计算非常困难，只能采用近似的方法。在该论文中使用了拉普拉斯近似。

虽然需要计算参数的后验，但该论文的另一层次目的是展示如何对模型 $\mathcal{H}_i$ 的后验进行评估。其中模型的后验被设计为：

$$
P\left(\mathcal{H}_{i} \mid \mathcal{D}\right) \propto P\left(\mathcal{D} \mid \mathcal{H}_{i}\right) P\left(\mathcal{H}_{i}\right) \tag{11}
$$

该公式可以解释为：

$$
\text{Model Posterior} \propto \text{Evidence} \times \text{Model Prior} \notag
$$

式 11 中的数据依赖项是该模型的证据。尽管对其做出 `后验归一化常数` 的解释很好理解，但和前面所提到的一样，对于大多数贝叶斯神经网络来说，求证据的分布非常困难。

论文假设证据呈高斯分布，并提出了证据的拉普拉斯近似：

$$
\begin{align*} 
P\left(\mathcal{D} \mid \mathcal{H}_{i}\right) &=\int P\left(\mathcal{D} \mid \boldsymbol{\omega}, \mathcal{H}_{i}\right) P\left(\boldsymbol{\omega} \mid \mathcal{H}_{i}\right) d \boldsymbol{\omega} \tag{11} \\ 
& \approx P\left(\mathcal{D} \mid \boldsymbol{\omega}_{\mathrm{MAP}}, \mathcal{H}_{i}\right)\left[P\left(\boldsymbol{\omega}_{\mathrm{MAP}} \mid \mathcal{H}_{i}\right) \Delta \omega\right]  \tag{12} \\ 
&=P\left(\mathcal{D} \mid \boldsymbol{\omega}_{\mathrm{MAP}}, \mathcal{H}_{i}\right)\left[P\left(\boldsymbol{\omega}_{\mathrm{MAP}} \mid \mathcal{H}_{i}\right)(2 \pi)^{\frac{k}{2}} \mathrm{det}^{-\frac{1}{2}} \mathbf{A}\right] \tag{13}\\ 
&=\text { Best Likelihood Fit } \times \text { Occam Factor } 
\end{align*}
$$

这可以解释为对模型证据的一种黎曼近似，是代表证据峰值的 `最佳似然拟合（Best Likelihood Fit）`， `奥卡姆因子（Occam Factor）` 是高斯分布峰值附近曲线的特征宽度，可以解释为给定模型 $\mathcal{H}_i$ 的后验宽度 $∆ω$ 与先验宽度 $∆ω_0$ 之比，计算公式为：

$$
\text{Occam Factor}  =\frac{\Delta \omega}{\Delta \omega_{0}} \tag{15}
$$

这意味着奥卡姆因子是参数空间中从先验到后验的变化率。图 4 展示了此概念，一个能够表示大范围数据的复杂模型（ $\mathcal{H}_2$ ）将拥有更宽的证据，因此具有更大的奥卡姆因子。而简单模型（$\mathcal{H}_1$ ）捕获复杂生成过程的能力较弱，但较小范围的数据能够更确定地建模，从而产生较低的奥卡姆因子。这导致了模型复杂性的天然正规化：不必要的复杂模型通常会导致较宽的后验分布，从而导致较大奥卡姆因子以及给定模型较低的证据。类似的，一个弱信息（分散平坦的）先验将导致奥卡姆因子降低，进一步直观地解释了贝叶斯设置中的正则化（即所谓 “贝叶斯方法内置奥卡姆剃刀”）。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021052615491522.webp)

图 4：证据在评估不同模型中发挥的作用。简单模型 $\mathcal{H}_1$ 能够以更大的强度预测较小范围的数据，而较复杂的模型 $\mathcal{H}_2$ 尽管概率较低，但能够表示较大范围的数据，改编自 `[46，47]`。

如前所述，使用该证据框架需要计算边缘似然，这是贝叶斯建模中最关键的挑战。考虑到近似计算边缘似然所需的大量成本，利用该证据框架比较许多不同的模型似乎不可行。

尽管如此，证据框架可以用来评估贝叶斯神经网络的解决方案。对于大多数感兴趣的神经网络结构，目标函数是非凸的，具有许多局部极小值。每一个局部极小都可以看作是推断问题的一个可能解。`MacKay` 以此为动机，使用所有局部最小值的 `证据函数（Evidence·Function）` 来进行模型比较 `[48]` 这允许在不需要大量计算的情况下评估模型方案的复杂性。

### 2.3.2 早期 BNN 的变分推断

机器学习社区在优化问题上一直表现出色。其中许多最大似然模型（如支持向量机和线性高斯模型）的目标函数都是凸函数，但神经网络的目标函数是高度非凸的，具有许多局部极小值。此特点催生了基于梯度的优化方法，如反向传播 `[3]` 。这种优化方法可以通过 `变分推断（Variational Inference ）` 在贝叶斯统计方法中进行推广。

#### （1）变分推断的原理

变分推断是一种近似推断方法，它将贝叶斯推断过程中所需的边缘化计算定义为一个优化问题 `[49，50，51]` 。变分推断首先假设后验分布的形式，然后通过优化方法找到最接近真实后验的解。这种假设简化了计算，并提供了可操作性。

变分推断的基本思想是：假设后验分布 $q_θ(ω)$ 是参数集 $ω$ 上的概率密度函数（被称为 `变分分布`），该分布被由 $\theta$ 参数化的某一分布族所控制，那么通过优化参数 $\theta$ 来减小变分分布 $q_θ(ω)$ 与真实后验 $p(ω|D)$ 之间的差异性，就可以逐步得到真实后验分布的近似分布 $q_θ(ω)$ 。

变分推断需要一种度量变分分布与真实分布之间相似性的手段作为目标函数，而 `KL 散度` 就是最常用的一种度量：

$$
\text{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega} \mid \mathcal{D})\right)=\int q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega} \mid \mathcal{D})} d \boldsymbol{\omega} \tag{16}
$$

对于变分推断，可将式 16 用作参数 $θ$ 的最小化目标函数，进而将推断问题转变成最优化问题。式 16 可进一步扩展为：

$$
\begin{align*} 
\mathrm{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega} \mid \mathcal{D})\right)&= \mathbb{E}_{q}\left[\log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega})}-\log p(\mathcal{D} \mid \boldsymbol{\omega})\right]+\log p(\mathcal{D}) \tag{17}\\ &=\mathrm{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega})\right)-\mathbb{E}_{q}[\log p(\mathcal{D} \mid \boldsymbol{\omega})]+\log p(\mathcal{D}) \tag{18} \\
&=-\mathcal{F}\left[q_{\theta}\right]+\log p(\mathcal{D}) \tag{19}
\end{align*}
$$

其中，$ \mathcal{F}\left[q_{\theta}\right]=-\mathrm{KL}\left(q_{\theta}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega})\right)+\mathbb{E}_{q}[\log p(\mathcal{D} \mid \boldsymbol{\omega})] $。该组合是为了将易处理的项从难处理的对数边缘似然中分离出来。

现在可以使用反向传播来优化该函数，由于对数边缘似然与参数 $θ$ 无关，因此目标函数关于 $θ$ 的导数为零，导数中只剩下包含变分参数的项 $\mathcal{F}[q_θ]$ 。

式 19 中包含 $\mathcal{F}[q_θ]$ 的负值项，是为强调“它是一个与目标分布不同但等价的推导”，并与文献保持一致。

该结果不是通过最小化真实分布和近似分布之间的 KL 散度，而是通过近似难以处理的对数边缘似然来获得的。通过应用 Jensen 不等式，可以发现 $\mathcal{F}[q_θ]$ 形成了对数边缘似然的下界 `[49,52]` 。通过公式 19 注意到： KL 散度严格 ≥ 0 且仅当两个分布相等时才等于零。对数边缘似然 $\log p(\mathcal{D})$ 等于近似后验与真实后验之间的 KL 散度与  $\mathcal{F}[q_θ]$  之和。通过最小化近似后验和真实后验之间的 KL 散度， $\mathcal{F}[q_θ]$  将越来越接近对数边缘似然。因此， $\mathcal{F}[q_θ]$  通常被称为证据下界（ELBO），图 5 可视化地说明了该点。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021052616535332.webp)

图 5：近似后验和真实后验之间的 KL 散度最小化导致的证据下界最大化。当近似后验和真实后验之间的 KL 散度被最小化时，证据下界 $\mathcal{F}[q_θ]$ 收紧到对数证据。因此，最大化证据下界 `ELBO` 等同于最小化 KL 散度。改编自 `[53]` 。

```{note}

最小化 KL 散度等效于最大化证据下界 `ELBO` 。

```

#### （2）变分推断在神经网络中的早期应用

`Hinton` 和 `Van Camp` `[54]` 首次将变分推断应用于贝叶斯神经网络，试图解决神经网络中的过拟合问题。他们认为，通过模型权重的概率观点，神经网络可包含的信息量会减少，网络会得到简化。该表述从信息论`（特别是最小描述性长度，Minimum Descriptive Length）` 角度出发，但导致了相当于变分推断的框架。

`Hinton` 等人的研究使用了变分推断中常用的 `平均场变分贝叶斯 (MFVB)` 方法。 平均场变分贝叶斯方法假设 `变分分布对参数实施了因子分解` ，而 `Hinton` 等人则进一步假设 `变分分布对参数实施了由若干独立高斯分布构成因子分解`：

$$
q_{\boldsymbol{\theta}}(\boldsymbol{\omega})=\prod_{i=1}^{P} \mathcal{N}\left(w_{i} \mid \mu_{i}, \sigma_{i}^{2}\right) \tag{20}
$$

其中 $P$ 是网络中权重的数量。对于只有一个隐层的回归网络，该变分分布有解析解。能够获得解析解是一种理想特性，因为它可以极大地减少执行推断的时间。

#### （3）变分推断的改进

这项工作存在几个问题，其中最突出的问题是假设变分分布被因子分解为单个网络权重。众所周知，神经网络中的参数之间存在强相关性。因子分解方法实际上是通过牺牲参数之间的相关性来简化计算。`Mackay` 在对贝叶斯神经网络的早期调研中强调了该问题 `[32]`，并提出通过对隐层的输入做预处理，可以获得更全面的近似后验分布。

`Barber` 和 `Bishop` `[53]` 再次强调了该问题，并扩展了 `[54]` 中的工作，允许通过使用 `满秩高斯` 的后验变分分布来捕获参数间的完全相关性。对于使用 Sigmoid 激活函数的单隐层回归网络，提供了评估 `ELBO` 的解析表达式。该方法通过使用适当缩放的误差函数替换 Sigmoid 来实现。

此方案的问题是参数数量过多。对于完全协方差模型，该方案中的参数数量是神经网络权重数量的二次函数。为纠正该点，`Barber` 和 `Bishop` 对因子分析中经常使用的协方差提出了一种限制形式，

$$
\mathbf{C}=\operatorname{diag}\left(d_{1}^{2}, \ldots, d_{n}^{2}\right)+\sum_{i=1}^{s} \mathbf{s}_{i} \mathbf{s}_{i}^{T} \tag{21}
$$

其中，$diag$ 运算符根据长度为 $n$ 的向量 $d$ 创建对角线矩阵，$n$ 为模型中权重的数量。该形式与网络中隐藏单元的数量呈线性关系。

上述工作为将反向传播方法应用于解决贝叶斯问题做出了贡献。使这两个研究领域的特性可以合并，并提供各自的优势。这些工作使我们具备了使用神经网络在概率意义上处理大数据集复杂回归任务的能力。

#### （4）变分推断的局限性

这些方法也存在局限性。`Hinton` 、 `Van Camp` 、 `Barber` 和 `Bishop` 的工作都集中在发展一种封闭形式的网络表示，对网络施加了许多限制。如前面所讨论的，`[54]` 假设后验因子分解为单个权重，它不能捕获参数的相关性。 `[53]` 捕获了协方差结构，但作者将其分析限制在使用误差函数近似 Sigmoid 激活函数上，而该函数由于梯度的幅度较低而在现代神经网络中很少使用。另外，两种方法都存在一个非常关键的局限性，即模型限制为单隐层神经网络。

如前所述，神经网络可通过添加额外的隐藏单元来任意地逼近任何函数。但现代神经网络实验表明，通过增加网络中隐层的数量，可以用更少的隐藏单元来表示相同的复杂函数，并由此产生了“深度学习”，其中深度指的就是隐藏层的数量。当试图近似各层之间的完全协方差结构时，减少权重变量的数量变得尤其重要。例如，可以捕获单层内的隐藏单元之间的相关性，同时假设不同层之间参数是独立的。此类假设可以显著减少相关参数的数量。现代出现了很多拥有很深层数、数以亿计权重的神经网络（目前大多只能提供点估计），因此开发超出单层的实用概率解释需求必不可少。

### 2.3.3 BNN 的混合蒙特卡罗方法

值得反思一下实际感兴趣的数量。到目前为止，重点一直放在寻找后验的良好近似上，但后验的准确表示通常不是最终设计目标，实际感兴趣的主要量是预测矩和区间。我们希望在信心充分的情况下做出良好的预测。之所以强调后验的良好近似，是因为预测矩和区间都必须根据后验 $π(ω|D)$  的来计算期望值。该期望值在式 9 中，为方便在此重复给出:

$$
\mathbb{E}_{\pi}[f]=\int f(\boldsymbol{\omega}) \pi(\boldsymbol{\omega} \mid \mathcal{D}) d \boldsymbol{\omega} 
$$

这就是强调后验计算的原因，因为准确的预测和推断依赖于难以处理的后验（或其近似）。

以前的方法采用优化方法（如变分推断或拉普拉斯近似）对后验的形式有强假设和限制，而这些限制常常造成预测的不准确。正如文献 `[55，56]` 强调的，为预测量而计算的期望值不仅仅是概率密度，而是概率密度和体积的乘积。概率密度是指后验分布 $π(ω|D)$ ，体积是指在 $dω$ 上积分的体积。对于很多模型，密度和体积乘积的期望值可能不会达到质量的最大值。因此，仅考虑质量的优化方案可能会提供不准确的预测量。为用有限的计算资源做出准确预测，不仅需要在质量最大的时候评估该期望，而且在质量和体积的乘积最大时也需要评估。最有希望实现这一点的方法是马尔可夫链蒙特卡罗（MCMC）。

MCMC 算法仍然处于贝叶斯研究和应用统计学的前沿。MCMC 是从任意和难以处理的分布中抽样的一种通用方法。从分布中采样的能力使得人们能够使用蒙特卡罗积分进行预测：

$$
\mathbb{E}_{\pi}[f]=\int f(\boldsymbol{\omega}) \pi(\boldsymbol{\omega} \mid \mathcal{D}) d \boldsymbol{\omega} \approx \frac{1}{N} \sum_{i=1}^{N} f\left(\boldsymbol{\omega}_{i}\right) \tag{22}
$$

其中 $ω_i$ 表示来自后验分布的一个独立样本。MCMC 允许从后验分布抽取`收敛于概率密度和体积的乘积最大时的样本`[55]`。

在 MCMC 上下文中，不需要变分推断方法所做的假设（如后验的因子分解等）。当样本数量趋近于无穷大时，MCMC 会收敛到真实后验。由于避免了假设限制，只要有足够时间和计算资源，我们就可以得到一个更接近真实预测量的解。这对于贝叶斯神经网络来说是一个重要挑战，因为后验分布通常相当复杂。

传统 MCMC 方法表现出随机游走的特点，因为序列中的新推荐是随机生成的。由于 BNN 后验结构的复杂性和高维性，这种随机游走特性很难在合理的时间内进行推断。为避免随机游走，可以采用在迭代过程中加入了梯度信息的 `混合/汉密尔顿蒙特卡罗（HMC）方法`。虽然 HMC 最初被提出用于统计物理 `[58]`，但 `Neal` 强调了 HMC 解决贝叶斯推断的潜力，并专门研究了其在 BNN 和更广泛统计社区中的应用 `[38]` 。

鉴于 HM C最初是为物理动力学提出的，因此通过物理类比来建立应用统计学直觉比较合适。将感兴趣的参数 $ω$ 视为位置变量。然后引入一个辅助变量来模拟当前位置的动量 $\mathbf{v}$ 。该辅助变量没有统计学意义，只是为帮助系统动力学发展而引入的。通过位置和动量变量，我们可以表示系统的势能 $U(ω)$ 和动能 $K(\mathbf{v})$。系统的总能量表示为：

$$
H(\boldsymbol{\omega}, \mathbf{v})=U(\boldsymbol{\omega})+K(\mathbf{v}) \tag{23}
$$

考虑到无损系统的总能量 $H(ω，v)$ 是常数。因此被描述为 `汉密尔顿（Hamiltonian）系统` ，并且被表示为微分方程组 `[59]`，

$$
\frac{d w_{i}}{d t}=\frac{\partial H}{\partial v_{i}} \tag{24}
$$


$$
\frac{d v_{i}}{d t}=-\frac{\partial H}{\partial w_{i}} \tag{25}
$$

其中 $t$ 表示时间，$i$ 表示 $ω$ 和  $\mathbf{v}$ 中的个体元素。

定义了系统动力学后，我们希望将物理解释与概率解释联系起来。这可以通过 `正则分布（canonical distribution）`来实现：

$$
P(\boldsymbol{\omega}, \mathbf{v})=\frac{1}{Z} \exp (-H(\boldsymbol{\omega}, \mathbf{v}))=\frac{1}{Z} \exp (-U(\boldsymbol{\omega})) \exp (-K(\mathbf{v})) \tag{26}
$$

其中 $Z$ 是归一化常数，$H(ω，\mathbf{v})$ 是公式 23 中定义的总能量。从该联合分布可看出，位置变量和动量变量相互独立。

我们的最终目标是找到可预测的矩和区间。对于贝叶斯来说，关键量是后验分布。因此，可以将我们想要采样的势能设置为：

$$
U(\boldsymbol{\omega})=-\log (p(\boldsymbol{\omega}) p(\mathcal{D} \mid \boldsymbol{\omega}))\tag{27}
$$

在 HMC 中，动能可以从一系列合适的函数中自由选择。但通常采用选取以原点为中心的对角高斯分布作为 $\mathbf{v}$ 的边缘分：

$$
K(\mathbf{v})=\mathbf{v}^{T} M^{-1} \mathbf{v}\tag{28}
$$

这里 $M$ 是一个对角矩阵，在该物理解释中，被认为是变量的“质量”。但需要注意，虽然该动能函数最为常用，但不一定是最合适的。`[55]` 综述了其他高斯动能的选择和设计，并着重做出了几何解释。同时必须强调，选择合适的动能函数仍然是一个开放的研究课题，尤其是非高斯函数的情况。

由于汉密尔顿动力学使总能量保持不变，当以无限精度实现时，所提出的动力学是可逆的。可逆性是满足详细平衡条件的充分性质，这是确保目标分布（试图从其采样的后验分布）保持不变所必需的。在实际应用中，变量离散化会产生数值误差。最常用的离散化方法是`跳步（LeapFrog）法` 。跳步法指定步长 $\epsilon$ 以及在可能接受更新前要使用的步骤 $L$ 。跳步法首先执行动量变量 $\mathbf{v}$ 的一半更新，接着是位置 $w$ 的完全更新，然后是动量的剩余一半更新 `[59]`，


$$
v_{i}\left(t+\frac{\epsilon}{2}\right)=v_{i}(t)+\frac{\epsilon}{2} \frac{d v_{i}}{d t}(v(t)) \tag{29}
$$

$$
w_{i}(t+\epsilon)=w_{i}(t)+\epsilon \frac{d w_{i}}{d t}(w(t)) \tag{30}
$$

$$
v_{i}(t+\epsilon)=v_{i}\left(t+\frac{\epsilon}{2}\right)+\frac{\epsilon}{2} \frac{d v_{i}}{d t}\left(v\left(t+\frac{\epsilon}{2}\right)\right)\tag{31}
$$

如果步长 $\epsilon$ 的取值能够使该动力系统保持稳定，则可以证明跳步法保持了汉密尔顿体积（总能量）。

对于使用式 22 近似的期望值，我们要求每个样本 $ω_i$ 独立于后续样本。可以通过使用多个跳步 $L$ 来实现这种独立性。采用这种方式，在 $L$ 个 $\epsilon$ 步长之后，会推荐新的位置，从而降低了样本之间的相关性，并允许更快地探索后验空间。Metropolis 步骤可以用来确定新推荐是否被接受为马尔可夫链中的最新状态 `[59]` 。

对于 `[38]` 建议的贝叶斯神经网络，引入超先验 $p(γ)$ 对先验参数精度和似然精度的方差进行建模。对先验参数使用高斯先验，并且设置似然也为高斯。则此时，$γ$ 的先验是伽玛分布，因此是条件共轭的。这允许使用 `Gibbs采样` 来执行对超参数的推断。然后使用 HMC 更新后验参数。然后，联合后验 $P(ω，γ|D)$ 的采样就在超参数的吉布斯采样步长和模型参数的汉密尔顿动力学之间交替进行。然后，我们展示了 HMC 在简单 BNN 模型中的优越性能，并与随机游走 MCMC  和 Langevin 方法进行了比较 `[38]`。

## 2.4 深层贝叶斯神经网络

在 `Neal`、`MacKay` 和 `Bishop` 于90年代提出早期工作之后，对贝叶斯神经网络的研究变少了。同其他神经网络系统类似，这在很大程度上是由于训练神经网络的计算需求太高所导致。神经网络能够以任意精度捕获任何函数，但准确地捕获复杂函数需要具有许多参数的大型网络。即使从传统的频率主义观点来看，训练如此庞大的网络很困难，而研究信息量更大的贝叶斯神经网络，计算需求会更高。

不过在证明了 GPU 可以加速训练大型网络后，人们对神经网络的研究热情又重新燃起。GPU 实现了在反向传播期间执行大规模线性代数并行，这种加速计算允许训练更深层次的网络。随着 GPU 在优化复杂网络方面的成熟以及此类模型取得的巨大成功，人们也对 BNN 重新产生了兴趣。

`现代 BNN 研究主要集中在变分推断方法上，因为这些问题可以使用反向传播方法来优化`。考虑到大多成功网络均为深层次网络，文献 [54，53] 中的原始变分推断方法（侧重于利用单个隐层的回归网络的解析近似）变得不适用。现代神经网络呈现出不同的体系结构，具有不同维度、隐藏层、激活函数和应用。需要在概率意义上审视神经网络的更一般的方法。

考虑到现代神经网络的大规模性，稳健的推断能力通常需要建立在大数据集上。而对于大数据集，完整的对数似然评估变得不可行。为解决该问题，产生了一种采用随机梯度下降（SGD），利用小批量数据来近似似然项的方法。这时变分目标就变成：

$$
\mathcal{L}(\boldsymbol{\omega}, \boldsymbol{\theta})=-\frac{N}{M} \sum_{i=1}^{N} \mathbb{E}_{q}\left[\log \left(p\left(\mathcal{D}_{i} \mid \boldsymbol{\omega}\right)\right)\right]+\mathrm{KL}\left(q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) \| p(\boldsymbol{\omega})\right) \tag{32}
$$

其中 $D_i⊂D$ ，并且每个子集 $D_i$ 的大小均为 $M$ 。这为训练期间利用大数据集提供了有效方法。在传递单个子集 $D_i$ 之后，应用反向传播来更新模型参数。这种似然的子采样会在推断过程中引入噪声，因此得名 SGD 。该噪声在所有单独子集的评估过程中会被平均掉 `[61]` 。SGD 是利用变分推断方法训练 NN 和 BNN 的最常用方法。

Graves 在 2011 年发表了一篇关于 BNN 研究复兴的关键论文 `[62]`，`《Practical variational inference for neural networks》`。这项工作提出了一种使用了因子分解的高斯近似后验的MFVB处理方法。其关键贡献是导数的计算。变分推断目标（即证据下界 ELBO 最大化）可被视为两个期望的总和，

$$
\mathcal{F}\left[q_{\theta}\right]=\mathbb{E}_{q}[\log (p(\mathcal{D} \mid \boldsymbol{\omega}))]-\mathbb{E}_{q}\left[\log q_{\boldsymbol{\theta}}(\boldsymbol{\omega})-\log p(\boldsymbol{\omega})\right] \tag{33}
$$

这两个期望是优化模型参数所需要的，同时意味着需要计算的梯度。该论文显示了如何使用 `[63]` 提出的高斯梯度特性对参数进行更新：
$$
\nabla_{\boldsymbol{\mu}} \mathbb{E}_{p(\boldsymbol{\omega})}[f(\boldsymbol{\omega})]=\mathbb{E}_{p(\boldsymbol{\omega})}\left[\nabla_{\boldsymbol{\omega}} f(\boldsymbol{\omega})\right] \tag{34}
$$

$$
\nabla_{\Sigma} \mathbb{E}_{p(\boldsymbol{\omega})}[f(\boldsymbol{\omega})]=\frac{1}{2} \mathbb{E}_{p(\boldsymbol{\omega})}\left[\nabla_{\boldsymbol{\omega}} \nabla_{\boldsymbol{\omega}} f(\boldsymbol{\omega})\right] \tag{35}
$$

MC 积分可以应用于公式 34 和 35 ，以近似均值和方差参数的梯度。该框架允许对 ELBO 进行优化，以推广到任何对数损失参数模型。

虽然解决了将变分推断应用于具有更多隐藏层的复杂 BNN 问题，但实际实现显示出性能不足，这归因于梯度计算的 MC 近似的巨大方差 `[64]` 。开发减少方差的梯度估计方法已成为变分推断中一个重要的研究课题 `[65]` 。推导梯度近似的两种最常见方法是打分函数估计器和路径导数估计器。

打分函数估计器依赖于对对数导数特性的使用，

$$
\frac{\partial}{\partial \theta} p(x \mid \theta)=p(x \mid \theta) \frac{\partial}{\partial \theta} \log p(x \mid \theta) \tag{36}
$$

利用这一性质，可以形成对期望导数的蒙特卡罗估计，这在变分推断中经常使用。

$$
\begin{aligned} \nabla_{\theta} \mathbb{E}_{q}[f(\omega)] &=\int f(\omega) \nabla_{\theta} q_{\theta}(\omega) \partial \omega \\ &=\int f(\omega) q_{\theta}(\omega) \nabla_{\theta} \log \left(q_{\theta}(\omega)\right) \partial \omega \\ & \approx \frac{1}{L} \sum_{i=1}^{L} f\left(\omega_{i}\right) \nabla_{\theta} \log \left(q_{\theta}\left(\omega_{i}\right)\right) \end{aligned} \tag{37}
$$



打分函数梯度估计的一个常见问题是它们表现出相当大的方差[65]。减少蒙特卡罗估计方差的最常见方法之一是引入控制变量[66]。

变分推断文献中常用的第二种梯度估计器是路径导数估计器。这项工作建立在 `重参数化技巧` `[67，68，69]` 基础上，其中随机变量被表示为确定性的和可微的表达式。例如，对于参数为 $θ={\mu，\sigma}$ 的高斯：

$$
\begin{align*} 
\boldsymbol{\omega} & \sim \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\sigma}^{2}\right) \\ \boldsymbol{\omega}=g(\boldsymbol{\theta}, \boldsymbol{\epsilon}) &=\boldsymbol{\mu}+\boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \tag{38}
\end{align*}
$$

其中 $\epsilon∼N(0，I)$ 和 $\odot$ 表示 Hadamard 积。使用这种方法可以对期望的蒙特卡罗估计进行有效抽样。正如文 `[68]` 所示，当 $ω=g(θ，\epsilon)$ 时，有 $q(ω|θ)dω=p(\epsilon)d\epsilon$， 因此，可以证明:


$$
\begin{align*}
\int q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) f(\boldsymbol{\omega}) d \boldsymbol{\omega} &=\int p(\boldsymbol{\epsilon}) f(\boldsymbol{\omega}) d \boldsymbol{\epsilon} \\ &=\int p(\boldsymbol{\epsilon}) f(g(\boldsymbol{\theta}, \boldsymbol{\epsilon})) d \boldsymbol{\epsilon} \\ \approx \frac{1}{M} \sum_{i=1}^{M} f\left(g\left(\boldsymbol{\theta}, \boldsymbol{\epsilon}_{i}\right)\right) &=\frac{1}{M} \sum_{i=1}^{M} f\left(\boldsymbol{\mu}+\boldsymbol{\sigma} \odot \boldsymbol{\epsilon}_{i}\right) \tag{39}
\end{align*}
$$

由于式 39 相对于 θ 是可微的，因此可使用梯度下降方法来优化该期望的近似。这是变分推断中的一个重要属性，因为变分推断的目标中包含通常难以处理的对数似然期望值。`重参数化技巧` 是路径梯度估计器的基础。路径估计因其比打分函数估计更低的方差而更受欢迎 `[68，65]` 。

对神经网络进行贝叶斯处理的一个关键好处是能够从模型及其预测中提取不确定性。这是最近在神经网络背景下引起高度兴趣的研究课题。通过将现有正则化技术（如Dropout[70]）与近似推断联系起来，已经发现了神经网络中关于不确定性估计的有前途的发展。丢弃（Dropout）是一种随机正则化技术，它是为解决点估计网络中常见的过拟合问题而提出的。在训练过程中，Dropout引入了一个独立的随机变量，该变量是伯努利分布的，并将每个单独的权重元素乘以该分布中的样本。例如，实现Dropout的简单MLP是这样的形式，

$$
\rho_{u} \sim \operatorname{Bernoulli}(p) \notag
$$

$$
\phi_{j}=\theta\left(\sum_{i=1}^{N_{1}}\left(x_{i} \rho_{u}\right) w_{i j}\right) \tag{40}
$$

由式 40 可以看出，Dropout的应用 `以与重参数化技巧类似的方式`将随机性引入网络参数。一个关键区别是：在Dropout 情况下，随机性被引入到输入空间，而不是贝叶斯推理所需的参数空间。`Yarin Gal`  `[39]` 证明了这种相似性，并演示了如何将 Dropout 引入的噪声有效地传递到网络权重：

$$
\mathbf{W}_{\rho}^{1} =\operatorname{diag}(\boldsymbol{\rho}) \mathbf{W}^{1} \tag{41}
$$

$$
\boldsymbol{\Phi}_{\rho}=a\left(\mathbf{X}^{T} \mathbf{W}_{\rho}^{1}\right) \tag{42}
$$

其中 $ρ$ 是从伯努利分布采样的向量，$\operatorname{diag}(·)$ 运算符从向量创建平方对角矩阵。如此可以看出，一个 Dropout 变量在权重矩阵的每一行之间被共享，从而允许维持行间的某些相关性。通过查看权重参数的随机分量，该公式适用于使用变分框架的近似推断。在这项工作中，近似后验是伯努利分布与权重乘积的形式。

应用重参数化技巧获得相对于网络参数的偏导数，然后形成 ELBO 并执行反向传播以最大化下界。MC 积分被用来逼近解析上难以处理的对数似然。通过用两个小方差高斯分布的混合模型来近似伯努利后验，得到 ELBO 中近似后验与先验分布之间的KL散度。

在这项工作同时，`Kingma`等人[71] 也发现了 Dropout 和其在变分框架中的应用潜力。与 Dropout 引入的典型伯努利分布随机变量相比，`[71]` 将注意力集中在引入高斯随机变量 `[72]` 。文中表明在选择与参数无关的适当先验情况下，使用 Dropout 的神经网络可被视为近似推断。

Kingma等人还希望使用改进的局部重参数化来降低随机梯度中的方差。这不是在应用仿射变换之前从权重分布进行采样，而是在之后执行采样。例如，考虑MFVB的情况，其中假设每个权重是独立的高斯 $W_{ij}∼N(µ_{ij}，σ^2_{ij})$。在仿射变换 $\phi_j=\sum_{i=1}^{N_1}(x_iρ_i)w_{ij}$ 之后， $\phi_j$ 的条件后验分布也将是因子分解的高斯形式：


$$
\begin{align*}
q\left(\phi_{j} \mid \mathbf{x}\right) &=\mathcal{N}\left(\gamma_{j}, \delta_{j}^{2}\right) \tag{43} \\
\gamma_{j} &=\sum_{i=1}^{N} x_{i} \mu_{i, j} \tag{44}\\
\delta_{j}^{2} &=\sum_{i=1}^{N} x_{i}^{2} \sigma_{i, j}^{2}\tag{45}
\end{align*}
$$
相对于权重 $w$ 本身的分布，从 $\phi$ 的分布中采样更有利，因为这使得梯度估计器的方差与训练期间使用的小批次数量呈线性关系。

上述工作对于解决机器学习研究中缺乏严谨性的问题很重要。例如，最初的 Dropout 论文 `[70]` 缺乏任何重要的理论基础。相反，该方法引用了有性繁殖理论`[73]`作为方法动机，并在很大程度上依赖于所给出的实证结果。这些结果在许多高影响力的研究项目中得到了进一步的证明，这些项目仅仅将该技术作为一种正规化方法来使用。`[39]` 和 `[71]` 中的工作表明，该方法有理论上的合理性。在试图减少网络过拟合影响时，频率主义方法论依赖于弱合理性的成功经验，而贝叶斯分析提供了丰富的理论体系，导致对神经网络强大近似能力的有意义理解。

虽然解决了将变分推断应用于具有更多隐层的复杂BNN问题，但实际实现已经显示出性能不足，这归因于梯度计算的MC近似的巨大方差。`Hernandez` 等人 `[64]` 承认了这一局限性，并提出了一种新的BNN实用推断方法，名为概率反向传播(PBP)。PBP 偏离了典型的变分推断方法，取而代之的是采用 `假设密度滤波（Assumed Density  Filtering,ADF)`  方法 `[74]`。在该格式中，通过应用贝叶斯规则以迭代方式更新后验概率：
$$
p\left(\boldsymbol{\omega}_{t+1} \mid \mathcal{D}_{t+1}\right)=\frac{p\left(\boldsymbol{\omega}_{t} \mid \mathcal{D}_{t}\right) p\left(\mathcal{D}_{t+1} \mid \boldsymbol{\omega}_{t}\right)}{p\left(\mathcal{D}_{t+1}\right)} \tag{46}
$$


与以预测误差为目标函数的传统网络训练不同，PBP使用前向传播来计算目标的对数边缘概率，并更新网络参数的后验分布。在 `[75]` 中定义的矩匹配方法使用了反向传播的变种来更新后验，同时在近似分布和变分分布之间保持等效均值和方差：
$$
\begin{align*}
\mu_{t+1} &=\mu_{t}+\sigma_{t} \frac{\partial \log p\left(\mathcal{D}_{t+1}\right)}{\partial \mu} \tag{47}\\ 
\sigma_{t+1} &=\sigma_{t}+\sigma_{t}^{2}\left[\left(\frac{\partial p\left(\mathcal{D}_{t+1}\right)}{\partial \mu_{t}}\right)^{2}-2 \frac{\partial p\left(\mathcal{D}_{t+1}\right)}{\partial \sigma}\right] \tag{48}
\end{align*}
$$
在多个小数据集上的实验结果表明，与简单回归问题的 HMC 方法相比，该方法在预测精度和不确定性估计方面具有合理的性能 [64] 。这种方法的关键问题是在线训练方法带来的计算瓶颈。该方法可能适用于某些应用，或者适用于在现有BNN可用时用额外的附加数据更新现有BNN，但是对于大数据集推断，该方法在计算性能上令人望而却步。

Blundell等人提出了一种很有前途的BNN近似推断方法，名为 “`Bayes by Backprop`” `[76]`。该方法利用重参数化技巧来显示如何找到期望导数的无偏估计。对于可重参数化为确定性且可微函数 $ω=g(\epsilon，θ）$ 的随机变量 $ω \sim q_\theta(\omega)$，任意函数 $f(ω,θ)$ 的期望的导数可表示为：
$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{\theta}} \mathbb{E}_{q}[f(\boldsymbol{\omega}, \boldsymbol{\theta})] &=\frac{\partial}{\partial \boldsymbol{\theta}} \int q_{\boldsymbol{\theta}}(\boldsymbol{\omega}) f(\boldsymbol{\omega}, \boldsymbol{\theta}) d \boldsymbol{\omega} \tag{49}\\ 
&=\frac{\partial}{\partial \boldsymbol{\theta}} \int p(\boldsymbol{\epsilon}) f(\boldsymbol{\omega}, \boldsymbol{\theta}) d \boldsymbol{\epsilon} \tag{50}\\ 
&=\mathbb{E}_{q(\epsilon)}\left[\frac{\partial f(\boldsymbol{\omega}, \boldsymbol{\theta})}{\partial \boldsymbol{\omega}} \frac{\partial \boldsymbol{\omega}}{\partial \boldsymbol{\theta}}+\frac{\partial f(\boldsymbol{\omega}, \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] \tag{51}
\end{align*}
$$


在 `·Bayes by Backprop` 算法中，函数 $f(ω,θ)$ 被设为：
$$
f(\boldsymbol{\omega}, \boldsymbol{\theta})=\log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega})}-\log p(\mathbf{X} \mid \boldsymbol{\omega}) \tag{52}t
$$
这个 $f(ω，θ)$ 可被视为式 17 中执行的期望的自变量，它是下界的一部分。组合公式 51 和 52 ：
$$
\mathcal{L}(\boldsymbol{\omega}, \boldsymbol{\theta})=\mathbb{E}_{q}[f(\boldsymbol{\omega}, \boldsymbol{\theta})]=\mathrm{e}_{q}\left[\log \frac{q_{\boldsymbol{\theta}}(\boldsymbol{\omega})}{p(\boldsymbol{\omega})}-\log p(\mathcal{D} \mid \boldsymbol{\omega})\right]=-\mathcal{F}\left[q_{\boldsymbol{\theta}}\right] \tag{53}
$$
是 ELBO 的相反数，意味着 `Bayes By Backprop` 旨在最小化近似后验和真实后验之间的 KL 散度，可以使用蒙特卡罗积分来近似公式 53 中的成本：
$$
\mathcal{F}\left[q_{\boldsymbol{\theta}}\right] \approx \sum_{i=1}^{N} \log \frac{q_{\boldsymbol{\theta}}\left(\boldsymbol{\omega}_{i}\right)}{p\left(\boldsymbol{\omega}_{i}\right)}-\log p\left(\mathbf{X} \mid \boldsymbol{\omega}_{i}\right)\tag{54}
$$
其中 $ω_i$ 是来自 $q_θ(ω)$ 的 第 $i$ 个样本。通过公式 54 中的近似，可以使用公式 51 所示结果来找到无偏梯度。

对于 `Bayes by Backprop` 算法，假设一个完全因子分解的高斯后验，使得 $θ={\mu，\rho}$ ，其中 $\sigma=\text{softplus}(\rho) $  用于确保标准偏差参数为正。由此，网络中的权重分布 $ω∼\mathcal{N}(\mu，softplus(\rho)^2)$ 被重参数化为：
$$
\boldsymbol{\omega}=g(\boldsymbol{\theta}, \boldsymbol{\epsilon})=\mu+\operatorname{softplus}(\boldsymbol{\rho}) \odot \boldsymbol{\epsilon} \tag{55}
$$
在该贝叶斯神经网络中，可训练参数为 $\mu$ 和 $\rho$ 。由于使用了全因子分解分布，根据公式 20，近似后验的对数可表示为：
$$
\log q_{\boldsymbol{\theta}}(\boldsymbol{\omega})=\sum_{l, j, k} \log \left(\mathcal{N}\left(w_{l j k} ; \mu_{l j k}, \sigma_{l j k}^{2}\right)\right) \tag{56}
$$


算法 1 描述了完整的 `Bayes by Backprop`  算法。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210527134042_a1.webp)



## 2.5 BNN 的高斯过程特性

Neal[38] 还给出了推导和实验结果，以说明对于只有一个隐层的网络，当隐藏单元的数量接近无穷大时，会出现高于网络输出的高斯过程，并且将高斯先验置于参数 22 之上。图 6 说明了该结果。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210527170036_e5.webp)

图6：当在参数上放置高斯先验时，随着网络规模的增加，先验在输出上诱导的图示。其中图中的每个点对应于一个网络的输出，参数从先验分布中进行采样。对于每个网络，隐藏单元的数量是图（a）对应着 1 个单元，图（b）对应着 3 个单元，图（c）对应着 10 个单元，图（d）对应着 100 个单元。



从公式1和公式2可以看出NNS和高斯过程之间的这种重要联系。从这些表达式可以看出，具有单个隐藏层的NN是应用于输入数据的N个参数基函数的总和。如果方程1中每个基函数的参数是R.V.，则方程2成为R.V.的和。在中心极限定理下，随着隐层数N→∞，输出变为高斯。由于输出随后被描述为基函数的无穷和，因此可以看到输出变成了高斯过程。根据这一结果的完整推导和图6中所示的插图，[38]显示了如何在有限的计算资源下实现近似高斯性质，以及如何保持该和的大小。威廉姆斯随后演示了如何针对不同的激活函数分析协方差函数的形式[77]。高斯过程与具有单一隐层功的无限宽网络之间的关系最近已扩展到深层网络的情况[78]。



这种联系的识别激发了在BNN中的许多研究工作。高斯过程提供了许多我们希望获得的特性，例如可靠的不确定性估计、可解释性和鲁棒性。高斯过程提供了这些好处，但代价是随着数据集大小的增加，预测性能和所需的计算资源呈指数级增长。高斯过程和BNN之间的这种联系推动了这两个建模方案的合并；既保持了NNS中看到的预测性能和灵活性，又融入了高斯过程带来的稳健性和概率属性。这导致了深高斯过程的发展。

深度高斯过程是单个高斯过程的级联，其中很像NN，前一个高斯过程的输出用作新高斯过程的输入[79，80]。这种高斯过程的堆叠允许从高斯过程 23的组合中学习非高斯密度。高斯过程的一个关键挑战是适应大数据集，因为单个高斯过程的Gram矩阵的维度与数据点的数量是平方的。由于级联中的每个单独高斯过程都会产生一个独立的Gram矩阵，因此这个问题会被深度高斯过程放大。此外，由于产生的函数中的非线性，深度高斯过程的边际可能性在分析上是难以处理的。在[82]工作的基础上，Damianou和Lawrence[79]使用VI方法来创建易于处理的近似，并将计算复杂度降低到稀疏高斯过程中常见的计算复杂度[83]。



深入的高斯过程已经展示了高斯过程如何从NNS中看到的方法中受益。Gal和Ghahramani[84，85，86]是在这项工作的基础上建立的，以说明如何用BNN24来近似Deep 高斯过程。这是一个预期的结果；鉴于Neal[38]发现具有单个隐藏层的无限宽网络收敛到高斯过程，通过连接多个无限宽的层，我们收敛到深度高斯过程。

除了对深度高斯过程的分析外，[84，85，86]在[77]工作的基础上分析了BNN中使用的现代非线性激活与高斯过程的协方差函数之间的关系。这是一项有希望的工作，可能允许在神经网络中更原则性地选择激活功能，类似于高斯过程的激活功能。哪些激活函数会产生一个稳定的过程？我们流程的预期长度尺度是多少？这些问题或许可以用高斯过程现有的丰富理论来解决。

高斯过程属性不限于MLP BNN。最近的研究已经确定了在卷积BNN中诱导高斯过程性质的某些关系和条件[87，88]。这一结果是预期的，因为CNN可以被实现为具有在权重中实施的结构的MLP。这项工作确定的是，当实施这一结构时，高斯过程是如何构建的。Van der Wilk等人[89]提出了卷积高斯过程，它实现了一种类似于在CNN中看到的基于面片的操作来定义高斯过程优先于函数。这种方法的实际实现需要使用近似方法，因为评估大数据集的成本高得令人望而却步，甚至在每个面片上评估都是令人望而却步的。诱导点通过虚拟仪器框架形成，以减少需要评估的数据点数量和评估的面片数量。





由此可知在中心极限定理下，随着隐层数 N 逐渐增大，输出服从高斯分布。由于输出被描述为基函数的无穷和，因此可以将输出看作高斯过程。高斯过程和具有单个隐藏层工作的无限宽网络之间的关系最近已扩展到深度神经网络的当中，该联系的识别激发了贝叶斯神经网络的许多研究工作。

高斯过程提供了可靠的不确定性估计、可解释性和鲁棒性。但是高斯过程提供这些好处的代价是预测性能和随着数据集大小的增加所需的大量计算资源。高斯过程和贝叶斯神经网络之间的这种联系促使了两种建模方案的合并；既能维持神经网络的预测性能和灵活性，同时结合了高斯过程的的鲁棒性和概率性。

最近的研究已经确定了高斯过程属性不限于 MLP-贝叶斯神经网络，而且在卷积中也可以应用该属性。因为 C神经网络 可以实现为 MLP，其结构在权重中被强制执行。Vander Wilk 等人在 2003 年发表的论文《Convolutional gaussian processes》中提出了卷积高斯过程，它实现了一种类似于 C神经网络 中的基于 patch 的操作来定义 高斯过程 先验函数。

如下图所示，分析显示了高斯过程在预测偏差和方差方面的对比性能。用 Backprop 和一个因式高斯近似后验概率对 Bayes 模型进行训练，在训练数据分布的情况下，虽然训练数据区域外的方差与高斯过程相比显著低估，但预测结果是合理的。具有标度伯努利近似后验的 MC-Dropout 通常表现出更大的方差，尽管在训练数据的分布中保持不必要的高方差。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Xc神经网络rRXlJTmZIMUtYd1g0R0pMYlpPN0NTcHdneTJuUmZpY3ZkSTd2T2JUZnJjMzR5aWF2Njg2cnoxRENab29pYUpWSlZpYW1EYU9yaE1EeUtPZnQxQ2FoQmcvNjQw?x-oss-process=image/format,png)

## 2.6 当前 BNN 的局限性

虽然人们已经付出了很大的努力来开发在神经网络中执行推断的贝叶斯方法，但这些方法有很大的局限性，文献中还存在许多空白。一个关键的限制是严重依赖变分推断方法。在变分推断框架内，最常用的方法是平均场方法。MFVB 通过强制参数之间独立的强假设，提供了一种表示近似后验分布的方便方法。该假设允许使用因式分解分布来近似后验分布。这种独立假设大大降低了近似推断的计算复杂度，但代价是概率精度。

VI方法的一个共同发现是，结果模型过于自信，因为预测均值可能是准确的，而方差被大大低估了[90，91，92，51，93]。文献[2]的第10.1.2节和文献[35]的第21.2节描述了这一现象，这两个章节都附有举例和直观的数字来说明这一性质。这种低估方差的特性存在于当前BNN的大部分研究中[39]。最近的工作旨在通过使用噪声对比先验[94]和使用校准数据集[95]来解决这些问题。作者在[96]中使用了具体分布[97]来近似MC Dropout方法[85]中的Bernoulli参数，允许对其进行优化，从而得到更好校准的后验方差。尽管做出了这些努力，但在边界神经网络的VI框架内制定可靠和校准的不确定性估计的任务仍然没有解决。

有理由认为，当前VI方法的局限性可能受到所使用的近似分布的选择的影响，特别是独立高斯人通常使用的MFVB方法。如果使用更全面的近似分布，我们的预测会不会与我们已有和未曾见过的数据更加一致？对于一般的VI方法[98，49]，已经提出了基于混合的近似，尽管N个混合的引入增加了N的变分参数的数量。在BNN的情况下，引入了矩阵-正态近似后验[99]，这与满秩高斯相比减少了模型中的变分参数的数量，尽管这项工作仍然考虑了单个权重，这意味着没有建模协方差结构25。MCDropout在折衷低熵近似后验的情况下，能够在权矩阵行内保持一定的相关性信息。

最近提出了一种用于VI的方法，通过使用归一化流动来捕捉更复杂的后验分布[100,101]。在归一化流中，初始分布通过一系列可逆函数“流动”，以产生更复杂的分布。这可以使用摊销推理在VI框架内应用[102]。分期推理引入了一个推理网络，将输入数据映射到产生式模型的变量参数。然后，这些参数被用来从生成过程的后部取样。正常化流动的使用已扩展到BNN的情况[103]。这种方法出现了与计算复杂性相关的问题，以及分期推理的局限性。归一化流动需要计算雅可比的行列式，以应用用于每个可逆函数的变量的变化，这对于某些模型来说可能计算昂贵。通过将归一化流限制为包含数值稳定的可逆运算，可以降低计算复杂度[102,104]。这些限制已被证明严重限制了推理过程的灵活性，以及由此产生的后验近似的复杂性[105]。

如前所述，在VI框架中，选择近似分布，然后最大化ELBO。这个ELBO源于在真实后方和近似后方之间应用KL发散，但这回避了一个问题，为什么要使用KL呢？KL发散度是评估两个分布之间相似性的一个众所周知的度量，它满足发散度的所有关键性质(即，KL发散度，KL发散度)。为正，并且当两个分布相等时仅为零)。发散可以让我们知道我们的近似是否接近真实分布，但不能知道我们离真实分布有多近。为什么不用定义明确的距离来代替发散呢？

使用KL发散度是因为它允许我们从我们可以优化的目标函数(ELBO)中分离出难以处理的量(边际可能性)。我们的贝叶斯推理的目标是在先验知识和观测数据的分布下识别最适合我们模型的参数。VI框架将推理视为优化问题，我们优化参数以最小化近似分布和真实分布之间的KL偏差(这将最大化ELBO)。由于我们正在优化参数，通过将边际似然从目标函数中分离出来，我们能够计算关于易处理量的导数。由于边际似然与参数无关，当求导数时，该分量消失。这就是为什么使用KL发散度的关键原因，因为它允许我们从目标函数中分离出难以处理的量，然后在使用梯度信息执行优化时，目标函数将被评估为零。

KL发散已被证明是称为α-发散的一般发散族的一部分。α散度表示为，
$$
D_{\alpha}[p(\omega) \| q(\omega)]=\frac{1}{\alpha(1-\alpha)}\left(1-\int p(\omega)^{\alpha} q(\omega)^{1-\alpha} d \omega\right) \tag{57}
$$
在α→−1的极限中从公式57中找到VI中使用的正向KL发散，并且在期望传播期间使用的α→1的极限中出现反向KL发散KL(p||q)。虽然在VI中使用正向KL散度通常会导致低估方差，但是使用反向KL通常会高估方差[2]。类似地，当α=0时，海灵格距离从57开始，
$$
D_{H}(p(\omega) \| q(\omega))^{2}=\int\left(p(\omega)^{\frac{1}{2}}-q(\omega)^{\frac{1}{2}}\right)^{2} d \omega \tag{58}
$$
这是一个有效距离，因为它满足三角形不等式并且是对称的。与两个KL发散相比，Hellinger距离的最小化在方差估计上提供了合理的折衷[107]。虽然这些措施可能提供理想的质量，但它们不适合在VI中直接使用，因为难以处理的边际可能性不能与其他利益项分开26。虽然这些度量不能立即使用，但它说明了客观度量的变化如何导致不同的近似。通过对目标函数使用不同的度量，可以找到更准确的后验期望。这是一个有效距离，因为它满足三角形不等式并且是对称的。与两个KL发散相比，Hellinger距离的最小化在方差估计上提供了合理的折衷[107]。虽然这些措施可能提供理想的质量，但它们不适合在VI中直接使用，因为难以处理的边际可能性不能与其他利益项分开26。虽然这些度量不能立即使用，但它说明了客观度量的变化如何导致不同的近似。通过对目标函数使用不同的度量，可以找到更准确的后验期望。

绝大多数现代作品都围绕着VI的概念展开，这在很大程度上要归功于它对SGD的顺应性。现在存在复杂的工具来简化和加速自动区分和反向传播的实现[108,109,110,111,112,113,114]。VI的另一个好处是它在可能性上接受二次抽样。子采样减少了对当前可用的大数据集进行训练所需的推理执行的计算开销。这就是更传统的基于MCMC的方法在BNN社区受到的关注明显较少的关键原因

MCMC具有丰富的理论发展、渐近保证和实用的收敛诊断能力，是进行贝叶斯推理的金标准。传统的基于MCMC的方法需要从完全联合似然抽样来执行更新，要求在提出任何新建议之前看到所有训练数据。次抽样MCMC或随机梯度MCMC(SG-MCMC)方法已在[61,115,116]中提出，并已应用于BNN[117]。已有研究表明，MCMC内的幼稚子抽样将使随机更新的轨迹偏离后验[118]。这种偏向消除了传统MCMC方法在理论上的优势，使得它们不如计算成本通常较低的VI方法。为了使抽样方法变得可行，需要发展确保收敛于后验分布的次抽样方法。



# 3 现代 BNN 的比较

## 3.1 普通 BNN

从文献综述来看，贝叶斯神经网络中两种重要的近似推断方法是 Backprop 的 Bayes[76] 和 MC Dropout[85]。这些方法被认为是贝叶斯神经网络中最有前途、影响最大的近似推断方法。这两种变分推断方法都足够灵活，可以使用 SGD，从而使部署到大型、实用的数据集成为可能。鉴于这些方法的突出之处，有必要对这些方法进行比较，看看它们的表现如何。

为了比较这些方法，进行了一系列简单的同方差回归任务。对于这些回归模型，概率用高斯表示。有了这个，我们就可以写出未归一化的后验是，
$$
p(\boldsymbol{\omega} \mid \mathcal{D}) \propto p(\boldsymbol{\omega}) \mathcal{N}\left(\mathbf{f}^{\boldsymbol{\omega}}(\mathcal{D}), \sigma^{2} \mathbf{I}\right) \tag{59}
$$
其中fω(D)是由BNN表示的函数。在这两个模型中，混合高斯被用来对钉板进行建模。然后，使用所提出的各自的方法，求出每个模型的近似后验Q(θ(ω)。对于Backprop的贝叶斯，近似的后验分布是完全分解的高斯分布，而对于MC Dropout，近似的后验分布是缩放的伯努利分布。利用每个模型的近似后验信息，可以使用MC积分找到预测量。前两个时刻可以近似为[39]，
$$
\begin{align*} \mathbb{E}_{q}\left[\mathbf{y}^{*}\right] & \approx \frac{1}{N} \sum_{i=1}^{N} \mathbf{f}^{\boldsymbol{\omega}_{i}}\left(\mathbf{x}^{*}\right) \tag{60}\\
\mathbb{E}_{q}\left[\mathbf{y}^{* T} \mathbf{y}^{*}\right] & \approx \sigma^{2} \mathbf{I}+\frac{1}{N} \sum_{i=1}^{N} \mathbf{f}^{\boldsymbol{\omega}_{i}}\left(\mathbf{x}^{*}\right)^{T} \mathbf{f}^{\boldsymbol{\omega}_{i}}\left(\mathbf{x}^{*}\right) \tag{61}
\end{align*}
$$


其中星号上标表示来自测试集中的新输入和输出样本x∗，y∗。

用于评估这些模型的数据集是来自高影响力论文的简单玩具数据集，其中提供了类似的实验作为经验证据[76,119]。然后将两种BNN方法与GP模型进行比较。图7显示了这些结果。

对图7中所示回归结果的分析显示，在预测中的偏差和方差方面表现不同。用Backprop的贝叶斯和分解后的高斯近似后验数据训练的模型显示出合理的训练数据分布预测结果，尽管与GP相比，训练数据区域外的方差被显著低估。具有缩放伯努利近似后验的MC Dropout对于非分布数据通常表现出更大的方差，尽管在训练数据的分布内保持不必要的高方差。对这些模型的超参数进行了很少的调整。通过更好地选择超参数，可以获得更好的结果，特别是对于MC Dropout。或者，可以使用更完整的贝叶斯方法，其中将超参数视为潜在变量，并对这些变量执行边际化。

值得注意的是，这些方法在计算和实际应用中都遇到了困难。MC Dropout方法非常灵活，因为它对先验分布的选择不那么敏感。它还设法用更少的样本和训练迭代来适应更复杂的分布。最重要的是显著节省了计算资源。考虑到使用MC Dropout训练一个模型通常与训练多少个现有的深层网络相同，因此推理与传统的普通网络同时进行。它也没有增加网络的参数数量，而Backprop的贝叶斯需要两倍的参数。在实际情况下应考虑这些因素。如果被建模的数据是平滑的，有足够的数量，并且允许额外的时间进行推理，使用Backprop的贝叶斯可能更可取。对于功能复杂、数据稀疏、时间要求较严格的大型网络，MC Dropout可能更合适。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210527171044_a5.webp)

图7：在三个玩具数据集上的回归任务中，BNN和GP的比较。顶行是由Backprop[76]用贝叶斯训练的BNN，中间行用MC退学[39]训练，底部是具有安装有GPflow包的Mattern52核的GP[120]。这两个BNN由利用RELU激活的两个隐藏层组成。训练数据用深灰色散点表示，平均值用紫色表示，真实测试函数用蓝色表示，阴影区域表示±一个和两个标准差。从中庸之道。最好在电脑屏幕上观看。



## 3.2 卷积 BNN

虽然 MLP 是神经网络的基础，但最突出的神经网络架构是卷积神经网络。这些网络在具有挑战性的图像分类任务方面表现出色，其预测性能远远超过先前基于核或特征工程的方法。C神经网络 不同于典型的 MLP，它的应用是一个卷积型的算子，单个卷积层的输出可以表示为：
$$
\boldsymbol{\Phi}=u\left(\mathbf{X}^{T} * \mathbf{W}\right) \tag{62}
$$
其中u(·)是非线性激活，∗表示类卷积运算。这里，输入X和权重矩阵W不再局限于向量或矩阵，而是可以是多维数组。可以表明，CNN可以被写成具有等效的MLP模型，从而允许优化的线性代数包用于利用反向传播进行训练[122]。

在现有研究方法的基础上，可以发展一种新型的贝叶斯卷积神经网络(BCNN)。这是通过将贝叶斯的Backprop方法[76]扩展到适合于图像分类的模型的情况来实现的。卷积层中的每个权重被假定为独立的，从而允许对每个单独的参数进行因式分解。

通过实验研究了BCNN的预测性能及其不确定性估计的质量。这些网络被配置用于MNIST手写数字数据集的分类[123]。

由于该任务是分类任务，因此BCNN的可能性被设置为SoftMax函数，

$$
\operatorname{softmax}\left(\mathbf{f}_{i}^{\omega}\right)=\frac{\mathbf{f}_{i}^{\omega}(\mathcal{D})}{\sum_{j} \exp \left(\mathbf{f}_{j}^{\omega}(\mathcal{D})\right)} \tag{63}
$$

然后未归一化的后验可以表示为，
$$
p(\boldsymbol{\omega} \mid \mathcal{D}) \propto p(\boldsymbol{\omega}) \times \operatorname{softmax}\left(\mathbf{f}^{\boldsymbol{\omega}}(\mathcal{D})\right) \tag{64}
$$
然后利用Backprop的贝叶斯求出近似的后验概率。可以使用公式60找到测试样本的预测平均值，并且使用MC积分来近似可信区间[35]。

并与普通CNN进行了比较，评价了BCNN的预测性能。对于香草和BCNN，都使用了流行的LeNet架构[123]。使用BCNN的平均输出进行分类，并使用可信区间来评估模型的不确定性。在MNIST数据集中的10,000张测试图像上，这两个网络的总体预测性能显示出相当的性能。BCNN的测试预测准确率为98.99%，而香草网络的预测准确率为99.92%，略有提高。虽然竞争性预测性能是必不可少的，但BCNN的主要好处是我们提供了有关预测不确定性的有价值的信息。难以分类的数字示例显示在附录中，并附有平均预测值和每类95%可信区间的曲线图。从这些例子中，我们可以看到这些具有挑战性的图像的大量预测不确定性，这些不确定性可以用来在实际场景中做出更明智的决策。

这种不确定性信息对于许多感兴趣的场景来说都是无价的。随着统计模型越来越多地用于包含人类交互的复杂任务，这些系统中的许多系统基于其感知的世界模型做出负责任的决策是至关重要的。例如，神经网络在自动驾驶汽车的开发中被大量使用。由于场景的高度可变性和与人类互动相关的复杂性，自动驾驶汽车的开发是一项令人难以置信的具有挑战性的壮举。目前的技术不足以安全地实现这项任务，正如前面讨论的那样，这些技术的使用已经导致多人死亡[24，25]。在这样一个高度复杂的系统中对所有变量进行建模是不可能的。这伴随着不完美的模型和对近似推理的依赖，重要的是我们的模型可以传达与决策相关的任何不确定性。至关重要的是，我们必须承认，从本质上讲，我们的模型是错误的。这就是为什么概率模型在这种情况下更受青睐的原因；有一个基本的理论来帮助我们处理数据中的异质性，并解释模型中没有包括的变量引起的不确定性。至关重要的是，用于此类复杂场景的模型在用于此类复杂和高风险场景时能够传达其不确定性。



## 3.3 循环 BNN



# 4 结论

本报告阐明了典型的神经网络和特殊模型设计中的过度自信预测问题，而贝叶斯分析被证明可用来解决这些挑战。尽管对于贝叶斯神经网络来说，精确推断仍然是分析和计算上的难题，但实践表明，可以依靠近似推断方法获得较为精确的近似后验。

贝叶斯神经网络中的许多近似推断方法都围绕 MFVB 方法展开。这为优化变分参数提供了一个易于处理的下限。这些方法在易用性、预测均值的准确性、可接受的参数数目等方面具有吸引力。文献调查和实验结果表明，在完全分解的 MFVB 方法中所做的假设导致了过度自信的预测。结果表明，这些 MFVB 方法可以推广到更复杂的模型，如卷积神经网络。实验结果表明，对于图像分类任务，贝叶斯预测性能与基于点估计的卷积神经网络相当。贝叶斯卷积神经网络能够为预测提供可信的区间，而这些预测是对难以分类的数据点提供的高度信息性和直观性的不确定性度量。

这篇综述和实验突出了贝叶斯分析解决机器学习社区中常见挑战的能力。这些结果还突显了当前用于贝叶斯神经网络的近似推断方法的不足，甚至可能提供不准确的方差信息。不仅要确定网络是如何运行的，而且要确定现代大型网络如何才能实现精确推断，这尚有待进一步研究。将 MCMC 等推断方法扩展到大数据集上允许更有原则性的推断。MCMC 提供了评估收敛和推断质量的诊断方法。对变分推断的类似诊断允许研究人员和实践者评估他们假设后验的质量，并告知他们改进该假设的方法。实现这些目标将使我们能够获得更精确的后验近似。由此，我们将能够充分确定模型知道什么，也可以确定模型不知道什么。



# 参考文献



1. F. Rosenblatt, “The perceptron: A probabilistic model for information storage and
organization in the brain.” Psychological Re变分推断ew, vol. 65, no. 6, pp. 386 – 408, 1958.
2. C. Bishop, Pattern recognition and machine learning. New York: Springer, 2006.
3. D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning representations by
back-propagating errors,” nature, vol. 323, no. 6088, p. 533, 1986.
4. K.-S. Oh and K. Jung, “GPU implementation of neural networks,” Pattern Recogni-
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
conference on computervision and pattern recognition, 2014, pp. 580–587.
10. S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object de-
tection with region proposal networks,” in Advances in neural information processing
systems, 2015, pp. 91–99.
11. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified,
real-time object detection,” in Proceedings of the IEEE conference on computervision
and pattern recognition, 2016, pp. 779–788.
12. A. Mohamed, G. E. Dahl, and G. Hinton, “Acoustic modeling using deep belief
networks,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 20,
no. 1, pp. 14–22, 2012.
13. G. E. Dahl, D. Yu, L. Deng, and A. Acero, “Context-dependent pre-trained deep neu-
ral networks for large-vocabulary speech recognition,” IEEE Transactions on audio,
speech, and language processing, vol. 20, no. 1, pp. 30–42, 2012.
14. G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Van-
houcke, P. Nguyen, T. N. Sainath et al., “Deep neural networks for acoustic modeling
in speech recognition: The sharedviews of four research groups,” IEEE Signal Pro-
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
V. Sohal, A. Widge, H. Mayberg, G. Sapiro, and K. Dzirasa, “A sharedvision for
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
61. M. Welling and Y. Teh, “Bayesian learningvia stochastic gradient langevin dynam-
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
98. T. S. Jaakkola and M. I. Jordan, “Improving the mean field approximationvia the
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
I. Sutskever, K. Talwar, P. A. Tucker, V. Vanhoucke, V. Vasudevan, F. B.vi´ egas,
O.vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, “Tensor-
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
119. I. Osband, C. Blundell, A. Pritzel, and B. V. Roy, “Deep explorationvia bootstrapped
DQN,” CoRR, vol. abs/1602.04621, 2016.
120. A. G. d. G. Matthews, M. van der Wilk, T. Nickson, K. Fujii, A. Boukouvalas,
P. Le´  on-villagr´  a, Z. Ghahramani, and J. Hensman, “GPflow: A Gaussian process
library using TensorFlow,” Journal of Machine Learning Research, vol. 18, no. 40,
pp. 1–6, 4 2017.
121. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and
L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural
computation, vol. 1, no. 4, pp. 541–551, 1989.
122. I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT Press, 2016.
123. Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to
document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov
1998.