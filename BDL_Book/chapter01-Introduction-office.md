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

本文的目的是向读者提供容易理解的贝叶斯神经网络介绍，同时伴随该领域的一些开创性工作和实验的调研，以激发对当前方法的能力和局限性的讨论。涉及贝叶斯神经网络的资料太多，无法一一列举，因此本文仅列出了其中具有里程碑性质的关键项目。同样，许多关键成果的推导也被省略了，仅列出了最后结果，并附有对原始来源的引用。我们也鼓励受相关研究启发的读者参考之前的一些调研报告：[32] 调查了贝叶斯神经网络的早期发展，[33] 讨论了对神经网络进行全面贝叶斯处理的细节，以及 [34] 调查了近似贝叶斯推理在现代网络结构中的应用。

本文应该适合所有统计领域的读者，但更感兴趣的读者可能是那些熟悉机器学习概念的人。尽管机器学习和贝叶斯概率方法都很重要[2，35]，但实践中许多现代机器学习方法和贝叶斯统计研究之间存在分歧。希望这项调研能够有助于突出贝叶斯神经网络的现代研究与统计学之间的相似之处，强调概率观点对机器学习的重要性，并促进机器学习和统计学领域未来的合作。

# 2 文献调研

## **2.1  神经网络**

在讨论神经网络的贝叶斯观点之前，简要介绍神经计算的基本原理并定义本章要使用的符号是很重要的。本调查将重点介绍感兴趣的主要网络结构-多层感知器(MLP)网络。MLP是NNS的基础，现代体系结构如卷积网络具有等价的MLP表示。图2显示了一个简单的MLP，它有一个适合于回归或分类的隐藏层。对于具有维度n1的输入x的该网络，f网络的输出可以被建模为，



## 2.2贝叶斯神经网络

在频率主义框架内，模型权重不被视为随机变量，而是假设权重是一个确切的值，只是其值尚未知，同时将数据视为随机变量。这似乎有些违反解决问题的原始直觉，因为我们想知道根据手头信息，未知模型的权重是多少。与频率学派不同，贝叶斯统计建模视数据为可获得的信息，而将未知的权重视为随机变量。其基本逻辑是；未知（或潜在）参数被视为随机变量，希望在可观测的训练数据支持下，了解参数的分布。

在贝叶斯神经网络的“学习”过程中，未知的模型权重可以在已知信息和所观测到的信息基础上推断出来。这是个逆概率问题，可以贝叶斯定理来解决。我们的模型中权重 $ω$ 是隐藏或潜在的变量，通常无法立即观察到其真实分布；而贝叶斯定理允许利用可观测的概率来表示权重的分布，形成以观测数据为条件的权重分布 $p(ω|D)$，我们称之为后验分布。

在训练前，可以观察权重和数据之间的联合分布 $p(ω，D)$ 。该联合分布由我们对潜在变量的先验信念 $p(ω)$ 和我们对模型/似然的选择 $p(D|ω)$ 来定义：

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

贝叶斯推断过程围绕着对未知模型权重的边缘化（集成）展开。通过边缘化方法，能够了解模型的生成过程，而不是在频率主义框架中使用的优化方案。通过使用生成模型，预测将以有效的条件概率形式表示。

在上例中假设诸如噪声方差 $σ$ 等参数或者所有先验参数是已知的，但实践中很少会出现此情况，因此需要对这些未知变量进行推断。贝叶斯框架允许对其进行推断，类似于对权重的推断方式；将额外的变量视为潜在变量，分配一个先验分布（有时称为 `超先验分布`），然后对其进行边缘化处理，以找到后验分布。有关如何对贝叶斯神经网络执行此操作的更多说明，请参考 [33，38]。

对于许多感兴趣的模型，后验（式 8) 的计算仍然很困难。这在很大程度上是由边际似然的计算造成的。对于非共轭模型或存在潜在变量的非线性模型（如前馈神经网络），边际似然几乎没有解析解，对于高维模型的计算更为困难。因此，一般会对后验进行近似。以下部分详细说明了如何在贝叶斯神经网络中实现近似贝叶斯推断。

## 2.3贝叶斯神经网络的起源

### 2.3.1 BNN 的起源

根据本调研和之前的调查报告 `[39]`，可以被认为是贝叶斯神经网络的第一个实例是在 `[40]` 中开发的。该论文通过对神经网络损失函数的统计解释，强调了其主要统计特性，证明了平方误差最小化等价于求高斯函数的最大似然估计（MLE）。更重要的是，通过在网络权重上指定先验，可用贝叶斯定理获得适当的后验。虽然此工作提供了对神经网络贝叶斯观点的关键见解，但没有提供计算边际似然的方法，也就意味着没有提出任何实用的推断方法。`Denker` 和 `LeCun[41]` 对该工作进行了扩展，提供了一种使用拉普拉斯近似进行近似推断的实用方法。

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

图 4：证据在评估不同模型中发挥的作用。简单模型 $\mathcal{H}_1$ 能够以更大的强度预测较小范围的数据，而较复杂的模型 $\mathcal{H}_2$ 能够表示较大范围的数据，尽管概率较低。改编自 `[46，47]`。

使用这种证据框架需要计算边际似然，这是贝叶斯建模中最关键的挑战。考虑到接近边际可能性所需的大量的计算成本，比较许多不同的体系结构可能是不可行的。

尽管如此，证据框架的使用可以用来评估贝叶斯神经网络的解决方案。对于大多数感兴趣的神经网络结构，目标函数是非凸的，具有许多局部极小值。每一个局部极小都可以看作是推断问题的一个可能解。`MacKay` 以此为动机，使用相应的证据函数来比较来自每个局部最小值的解 `[48]` ，这允许在没有大量计算的情况下评估每个模型方案的复杂性。

### 2.3.2 早期 BNN 的变分推断

机器学习社区在基于优化的问题上一直表现出色。许多最大似然模型（如支持向量机和线性高斯模型）的目标函数都是凸函数，而神经网络的目标函数是高度非凸的，具有许多局部极小值。难以定位全局最小值催生了基于梯度的优化方法，如反向传播 `[3]` 。这种优化方法可以通过 `变分推断（Variational Inference ）` 的方式在贝叶斯上下文中进行实践。

VI 是一种近似推理方法，它将贝叶斯推理过程中所需的边际化定义为优化问题 [49，50，51]。这是通过假设后验分布的形式并执行优化以找到最接近真实后验的假设密度来实现的。该假设简化了计算，并提供了一定程度的可操作性。

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

式 19 中的符号，特别是包含 $\mathcal{F}[q_θ]$ 的负值，是为强调“与同一个结果不相同但等价的导数”，并与文献保持一致。该结果不是通过最小化真实分布和近似分布之间的 KL 散度，而是通过近似难以处理的对数边缘似然来获得的。

通过应用 Jensen 不等式，可以发现 $\mathcal{F}[q_θ]$ 形成了边际对数似然的下界 `[49,52]` 。通过公式 19 注意到： KL 散度严格 ≥ 0 且仅当两个分布相等时才等于零。边缘对数似然 $\log p(\mathcal{D})$ 等于近似后验与真实后验之间的 KL 散度与  $\mathcal{F}[q_θ]$  之和。通过最小化近似后验和真实后验之间的 KL 散度， $\mathcal{F}[q_θ]$  将越来越接近边缘对数似然。因此， $\mathcal{F}[q_θ]$  通常被称为证据下界（ELBO），图 5 可视化地说明了该点。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021052616535332.webp)

图 5：近似后验和真实后验之间的 KL 散度最小化导致的证据下界最大化。当近似后验和真实后验之间的 KL 散度被最小化时，证据下界 $\mathcal{F}[q_θ]$ 收紧到对数证据。因此，最大化证据下界 `ELBO` 等同于最小化 KL 散度。改编自 `[53]` 。

`Hinton` 和 `Van Camp` `[54]` 首次将变分推断应用于贝叶斯神经网络，试图解决神经网络中的过拟合问题。他们认为，通过使用模型权重的概率观点，其可包含的信息量会减少，并将简化网络。该问题的表述是通过信息论基础，特别是最小描述长度原则，尽管其应用导致了一个相当于变分推断的框架。正如变分推断中常见的那样，使用平均场方法。`平均场变分贝叶斯 (MFVB)` 假设后验分布分解感兴趣的参数。对于 `[54]` 中的工作，假设模型权重上的后验分布是独立高斯的因子分解，

$$
q_{\boldsymbol{\theta}}(\boldsymbol{\omega})=\prod_{i=1}^{P} \mathcal{N}\left(w_{i} \mid \mu_{i}, \sigma_{i}^{2}\right) \tag{20}
$$

其中 $P$ 是网络中权重的数量。对于只有一个隐层的回归网络，这个后验的解析解是可用的。能够获得近似解析解是一种理想特性，因为解析解极大减少了执行推断的时间。

这项工作存在几个问题，其中最突出的问题是假设后验因子分解了单个网络权重。众所周知，神经网络中的参数之间存在强相关性。因子分解分布通过牺牲参数之间丰富的相关性信息来简化计算。`Mackay` 在对贝叶斯神经网络的早期调研中强调了该局限性 `[32]`，并提供了对隐藏层输入的预处理阶段如何允许更全面的近似后验分布的洞察力。

`Barber` 和 `Bishop` `[53]` 再次强调了该局限性，并提出了一种基于变分推断的方法，该方法扩展了 `[54]` 中的工作，允许通过使用 `全秩高斯近似后验` 来捕获参数之间的完全相关性。对于利用 Sigmoid 激活的单隐层回归网络，提供了用于评估`ELBO` 的解析表达式。这通过用适当缩放的误差函数替换 Sigmoid 来实现。

此建模方案的一个问题是参数数量的增加。对于完全协方差模型，参数的数量与网络中的权重数量成二次函数关系。为纠正该点，`Barber` 和 `Bishop` 对因子分析中经常使用的协方差提出了一种限制形式，

$$
\mathbf{C}=\operatorname{diag}\left(d_{1}^{2}, \ldots, d_{n}^{2}\right)+\sum_{i=1}^{s} \mathbf{s}_{i} \mathbf{s}_{i}^{T} \tag{21}
$$

其中，$diag$ 运算符从大小为 $n$ 的向量 $d$ 创建对角线矩阵，其中 $n$ 是模型中权重的数量。然后，该表单随网络中隐藏单元的数量线性扩展。

这些工作为如何将显著的反向传播方法应用于挑战贝叶斯问题提供了重要的见解。使得这两个研究领域的特性可以合并，并提供名义上单独看到的好处。现在可以使用神经网络在概率意义上处理大量数据集的复杂回归任务。

尽管这些方法提供了洞察力，但这些方法也有局限性。`Hinton` 、 `Van Camp` 、 `Barber` 和 `Bishop` 的工作都集中在发展一种封闭形式的网络表示。这种分析处理能力对网络施加了许多限制。如前面所讨论的，`[54]` 假设后验因子分解超过单个权重，它不能捕获参数中的任何相关性。在 `[53]` 中捕获了协方差结构，尽管作者将他们的分析限制在使用 Sigmoid 激活函数（该函数由误差函数很好地近似）的情况下，该函数很少在现代网络中使用，因为梯度的幅度较低。这两种方法共有的一个关键限制是单个隐含层网络的限制。

如前所述，神经网络 可以通过添加额外的隐藏单元来任意地逼近任何函数。对于现代网络，经验结果表明，通过增加网络中的隐含层数，可以用更少的隐含单元来表示类似的复杂函数。这就产生了“深度学习”一词，其中深度指的是隐藏层的数量。当试图近似各层之间的全部协方差结构时，减少权重变量的数量尤其重要。例如，可以捕获单个层内的隐藏单元之间的相关性，同时假设不同层之间的参数是独立的。这样的假设可以显著减少相关参数的数量。随着现代网络在许多层上拥有数以亿计的权重（这些网络只能提供点估计），开发超出单层的实用概率解释的需求是必不可少的。

### 2.3.3 BNN 的混合蒙特卡罗方法

在该点上，值得反思一下实际感兴趣的数量。到目前为止，重点一直放在寻找后验的良好近似上，尽管后验的准确表示通常不是最终设计要求。感兴趣的主要数量是预测时刻和间隔。我们希望在提供信心信息的同时做出良好的预测。我们强调后验的原因是预测矩和区间都是根据后验π(ω|D)13 的期望值来计算的。该期望值列在方程式 9 中，为方便起见在这里重复

贝叶斯神经网络的重点是寻找良好的后验分布近似值上，预测值和区间都是作为后验 的期望值来计算的，其中精确的预测依赖于对难以处理的后验概率的精确近似。具体的计算公式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1OdnFqUzNDWGMyVEw2MXhoTU1pY09nTFlEb2s5a0dyYzEyRXFwVER2VVZDU0tiNU1PeDZkeVRRLzY0MA?x-oss-process=image/format,png)

上面的积分公式的求解很困难，以前的方法是通过基于优化的方案，但优化方法中设置的限制通常会导致预测值不准确，所以基于优化的方案可以提供不准确的预测量。为了在有限的计算资源下做出准确的预测，通过使用马尔可夫链蒙特卡罗（MCMC）方法来求解上积分。

MCMC 是一种可以从从任意和难以处理的分布中进行采样的通用方法，会有如下公式：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1KNEpQR1hKQnVEREswaWNMSWxVdDN4UlBId0VFeVlmdGV0SXVKSXIyaWNpYUFOMW5sR094T1ZsdmcvNjQw?x-oss-process=image/format,png)

传统的 MCMC 方法表现出一种随机游走行为，即序列是随机产生的。由于贝叶斯神经网络后验函数的复杂性和高维性，这种随机游走行为使得这些方法不适合在任何合理的时间内进行推断。为了避免随机行走行为，本文采用混合蒙特卡洛（HMC）方法用于将梯度信息合并到迭代行为中。

对于贝叶斯神经网络，首先引入超先验分布 来模拟先验参数精度，其中先验参数服从高斯先验。 上的先验服从伽马分布，并且它是条件共轭的，这就使得吉布斯抽样可以用于对超参数进行推断，然后可以使用 HMC 来更新后验参数，最后从联合后验分布 进行取样。

## 2.4 现代 BNN

在 Neal，MacKay 和 Bishop 于 90 年代提出的早期工作之后，对贝叶斯神经网络进行的研究要少得多。这种相对停滞在大多数神经网络研究中都可以看到，这在很大程度上是由于训练神经网络的计算需求很高。神经网络是能够以任意精度捕获任何函数的参数模型，但是准确地捕获复杂函数需要具有许多参数的大型网络。即使从传统的频域观点来看，训练如此庞大的网络也变得不可行，而为了研究信息量更大的贝叶斯网络，计算需求也大大增加。

考虑到网络的大规模性，强大的推断能力通常需要建立在大数据集上。对于大数据集，完全对数似然的评估在训练目的上变得不可行。为了解决该问题，作者采用了随机梯度下降（SGD）方法，利用小批量的数据来近似似然项，这样变分目标就变成：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek00Q2J神经网络0ZFQWFjTElpYUFiaWE3bVUwNG5FVUZlRldyVURvWXAzb0NIaWNhdFNLdzRpYzI0ZE9FY2hBLzY0MA?x-oss-process=image/format,png)

其中 ，每个子集的大小为 。这为在训练期间使用大型数据集提供了一种有效的方法。在通过一个子集 后，应用反向传播来更新模型参数。SGD 是使用变分推断方法训练神经网络和贝叶斯神经网络的最常用方法。

Graves 在 2011 年发表了贝叶斯神经网络研究的一篇重要论文《Practical variational inference for neural networks》。这项工作提出了一个 MFVB 处理使用因子高斯近似后验分布。这项工作的主要贡献是导数的计算。变分推断的目标可以看作是两个期望值的总和如下所示：

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

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1pYUxpYkRqc1ViYnZaU2ljWTBlVG9nd1BYaWNDZldBMHNpYUpKd2ljYmpIUV神经网络c3hpYVlXV25HNjRpYm9SUS82NDA?x-oss-process=image/format,png)

由于上式是可微的，所以可以使用梯度下降法来优化这种期望近似。这是变分推断中的一个重要属性，因为变分推断的目标的对数似然的期望值的求解困难很大。与分数函数估计量相比，路径估计值法更有利于降低方差。

Blundell 等人在论文《Bayes by Backprop》中提出了一种在贝叶斯神经网络中进行近似推断的方法。该方法利用重参数化技巧来说明如何找到期望导数的无偏估计。其期望导数的具体形式如下所示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1VSTRZajhCSzFnRnFYUWljaWJuanhFQmlheENqZ神经网络KWjNZcHFHbWd5amlhTUU1cXBROWVFd3UwV29RLzY0MA?x-oss-process=image/format,png)

在贝叶斯的反向传播的算法中，函数 设为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek0yaFdJZ3RINjFzU2tHNWM2TjMwWTRpY3pRWG9UTkJNZ2lhaGxoR0xFZkZ6SFRyaWF2Um0yZllpYmZBLzY0MA?x-oss-process=image/format,png)

其中 可以看作是期望值的自变量，它是下界的一部分。

假设全因子高斯后验函数 ，其中 用于确保标准差参数为正。由此，将网络中的权重 的分布重新参数化为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek11aFAyRTc3cklieXVLNFJKbTZOaWJjQ0RKeUNvN2tCd3NERmR2WWdiaWNTMkxtVDVYS3ZiaWMwM1EvNjQw?x-oss-process=image/format,png)

在该贝叶斯神经网络中，可训练参数为 和 。由于采用全因子分布，则近似后验概率的对数可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek0xMDZNRzVSV1M0eUd1a21sZFVLeXhRaGVqNW1MRnZpY2lhNVVVdWtqaWJaQ2hSTG9LMnpvMmVsU2cvNjQw?x-oss-process=image/format,png)

综合上面提到的贝叶斯的反向传播算法的细节，会有如下完整的算法流程图。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9DeFBpY2lhcE8yVVdiWDdXMDVmSGQ0UGlhWTBvaFFCTVNvaWJlMTRPektiQnBnaWFuQTdhUlFGMlJZUktNMjgyUWljdW5Sc3NXY3FkcWNwR3lscDBOQ2licDJUOEEvNjQw?x-oss-process=image/format,png)

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ESGVmbFdUZFRJQkVpYVBNQmFLTXFrUnJWOURjZGFGUDNWeXhTY0hDdWFCeVo1OHZ3OUljNHMzaWFjVWljOFhHbmtZODZsSDg2MmJaWmVpY2cvNjQw?x-oss-process=image/format,png)

## 2.5 BNN 的高斯过程特性

Neal[38] 还给出了推导和实验结果，以说明对于只有一个隐层的网络，当隐藏单元的数量接近无穷大时，会出现高于网络输出的高斯过程 (GP)，并且将高斯先验置于参数 22 之上。图 6 说明了该结果。

下图为当在参数上放置高斯先验时，随着网络规模的增加，先验在输出上诱导的图示。其中图中的每个点对应于一个网络的输出，参数从先验分布中进行采样。对于每个网络，隐藏单元的数量是图（a）对应着 1 个单元，图（b）对应着 3 个单元，图（c）对应着 10 个单元，图（d）对应着 100 个单元。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek12aWJpYmoxRGUzSmVvRVpOZGpyMWh3aWJYdUkycTRVamliRUxNUU5Qem9GMk9wblpKV1hzaWJqelpLZy82NDA?x-oss-process=image/format,png)

由此可知在中心极限定理下，随着隐层数 N 逐渐增大，输出服从高斯分布。由于输出被描述为基函数的无穷和，因此可以将输出看作高斯过程。高斯过程和具有单个隐藏层工作的无限宽网络之间的关系最近已扩展到深度神经网络的当中，该联系的识别激发了贝叶斯神经网络的许多研究工作。

高斯过程提供了可靠的不确定性估计、可解释性和鲁棒性。但是高斯过程提供这些好处的代价是预测性能和随着数据集大小的增加所需的大量计算资源。高斯过程和贝叶斯神经网络之间的这种联系促使了两种建模方案的合并；既能维持神经网络的预测性能和灵活性，同时结合了高斯过程的的鲁棒性和概率性。

最近的研究已经确定了高斯过程属性不限于 MLP-贝叶斯神经网络，而且在卷积中也可以应用该属性。因为 C神经网络 可以实现为 MLP，其结构在权重中被强制执行。Vander Wilk 等人在 2003 年发表的论文《Convolutional gaussian processes》中提出了卷积高斯过程，它实现了一种类似于 C神经网络 中的基于 patch 的操作来定义 GP 先验函数。

如下图所示，分析显示了高斯过程在预测偏差和方差方面的对比性能。用 Backprop 和一个因式高斯近似后验概率对 Bayes 模型进行训练，在训练数据分布的情况下，虽然训练数据区域外的方差与高斯过程相比显著低估，但预测结果是合理的。具有标度伯努利近似后验的 MC-Dropout 通常表现出更大的方差，尽管在训练数据的分布中保持不必要的高方差。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Xc神经网络rRXlJTmZIMUtYd1g0R0pMYlpPN0NTcHdneTJuUmZpY3ZkSTd2T2JUZnJjMzR5aWF2Njg2cnoxRENab29pYUpWSlZpYW1EYU9yaE1EeUtPZnQxQ2FoQmcvNjQw?x-oss-process=image/format,png)

## 2.6 当前 BNN 的局限性

虽然人们已经付出了很大的努力来开发在神经网络中执行推理的贝叶斯方法，但这些方法有很大的局限性，文献中还存在许多空白。一个关键的限制是严重依赖 VI 方法。在 VI 框架内，最常用的方法是平均场方法。MFVB 通过强制参数之间独立的强假设，提供了一种表示近似后验分布的方便方法。该假设允许使用因式分解分布来近似后验分布。这种独立假设大大降低了近似推理的计算复杂度，但代价是概率精度。

# 3 现代 BNN 的比较

## 3.1 普通 BNN

从文献综述来看，贝叶斯神经网络中两种重要的近似推理方法是 Backprop 的 Bayes[76] 和 MC Dropout[85]。这些方法被认为是贝叶斯神经网络中最有前途、影响最大的近似推理方法。这两种 VI 方法都足够灵活，可以使用 SGD，从而使部署到大型、实用的数据集成为可能。鉴于这些方法的突出之处，有必要对这些方法进行比较，看看它们的表现如何。

为了比较这些方法，进行了一系列简单的同方差回归任务。对于这些回归模型，概率用高斯表示。有了这个，我们就可以写出未归一化的后验是，

## 3.2 卷积 BNN

虽然 MLP 是神经网络的基础，但最突出的神经网络架构是卷积神经网络。这些网络在具有挑战性的图像分类任务方面表现出色，其预测性能远远超过先前基于核或特征工程的方法。C神经网络 不同于典型的 MLP，它的应用是一个卷积型的算子，单个卷积层的输出可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1GWEFQN2N3QTdjSlhrOE90VHpuVzlyOG9kdG9MUWJXWHNiS2FpYUdpYmozWktwc25TaWFsQnFzUWcvNjQw?x-oss-process=image/format,png)

其中 是非线性激活， 表示类似卷积的运算。这里输入 X 和权重矩阵 W 不再局限于向量或矩阵，而是可以是多维数组。可以得到证明的是 c神经网络 可以编写成具有等效 MLP 模型，允许使用优化的线性代数包进行反向传播训练。

在现有研究方法的基础上，发展了一种新型的贝叶斯卷积神经网络（BC神经网络）。假设卷积层中的每个权重是独立的，允许对每个单独的参数进行因子分解，其中 BC神经网络 输出概率向量由 Softmax 函数表示：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek1KRGJMVTVLckRJeWNKdjhVa01oU3ZpYlVlZ0g1NTdmZFFyMGliOFpaWlFRQ2lhYUQ5UUZBektzSWcvNjQw?x-oss-process=image/format,png)

非标准化的后验分布可以表示为：

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9Qc2hvOWRtN29ER3lZVTJ2UmFWSkZ1U1p5aWN1OHVPek15NDRONzI3cW1TZHJCbTdmbzVOaWFTREdHVWN3eG5BQUEyV3dTZmx5RVlKMEdJcWliRWljMVpod2cvNjQw?x-oss-process=image/format,png)

作者做了相关的对比实验，BC神经网络 使用了流行的 LeNet 架构，并利用 BC神经网络 的平均输出进行分类，并使用可信区间来评估模型的不确定性。

在 MNIST 数据集中的 10000 个测试图像上，两个网络的总体预测性能显示出了比较好的性能。BC神经网络 的测试预测精度为 98.99%，香草网络的预测精度略有提高，预测精度为 99.92%。

虽然竞争性的预测性能是必不可少的，但 BC神经网络 的主要优点是可以提供有关预测不确定性的有价值的信息。从这些例子中，可以看到这些具有挑战性的图像存在大量的预测不确定性，这些不确定性可以用于在实际场景中做出更明智的决策。

## 3.3 循环 BNN



# 4 结论

本报告阐明了典型的神经网络和特殊模型设计中的过度自信预测问题，而贝叶斯分析被证明可用来解决这些挑战。尽管对于贝叶斯神经网络来说，精确推断仍然是分析和计算上的难题，但实践表明，可以依靠近似推断方法获得较为精确的近似后验。

贝叶斯神经网络中的许多近似推理方法都围绕 MFVB 方法展开。这为优化变分参数提供了一个易于处理的下限。这些方法在易用性、预测均值的准确性、可接受的参数数目等方面具有吸引力。文献调查和实验结果表明，在完全分解的 MFVB 方法中所做的假设导致了过度自信的预测。结果表明，这些 MFVB 方法可以推广到更复杂的模型，如卷积神经网络。实验结果表明，对于图像分类任务，贝叶斯预测性能与基于点估计的卷积神经网络相当。贝叶斯卷积神经网络能够为预测提供可信的区间，而这些预测是对难以分类的数据点提供的高度信息性和直观性的不确定性度量。

这篇综述和实验突出了贝叶斯分析解决机器学习社区中常见挑战的能力。这些结果还突显了当前用于贝叶斯神经网络的近似推理方法的不足，甚至可能提供不准确的方差信息。不仅要确定网络是如何运行的，而且要确定现代大型网络如何才能实现精确推断，这尚有待进一步研究。将 MCMC 等推断方法扩展到大数据集上允许更有原则性的推理。MCMC 提供了评估收敛和推断质量的诊断方法。对变分推断的类似诊断允许研究人员和实践者评估他们假设后验的质量，并告知他们改进该假设的方法。实现这些目标将使我们能够获得更精确的后验近似。由此，我们将能够充分确定模型知道什么，也可以确定模型不知道什么。

# 参考文献



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
