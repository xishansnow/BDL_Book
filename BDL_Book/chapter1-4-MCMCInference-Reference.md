# 第 4 章  MCMC 方法

## 4.1 概述

马尔可夫链蒙特卡罗(MCMC)方法为现实统计建模提供了巨大的空间。直到最近，承认许多应用程序的全部复杂性和结构是困难的，需要开发特定的方法和专门构建的软件。另一种选择是将问题强迫到一个可用方法的过于简单的框架中。现在，MCMC方法提供了一个统一的框架，在这个框架内可以使用通用软件来分析许多复杂的问题。

MCMC本质上是使用马尔可夫链的蒙特卡罗积分。贝叶斯人，有时也是频率学家，需要在可能的高维概率分布上进行积分，以对模型参数进行推断或进行预测。在给定数据的情况下，贝叶斯需要在模型参数的后验分布上进行积分，而在给定参数值的情况下，频率学家可能需要在可观测值的分布上进行积分。如下所述，蒙特卡罗积分从所需分布中抽取样本，然后形成样本平均值以近似预期。马尔可夫链蒙特卡罗通过长时间运行一条构造巧妙的马尔可夫链来提取这些样本。构建这些链的方法有很多种，但所有的方法，包括Gibbs采样器(Geman and Geman，1984)，都是Metropolis et at一般框架的特例。(1953年)和黑斯廷斯(1970年)。

MCMC用了近40年的时间才渗透到主流统计实践中。它起源于统计物理学文献，在空间统计和图像分析中已经使用了十年。在过去的几年里，MCMC对贝叶斯统计产生了深远的影响，并在经典统计中得到了应用。最近的研究大大增加了它的应用范围，丰富了它的方法论，以及它的理论基础。

这本书的目的是介绍MCMC方法及其应用，并提供指向文献的指针以了解更多详细信息。考虑到主要是应用型读者，我们作为编辑的角色一直是将书中的技术内容保持在最低限度，并专注于已证明对实际应用有帮助的方法。但也提供了一些理论背景。这本书中的应用来自广泛的统计实践，但在某种程度上反映了我们自己的生物统计学偏见。这些章节是由研究人员撰写的，他们在MCMC方法论及其应用的最新发展中做出了关键贡献。遗憾的是，我们无法将所有领先的研究人员都包括在我们的贡献者名单中，也无法涵盖理论、方法和应用的所有领域，达到他们应得的深度。

我们的目标一直是保持每一章都是独立的，包括注释和参考文献，尽管各章可能假定了解本章描述的基础知识。本章包含足够的信息，使读者能够以基本的方式开始应用MCMC。在文中，我们描述了Metropolis-Hastings算法、Gibbs采样器，以及在实现MCMC方法时出现的主要问题。我们还简要介绍了贝叶斯推理，因为下面的许多章节都假设有一个基本的知识。第2章通过一个工作示例说明了许多主要问题。第三章和第四章介绍了离散状态空间马尔可夫链理论和一般状态空间马尔可夫链理论中的重要概念和结果。第5章到第8章提供了有关实现MCMC或提高其性能的技术的更多信息。第9章到第13章描述了使用MCMC评估模型充分性和在模型之间进行选择的方法。第14章和第15章描述用于非贝叶斯推理的MCMC方法，第16章到第25章描述应用或总结应用领域。


## 4.2 问题的提出

### 4.2.1 贝叶斯推断
从贝叶斯角度来看，统计模型的可观测数据和参数之间没有根本区别：都被认为是随机量。设 $D$ 表示观测数据，$\theta$ 表示模型参数或缺失（潜在）变量。然后，形式推断要求在所有随机变量上建立联合概率分布 $P(D，\theta)$ 。该联合分布包括两部分：先验分布 $P(\theta)$ 和似然 $P(D \mid \theta)$ 。通过指定 $P(\theta)$ 和 $P(D \mid \theta)$ ，可以构造全概率模型：

$$
 P(D,\theta) = P(D \mid \theta) P(\theta)
$$

在观察到 $D$ 的情况下，使用贝叶斯定理来确定 $D$ 条件下 $\theta$ 的分布：

$$
 P(\theta \mid D)=\frac{P(\theta) P(D \mid \theta)}{\int P(\theta) P(D \mid \theta) d \theta}
$$

上式被称为 $\theta$ 的后验分布，是所有贝叶斯推断的目标。

后验分布的任何特征都可以是贝叶斯推断的对象，如：后验分布的矩（Moments）、分位数（Quantiles）、最高后验密度区间（Highest Posterior Density Region,HPDR）等。所有这些量都是用 `以 $\theta$ 为自变量的某个函数的后验期望` 来表示。其通用表示为函数 $f(\theta)$ 的后验期望：

$$
\begin{aligned}
E[f(\theta) \mid D] &=\int f(\theta) P(\theta \mid D) d \theta \\
&=\frac{\int f(\theta) P(\theta) P(D \mid \theta) d \theta}{\int P(\theta) P(D \mid \theta) d \theta}
\end{aligned}
$$

直到最近，该表达式中的积分一直是贝叶斯推断的实际困难，特别是在高维情况下。在大多数应用中，$E[f(\theta) \mid D]$ 无法获得精确的解析解；此时获得数值解可能是一种方案，但在大于 20 维时计算数值解非常困难并且不精确；获得与真实分布近似的解析解也是一种方案，例如 Kass 等人 1988 年提出的拉普拉斯近似方法；此外还可以选择 `蒙特卡罗积分（MC 积分）` ，其中包括 `MCMC`。

### 4.2.2 计算期望值

不仅贝叶斯统计中存在计算期望的问题，在频率主义框架中也存在计算高维分布期望值的问题。为了在后面的讨论中避免不必要的贝叶斯色彩，我们用更一般的术语重申这个问题。设 $X$ 是 $k$ 个随机变量构成的向量，其分布为 $\pi(.)$ 。在贝叶斯主义应用中，$X$ 将包括模型参数和缺失（潜在）变量；在频率主义应用中，它可能包括数据或随机效应。对于贝叶斯统计， $\pi(.)$ 指后验分布，而对于频率主义框架来说， $\pi(.)$ 指似然。无论哪种方式，目标任务都是评估某个函数 $f(.)$ 的期望：

$$
E[f(X)]=\frac{\int f(x) \pi(x) d x}{\int \pi(x) d x} \tag{1}
$$

这里我们考虑到这样一种可能性，即只有在知道了归一化常数后，才能知道 $X$ 的绝对分布。也就是说，上式中 $\int \pi (x)dx$ 是未知的。这在实际中是常见情况，例如在贝叶斯推理中，我们知道 $P(\theta \mid D) \propto P(\theta) P(D \mid \theta)$ ，但是很难计算出归一化常数 $\int P(\theta) P(D \mid \theta) d \theta$ 。为简单起见，假设 $X$ 取值于 $k$ 维欧氏空间，即 $X$ 包含 $k$ 个连续的随机变量。当然，此处描述的方法是通用的，例如，$X$ 可以由离散随机变量组成，式 (1) 中的积分将被求和取代）。或者，$X$ 可以是离散和连续随机变量的混合，或者是任何概率空间上的随机变量的集合。因此，后面会交替使用 `分布`（面向离散随机变量） 和 `密度` （面向连续随机变量） 这两个术语。


## 4.3 马尔科夫链蒙特卡罗 MCMC

### 4.3.1 蒙特卡洛积分（ MC 积分）

MC 积分从 $\pi (.)$ 中抽取样本集 ${X_t，t=1，...，n}$ ，然后近似计算 $E[f(X)]$：

$$
E[f(X)] \approx \frac{1}{n} \sum_{t=1}^{n} f\left(X_{t}\right) 
$$

因此，$f(X)$ 的总体均值是通过样本均值来估计的。当样本 ${X_t}$ 独立时，大数定律可确保通过增加样本数量 $n$ 来获得所需的近似精度。此处 $n$ 可变并由分析师控制，虽然都被称为 `样本（sample）`，但一定要注意它和机器学习中样本数据集大小的区别。

通常，从 $\pi (.)$ 中独立地抽取样本 ${X_t}$ 是不可行的，因为 $\pi(.)$ 可能是非常不标准。但 ${X_t}$ 的独立性并非必需。 ${X_t}$ 可以由在整个 $\pi (.)$ 范围内的任意过程以正确的比例抽样得到。其中一种方法是将 $\pi (.)$ 作为马尔科夫链的平稳分布实现，这就是 `马尔科夫链蒙特卡罗（Markov chain Monte Carlo，MCMC）`。

### 4.3.2 马尔科夫链 （ Markov chains ）

假设我们生成一个随机变量序列 ${x_0，x_1，x_2，……}$，对于每个时间 $t \geq 0$ ，下一状态 $X_{t+1}$ 抽自分布 $P(X_{t+1}|X_t)$ 。 也就是说，给定 $X_t$ ，下一状态$X_{t+1}$ 仅依赖于链的当前状态 $X_t$ ，而与 $\{X_0，X_1，…，X_{t-1} \}$ 的历史链无关。这种序列称为马尔可夫链，P(.|.) 被称为链的转移核；通常假设链是时间齐次的，即 $P(.|.)$ 不依赖于时间 $t$ 。

链的初始状态 $X_0$ 对 $X_t$ 会有什么影响？这个问题涉及给定 $X_0$ 时 $X_t$的分布，我们记为 $P^{(t)}(X_t|X_0)$ 。这里没有给出中间变量 $\{X_1，X_2，…，X_{t-1} \}$，所以 $X_t$ 直接依赖于 $X_0$ 。在满足正则性条件的情况下，该链将逐渐“忘记”其初始状态，并且 $P^{(t)}(.|X_0)$ 最终将收敛到唯一的平稳（或不变）分布，而该分布不依赖于 $t$ 或 $x_0$ 。此时，我们设  表示平稳分布，则随着 $t$ 的增加，采样点 ${X_t}$ 看起来越来越像来自 $\phi (.)$ 的相关样本。这如图 1 所示，其中 $\phi (.)$ 为一元标准正态分布。请注意，图 1 中 (a) 比 (b) 或 (c) 收敛快得多。

因此，在经过足够长的老化（比方说 $m$ 次迭代）后，点集 ${X_t；t=m+1，...，n}$ 将是从分布 $\phi (.)$ 中获取的相关样本。现在我们可以使用马尔可夫链的输出来估计期望 $E[f(X)]$ ，其中 $X$ 的分布为  $\phi (.)$ 。在计算中，老化样本通常被丢弃，从而给出一个估计值：
$$
\bar{f}=\frac{1}{n-m} \sum_{t=m+1}^{n} f\left(X_{t}\right) \tag{2}
$$

这被称为遍历平均值，遍历定理确保其收敛到所需的期望。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210605163313_5f.webp)

图 1  $N(0，1)$ 平稳分布和提案分布的 `Metropolis 算法`，500 次迭代。(a) $q(.|X)=N(X，0.5)$ ；(b)$q(.|X)=N(X，O.1)$ ；以及 (c) $q(.|X)=N(X,10.0)$ ，垂直虚线的左侧为老化区。


### 4.3.3 Metropolis-Hastings 算法

方程 (2) 显示了如何使用马尔可夫链来估计 $E[f(X)]$ ，其中期望取自其平稳分布 $\phi (.)$ 。这似乎为我们的问题提供了解决方案，但首先需要了解如何构造马尔可夫链，使其平稳分布 $\phi (.)$ 正好是我们感兴趣的分布 $\pi (.)$ 。

构建这样的马尔可夫链出奇地容易。我们描述了 Hastings(1970) 提出的形式，它是 Metropolis 等人方法的推广 (1953 年）。对于 Metropolis-Hastings 算法，在每个时间 $t$，首先从提案分布 $q(.|X_t)$ 中采样候选点 $Y$, 并依据 $Y$ 来选择下一状态 $X_{t+1}$ 。注意，提案分布可以取决于当前点 $X_t$ 。例如，$q(.|X)$ 可以是具有均值 $X$ 和固定协方差矩阵的多元正态分布。然后以概率 $\alpha(X_t，Y)$ 接受候选点 $Y$，其中

$$
\alpha(X, Y)=\min \left(1, \frac{\pi(Y) q(X \mid Y)}{\pi(X) q(Y \mid X)}\right) \tag{3}
$$

如果候选点被接受，则下一状态变为 $X_{t+1}=Y$ 。如果候选点被拒绝，链不移动，即 $X_{t+1}=X_t$ 。图 1 对单变量正态分布和目标分布进行了说明；图 1(c) 显示了链在几次迭代中没有移动的许多实例。

因此，Metropolis-Hastings 算法非常简单：

```{code-block}
Initialize $X_0$; set $t = O$. 
Repeat { 
    Sample a point $Y$ from $q(.|X_t)$ 
    Sample a Uniform(O,1) random variable $U$ 
    If $U \leq \alpha (X_t, Y)$ set $X_{t+1} = Y$ 
        otherwise set $X_{t+1} = X_t$ 
    Increment $t$
}. 
```

值得注意的是，提案 $q(.|.)$ 可以有任何形式，链的稳定分布将是 $\pi(.)$。这可以从以下论点中看出。Metropolis-Hastings 算法的转移核是：

\begin{align*} \tag{4}
P\left(X_{t+1} \mid X_{t}\right)=& q\left(X_{t+1} \mid X_{t}\right) \alpha\left(X_{t}, X_{t+1}\right) \\ 
&+I\left(X_{t+1}=X_{t}\right)\left[1-\int q\left(Y \mid X_{t}\right) \alpha\left(X_{t}, Y\right) d Y\right] 
\end{align*} 


其中 $I(.)$ 表示指示器函数 （当其参数为 true 时取值 1，否则取值 0）。式 (4) 中的第一项源于接受候选 $Y=X_{t+1}$ ，第二项源于对所有可能的候选 $Y$ 的拒绝，使用以下事实：

$$
\pi\left(X_{t}\right) q\left(X_{t+1} \mid X_{t}\right) \alpha\left(X_{t}, X_{t+1}\right)=\pi\left(X_{t+1}\right) q\left(X_{t} \mid X_{t+1}\right) \alpha\left(X_{t+1}, X_{t}\right) 
$$

从 (3) 开始，我们得到了详细的平衡方程：

$$
\pi\left(X_{t}\right) P\left(X_{t+1} \mid X_{t}\right)=\pi\left(X_{t+1}\right) P\left(X_{t} \mid X_{t+1}\right) \tag{5}
$$

关于 $X_t$ 对 (5) 的两个边进行积分可得到：

$$
\int \pi\left(X_{t}\right) P\left(X_{t+1} \mid X_{t}\right) d X_{t}=\pi\left(X_{t+1}\right) \tag{6}
$$

方程 (6) 的左边给出了假设 $X_t$ 来自 $\pi(.)$ 的 $X_{t+1}$ 的边际分布，因此 (6) 表示如果 $X_t$ 来自 $\pi(.)$ ，那么 $X_{t+1}$ 也将是。因此，一旦从平稳分布获得样本，所有后续样本都将来自该分布。这只证明了平稳分布是 $\pi(.)$ ，并不是 Metropolis-Hastings 算法的完全证明。完全证明需要证明 $P^{(t)}(X_t | X_0)$ 将收敛到平稳分布。进一步的细节见 Roberts(1995) 和 Tierney(1995)。

到目前为止，我们假设 $X$ 是 $k$ 个连续随机变量的定长向量。如第 2 节所述，还有许多其他可能性，特别是 $X$ 可以是可变维度的。例如，在贝叶斯混合模型中，混合成分的数量可能是可变的：每个成分都有其自己的尺度和位置参数。在这种情况下，$\pi(.)$ 必须指定 $k$ 和 $X$ 的联合分布，并且 $q(Y|X)$  必须能够提案在不同维度的空间之间移动。然后 `Metropolis-Hastings` 如上所述，具有形式上相同的接受概率表达式 (3)，但是必须仔细考虑不同维度空间之间移动的维度匹配条件 `(Green，1994a，b)`。另请参见 `Geyer等(1993)、Griander等(1994) 和 Phillips等 (1995)` 。 


## 4.4 算法实现

在实施 MCMC 时会出现几个问题。我们在这里简要讨论这些问题。最直接的问题是提案分布  $q(.|.)$  的选择。

### 4.4.1 提案分布的规范形式

如前所述，任何提案分布最终都将提供来自目标分布 $\pi(.)$ 的样本。然而，收敛到平稳分布的速度将关键取决于  $q(.|.)$  和 $\pi (.)$  之间的关系。此外，在“收敛”之后，链可能仍会缓慢混合（即在 $\pi(.)$ 的支撑处缓慢移动。这些现象如图 1 所示。图 1(a) 显示了从一个有点极端的起始值开始的快速收敛：此后链快速混合。图 1(b)，(c) 显示缓慢混合链：这些混合链必须运行更长时间才能从 (2) 获得可靠的估计，尽管已经以 $\pi (.)$  模式启动。

在几乎没有对称性的高维问题中，经常需要进行探索性分析来粗略地确定  $\pi (.)$  的形状和方向。这将有助于构造一个方案  $q(.|.)$ ，这将导致快速混合。在实践中的进展通常依赖于实验和工艺，尽管  $q(.|.)$  的未调谐规范形式通常工作得出奇地好。为了提高计算效率，应选择  $q(.|.)$  以便于抽样和评估。

这里我们描述了  $q(.|.)$  的一些经典形式。`Roberts(1995)`，`Tierney(1995)` 和 `Gilks and Roberts(1995)` 讨论了收敛速度和选择  $q(.|.)$  的一些策略。

#### （1） Metropolis 算法

`Metropolis 算法 (Metropolis et al.，1953)` 只考虑对所有 $X$ 和 $Y$ 具有形式  $q(Y|X)=q(X|Y)$  的对称提案。例如，当 $X$ 是连续的时， $q(.|X)$  可能是具有均值 $X$ 和恒定协方差矩阵 $\sum$ 的多元正态分布。通常，选择有条件地独立地产生 $Y$ 的每个分量的提案（给定 $X_t$ ) 是方便的。对于 `Metropolis 算法` ，接受概率 (3) 降低到

$$
\alpha(X, Y)=\min \left(1, \frac{\pi(Y)}{\pi(X)}\right) \tag{7}
$$

`Metropolis 算法` 的一个特例是 `随机游走 Metropolis`，其中 $q(Y|X)=q(|X-Y|)$。图 1 中的数据是由 `随机游走 Metropolis 算法` 生成的。

在选择提案分布时，可能需要仔细选择其规模（例如 $\sum$）。生成小步骤 $Y-X_t$ 的谨慎的提案分配通常会有很高的接受率 (7)，但仍会慢慢混合。这一点如图 1(b) 所示。生成大步骤的大胆提案分布通常会提案从中间移动到分布的尾部，给出较小的值 $\pi (Y) / \pi (X_t)$ 和较低的接受概率。这样的链通常不会移动，同样会导致缓慢的混合，如图 1(c) 所示。理想情况下，提案分布应该按比例进行调整，以避免这两个极端。

#### （2） 独立采样器

`独立采样器 (Tierney，1994)` 是一种 `Metropolis-Hastings 算法`，其提案 $q(Y|X)=q(Y)$ 不依赖于 $X$ 。为此，接受概率 (3) 可以写成以下形式

$$
\alpha(X, Y)=\min \left(1, \frac{w(Y)}{w(X)}\right) \tag{8}
$$

式中 $w(X)=\pi (X) / q(X)$ 。

一般来说，独立采样器可以工作得很好，也可以很差（参见 Roberts，1995）。要使独立采样器正常工作， $q(.)$ 。应该是 $\pi (.)$ 的一个很好的近似，但如果  $q(.)$ 尾部比 $\pi (.)$ 重时最为安全。要了解这一点，假设  $q(.)$ 。尾部比 $\pi (.)$ 轻，且 $X_t$ 当前位于 $\pi (.)$ 的尾部。大多数候选者不会出现在尾部，因此 $w( X_t )$ 将比 $w(Y)$ 大得多，从而提供较低的接受概率 (8)。因此，拖后腿的独立提案有助于避免长时间滞留在尾部上，代价是增加了候选者的总体拒绝率。

在某些情况下，特别是在认为大样本理论可能正在运行的情况下，可以尝试多变量正态分布方案，平均值为  $\pi (.)$ 。在模式下评估，`协方差矩阵` 比 `逆 Hessian 矩阵` 稍大：

$$
\left[-\frac{d^{2} \log \pi(x)}{d x^{\mathrm{T}} d x}\right]^{-1} 
$$


#### （3） 单组份 `Metropolis-Hastings`

将 $X$ 分成可能不同维度的组份 $\{X_{.1},X_{.2}，···，X_{.h}\}$，然后逐个更新这些组份而不是整体更新整个 $X$ ， 通常更方便且计算效率更高。这是 `Metropolis 等人` 最初提出的 MCMC 框架 `(1953)`，我们称其为`单组份 Metropolis-Hastings`。设 $X_{.-i}=\{X_{.1},...,X_{.i-1}，X_{.i+1},...,X_{.h}\}$，即 $X_{.-i}$ 包含除 $X_{.i}$ 之外的所有 $X$ 。

`单组份 Metropolis-Hastings 算法`的迭代包括 $h$ 个更新步骤，如下所示。设 $X_{t.i}$ 表示迭代 $t$ 结束时 $X_{.i}$ 的状态。对于迭代 $t+1$ 的步骤 $i$，使用 `Metropolis-Uastings` 更新 $X_{.i}$ 。候选 $Y_{.i}$ 由提案分布 $q_i(Y_i | X_{t.i},X_{t. -i})$ 生成，其中 $X_{t. -i}$ 表示完成第 $t+1$ 次迭代的步骤 $i-1$ 后 $X_{. -1}$ 的值：

$$
X_{t .-i}=\left\{X_{t+1.1}, \ldots, X_{t+1 . i-1}, X_{t . i+1}, \ldots, X_{t . h}\right\} 
$$

其中组份 $1、2、...、i-1$ 已经更新。这样，第 $i$ 个提案分布 $q_i(.|.,.)$ 仅生成 $X$ 的第 $i$ 个分量的候选，并且取决于 $X$ 的任何分量的当前值。该候选以概率 $\alpha(X_{t. -i},X_{t.i},Y_{.i})$ 被接受，其中

$$
\alpha\left(X_{-i}, X_{. i}, Y_{. i}\right)=\min \left(1, \frac{\pi\left(Y_{. i} \mid X_{.-i}\right) q_{i}\left(X_{. i} \mid Y_{. i}, X_{.-i}\right)}{\pi\left(X_{. i} \mid X_{.-i}\right) q_{i}\left(Y_{. i} \mid X_{. i}, X_{.-i}\right)}\right) \tag{9}
$$

这里 $\pi (X_{.i}|X_{.-i})$ 是 $X$ ，$i$ 在 $\pi (.)$ 下的全条件分布。如果 $Y_{.i}$ 被接受，我们设置 $X_{t+1.i}=Y_{.i}$ ；否则，我们设置 $X_{t+1.i=X_{t.i}$ 。其余组份在步骤 i 中不会更改。

因此，每个更新步骤都会在坐标轴方向上产生一个移动（如果候选对象被接受），如图 2 所示。提案分布区 $q_i(.|.,.)$ 可以通过本节前面讨论的任何方式进行选择。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210607073449_4f.webp)

图 2 说明了双变量目标分布 $\pi (.)$ 的 `单组份 Metropolis-Hastings 算法` 。组份 1 和组份 2 交替更新，在水平和垂直方向上交替移动

全条件分布 $\pi (X_{.i}|X_{.-i})$ 是 $X$ 的第 i 个分量在所有剩余分量上的条件分布，其中 $X$ 具有分布 $\pi (.)$ ：

$$
\pi\left(X_{. i} \mid X_{.-i}\right)=\frac{\pi(X)}{\int \pi(X) d X_{. i}} \tag{10}
$$

全条件分布在许多应用中扮演着重要角色，`Gilks (1995）` 对此进行了详细研究。具有由式 (9) 给出的接受概率的 `单组份 Metropolis-Hastings 算法` 确实从目标分布  $\pi (.)$ 中抽取，这是由于  $\pi (.)$ 是由其完全条件分布的集合唯一确定的 `(Besag，1974)` 。

在应用中，式(9) 通常会大大简化，特别是在  $\pi (.)$ 源自条件独立模型：参见 `Spiegelhalter 等人。(1995)` 和`Gilks (1995)`。这提供了一个重要的计算优势。当目标分布  $\pi (.)$ 。就像空间模型中常见的那样，自然地用它的完全条件分布来指定；参见 `Besag(1974)`，`Besag 等(1995)` 和 `Green (1995）` 。

#### （4） 吉布斯抽样

单组份 Metropolis-Hastings 的一个特例是 Gibbs 采样器。Gibbs 采样器是由 Geman 和 Geman(1984) 命名的，他们用它来分析晶格上的 Gibbs 分布。然而，它的适用性并不局限于 Gibbs 分布，所以“Gibbs 抽样”实际上是一个误区。此外，同样的方法已经在统计物理学中使用，在那里被称为热浴算法。然而，Geman 和 Geman(1984) 的工作通过 Gelfand and Smith(1990) 和 Gelfand 等人的文章将 MCMC 引入主流统计学。(1990 年）。到目前为止，MCMC 的大多数统计应用都采用了 Gibbs 抽样。

对于 Gibbs 采样器，更新 $X$ 的第 i 个分量的提案分布为

$$
q_{i}\left(Y_{. i} \mid X_{. i}, X_{.-i}\right)=\pi\left(Y_{. i} \mid X_{.-i}\right) \tag{11}
$$

其中 1i\“(iix.-i) 是完全条件分布 (1.10)。将 (1.11) 代入 (1.9) 得到接受概率为 1；也就是说，Gibbs 采样器候选者总是被接受的。因此，Gibbs 抽样纯粹是从完全条件分布中抽样。从完全条件分布中抽样的方法在 Gilks(1995：本卷）中有描述。

### 4.4.2 阻塞

我们在第 1.4.1 节中对单组份取样器的描述没有说明应该如何选择组份。通常，使用低维或标量分量。在某些情况下，多变量成分是很自然的。例如，在贝叶斯随机效应模型中，整个精度矩阵通常由单个分量组成。当分量在平稳分布  $\pi (.)$  中高度相关时，混合可能很慢；参见 Gilks and Roberts(1995：这卷）。将高度相关的分量分块到高维分量可能会改善混合，但这取决于方案的选择。

### 4.4.3 更新顺序

在上面对单组份 Metropolis-Hastings 算法和 Gibbs 抽样的描述中，我们假定 XT 的分量的更新顺序是固定的。虽然这很常见，但不需要固定的顺序：更新顺序的随机排列是可以接受的。此外，并非所有组份都需要在每次迭代中更新。例如，我们可以在每次迭代中只更新一个组份，选择具有某个固定概率 s(I) 的组份 i。一个自然的选择是设置 s(I)=t。Zeger 和 Karim(1991) 提案比其他组份更频繁地更新高度相关的组份，以改善混合。注意，如果 s(I) 被允许依赖于 $X_t$ ，则接受概率 (1.9) 应该被修改，否则链的静止分布可能不再是目标分布 71‘(.)。具体地说，接受概率变为

$$
\min \left(1, \frac{\pi\left(Y_{. i} \mid X_{.-i}\right) s\left(i \mid Y_{. i}, X_{.-i}\right) q_{i}\left(X_{. i} \mid Y_{. i}, X_{.-i}\right)}{\pi\left(X_{. i} \mid X_{.-i}\right) s\left(i \mid X_{. i}, X_{.-i}\right) q_{i}\left(Y_{. i} \mid X_{. i}, X_{.-i}\right)}\right) 
$$

### 4.4.4 链的数量

到目前为止，我们只考虑运行一条链，但允许运行多条链。文献中的提案是相互矛盾的，从许多短链 (Gelfand and Smith，1990)，到几条长链 (Gelman and Rubin，1992a，b)，再到一条很长的链 (Geyer，1992)。现在人们普遍认为，出于从 71‘(.) 获得独立样品的愿望而经营许多短链是错误的，除非有一些特殊的原因需要独立的样品。当然，在 (1.2) 中遍历平均不需要独立样本。这所办了几年的学校和一所办得很久的学校之间的争论似乎还会继续下去。后者认为，一次非常长的运行最有可能找到新的模式，链之间的比较永远不能证明收敛，而前者认为，如果链还没有接近平稳性，比较几个看似收敛的链可能会揭示真正的差异；参见 Gelman(1995：本卷）。如果有多个处理器可用，在每个处理器上运行一条链通常是值得的。

### 4.4.5 初始值

关于这个话题的文章还不多。如果链是不可约的，则初始值 XO 的选择不会影响平稳分布。快速混合链，如图 1.1(A) 所示，将从极端起始值快速找到方向。对于慢混链，可能需要更仔细地选择起始值，以避免长时间的老化。然而，很少有必要花费太多精力来选择起始值。Gelman 和 Rubin(1992a，b) 提案在多个链中使用“过度分散”的起始值，以帮助评估收敛；见下文和 Gelman(1995：本卷）。

### 4.4.6 确定老化程度

老化 m 的长度取决于 xO、p(T)(xt！xO) 到 1i\“( $X_t$ ) 的收敛速度以及 p(T)(.I) 和 11\”(.) 的相似程度。是必须的。从理论上讲，在明确了“足够相似”的标准之后，就可以解析地确定 m。然而，在大多数情况下，这种计算在计算上是远远不可行的（参见 Roberts，1995：本书）。目视检查蒙特卡罗输出{xt，t=1，...，n}的曲线图是确定老化的最明显和最常用的方法，如图 1.1 所示。启动接近 11\“(.) 模式的链条并不能消除老化的需要，因为链条仍应运行足够长的时间，以使其”忘记“其起始位置。例如，在图 1.1(B) 中，链条在 500 次迭代中没有偏离其起始位置太远。在这种情况下，m 应设置为大于 500。

已经提出了更正式的确定 m 的工具，称为收敛诊断法。收敛诊断使用了各种理论方法和近似，但都在某种程度上利用了蒙特卡罗的输出。到目前为止，至少已经提出了 10 种收敛诊断方法；有关最近的综述，请参阅 Cowles 和 Carlin(1994)。这些诊断中的一些也适用于确定运行长度 n（见下文）。

收敛诊断可以根据它们是否基于蒙特卡罗输出的任意函数 f(X)；它们是使用来自单链还是来自多个链的输出；以及它们是否可以纯粹基于蒙特卡罗输出来进行分类。

依赖于监测{f( $X_t$ )，t=1，...，n}（例如 Gelman 和 Rubin，1992b；Raftery 和 Lewis，1992；Geweke，1992) 的方法很容易应用，但可能是误导的，因为通过迭代 m，f( $X_t$ ) 可能看起来在分布上收敛，而另一个未监测的函数 g( $X_t$ ) 可能没有。无论函数 f(.)。如果受到监控，可能会有其他人的行为有所不同。

从理论上讲，最好是全局比较完整的联合分布 p(T)(.)。为了避免直接处理 p(T)(.)，有几种方法是通过运行多个并行链来获取样本的 (Ritter and Tanner，1992；Roberts，1992；Liu and Liu，1993)，并利用转移核 P(1.). 然而，为了程序的稳定性，可能需要运行多个并行链，当收敛速度较慢时，这是一个严重的实用限制。
运行并行链明显增加了计算负担，但对于诊断收敛缓慢可能是有用的，即使是在非正式的情况下也是如此。例如，几个平行链可能单独看起来已经收敛，但它们之间的比较可能会显示出表观平稳分布的显著差异 (Gelman 和 Rubin，1992a)。

从实用的角度来看，纯粹基于蒙特卡罗输出的方法特别方便，允许在不求助于转换核 P(I.) 的情况下评估收敛性，因此不需要特定于模型的编码。

本卷不包含对融合诊断的回顾。这仍然是一个活跃的研究领域，关于现有方法在实际应用中的行为，特别是在高维和收敛缓慢的情况下，仍有许多需要了解。相反，本书中 Raftery 和 Lewis(1995) 和 Gelman(1995) 的章节包含了对两种最流行的方法的描述。这两种方法都监控任意函数 f(.)，并且完全基于蒙特卡罗输出。前者使用单链，后者使用多链。

Geyer(1992) 提案不必计算老化长度，因为它很可能小于 (1.2) 中（见下文）中 [估计器] 足够长的游程总长度的 1%。如果避免了极端的起始值，Geyer 提案将 m 设置为游程长度 n 的 1%到 2%之间。

### 4.4.7 判定终止时间

决定何时停止这条链条是一件重要的实际问题。其目的是使链运行足够长，以便在 (1.2) 中的估计器中获得足够的精度。] 的方差（称为蒙特卡罗方差）的估计由于在迭代{xd 中缺乏独立性而变得复杂。

确定游程长度 n 的最明显的非正式方法是以不同的起始值并行运行多个链，并从 (1.2) 开始比较估计值。如果它们不完全一致，则必须增加 n。已经提出了更正式的方法来估计] 的方差：有关更多细节，请参阅本卷中的 Roberts(1995) 和 Raftery and Lewis(1995)。

### 4.4.8 输出分析

在贝叶斯推断中，通常用感兴趣分量 X.i 的均值、标准差、相关性、可信区间和边际分布来概括后验分布 11\“(.)。均值、标准差和相关性都可以通过蒙特卡罗输出{Xu，t=m+1，...，n}中的样本等价物来估计，如 (1.2)。例如，X，i 的边际均值和方差由下式估计

$$
\bar{X}_{. i}=\frac{1}{n-m} \sum_{t=m+1}^{n} X_{t . i} 
$$

和

$$
S_{. i}^{2}=\frac{1}{n-m-1} \sum_{t=m+1}^{n}\left(X_{t . i}-\bar{X}_{. i}\right)^{2} 
$$

请注意，这些估计只是忽略了蒙特卡罗输出中的其他部分。

标量分量 X.i 的 100(1-2p)%可信区间 [Cp，Cl-p] 可以通过将 Cp 设置为等于{Xu，t=m+1，…，n}的第 p 个分位数，并且 Cl-p 等于第 (1-p) 个分位数来估计。Besag et al.。(1995) 给出了一个计算二维或多维矩形可信区域的程序

边缘分布可以通过核密度估计来估计。对于 X，i 的边际分布，这是

$$
\pi\left(X_{. i}\right) \approx \frac{1}{n-m} \sum_{t=m+1}^{n} K\left(X_{. i} \mid X_{t}\right) 
$$

其中 K(.I $X_t$ ) 是集中在 Xu 周围的密度。K(x‘i ixt) 的自然选择是完全条件分布 1i’(x‘i ixt.-i)。Gelfand 和 Smith(1990) 使用这种结构来估计 $\pi (.)$ 以下的预期。因此，E[！(Xi)) 的 Rao-Blackwell 估计是

$$
\bar{f}_{R B}=\frac{1}{n-m} \sum_{t=m+1}^{n} E\left[f\left(X_{. i}\right) \mid X_{t .-i}\right] \tag{12}
$$

其中期望值是关于完全条件 aI1l‘(x.ixt._i) 的。对于相当长的运行，使用 (1.12) 而不是 (1.2) 的改进通常很小，而且在任何情况下 (1.12) 都需要完全条件期望的封闭形式。

## 4.5 讨论

本章简要介绍 MCMC。我们希望我们已经说服读者，MCMC 是一个简单的想法，具有巨大的潜力。接下来的章节补充了这里概述的许多想法，特别是指出了这些方法在哪些地方工作得很好，在哪些地方需要进行一些调整或进一步的开发。

MCMC 方法和贝叶斯估计很自然地结合在一起，正如本卷中的许多章节所证明的那样。然而，贝叶斯模型的验证仍然是一个困难的领域。第 9-13 章描述了使用 MCMC 进行贝叶斯模型验证的一些技术

贝叶斯人和非贝叶斯人之间的哲学争论已经持续了几十年，从实践的角度来看，很大程度上是没有结果的。对于许多应用统计学家来说，最有说服力的论点是可靠的方法和软件的可用性。多年来，贝叶斯人很难解决非贝叶斯人直截了当的问题，所以今天大多数应用统计学家都是非贝叶斯人也就不足为奇了。随着 MCMC 和相关软件的到来，特别是 Gibbs 抽样程序错误（参见 Spiegelhalter 等人，1995：本卷），我们希望更多的应用统计学家将熟悉和适应贝叶斯思想，并应用它们。

## 参考文献

Besag, J. (1974) Spatial interaction and the statistical analysis oflattice systems 
(with discussion). J. R. Statist. Soc. B, 36, 192-236. 

Besag, J., Green, P., Higdon, D. and Mengersen, K. (1995) Bayesian computation 
and stochastic ·systems. Statist. Sci. (in press). 

Cowles, M. K. and Carlin, B. P. (1994) Markov chain Monte Carlo convergence 
diagnostics: a comparative ·review. Technical Report 94-008, Division of Bio-
statistics, School of Public Health, University of Minnesota. 

Diebolt, J. and Ip, E. H. S. (1995) Stochastic EM: methods and application. In 
Markov Chain Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and 
D. J. Spiegelhalter), pp. 259-273. London: Chapman & Hall. 

Gelfand, A. E. and Smith, A. F. M. (1990) Sampling-based approaches to calcu-
lating marginal densities. J. Am. Statist. Ass., 85, 398-409. 

Gelfand, A. E., Hills, S. E., Racine-Poon, A. and Smith, A. F. M. (1990)' illus-
tration of Bayesian inference in normal data models using Gibbs sampling. J. 
Am. Statist. Ass., 85, 972-985. 

Gelman, A. (1995) Inference and monitoring convergence. In Markov Chain Monte 
Carlo in Practice (eds W. R. Gilks, S. Richardson and D. J. Spiegelhalter), 
pp. 131-143. London: Chapman & Hall. 

Gelman, A. and Rubin, D. B. (1992a) A single series from the Gibbs sampler 
provides a false sense of security. In Bayesian Statistics 4 (eds J. M. Bernardo, 
J. Berger, A. P. Dawid and A. F. M. Smith), pp. 625-631. Oxford: Oxford 
University Press. 

Gelman, A. and Rubin, D. B. (1992b) Inference from iterative simulation using 
multiple sequences. Statist. Sci., 7, 457-472. 

Geman, S. and Geman, D. (1984) Stochastic relaxation, Gibbs distributions and 
the Bayesian restoration of images. IEEE Trans. Pattn. Anal. Mach. Intel., 6, 
721-741. 

Geweke, J. (1992) Evaluating the accuracy of sampling-based approaches to the 
calculation of posterior moments. In Bayesian Statistics 4 (eds J. M. Bernardo, 
J. Berger, A. P. Dawid and A. F. M. Smith), pp. 169-193. Oxford: Oxford 
University Press. 

Geyer, C. J. (1992) Practical Markov chain Monte Carlo. Statist. Sci., 7, 473-511. 

Geyer, C. J. (1995) Estimation and optimization offunctions. In Markov Chain 
Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and D. J. Spiegel-
halter), pp. 241-258. London: Chapman & Hall. 

Geyer, C. J. and MI/JIler, J. (1993) Simulation procedures and likelihood inference 
for spatial point processes. Technical Report, University of Aarhus. 

Gilks, W. R. (1995) Full conditional distributions. In Markov Chain Monte Carlo 
in Practice (eds W. R. Gilks, S. Richardson and D. J. Spiegelhalter), pp. 75-88. 
London: Chapman & Hall. 

Gilks, W. R. and Roberts, G. O. (1995) Strategies for improving MCMC. In 
Markov Chain Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and 
D. J. Spiegelhalter), pp. 89-114. London: Chapman & Hall. 

Green, P. J. (1994a) Discussion on Representations of knowledge in complex 
systems (by U. Grenander and M. I. Miller). J. R. Stattst. Soc. B, 56, 589-
590. 

Green, P. J. (1994b) Reversible jump MCMC computation and Bayesian model 
determination. Technical Report, Department of Mathematics, University of 
Bristol. 

Green, P. J. (1995) MCMC in image analysis. In Markov Chain Monte Carlo in 
Practice (eds W. R. Gilks, S. Richardson and D. J. Spiegelhalter), pp. 381-399. 
London: Chapman & Hall. 

Grenander, U. and Miller, M. I. (1994) Representations of knowledge in complex 
systems. J. R. Statist. Soc. B, 56, 549-603. 

Hastings, W. K. (1970) Monte Carlo sampling methods using Markov chains and 
their applications. Biometrika, 57, 97-109. 

Kass, R. E., Tierney, 1. and Kadane, J. B. (1988) Asymptotics in Bayesian com-
putation (with discussion). In Bayesian Statistics 9 (eds J. M. Bernardo, M. 
H. DeGroot, D. V. Lindley and A. F. M. Smith), pp. 261-278. Oxford: Oxford 
University Press. 

Liu, C. and Liu, J. (1993) Discussion on the meeting on the Gibbs sampler and 
other Markov chain Monte Carlo methods. J. R. Statist. Soc. B, 55, 82-83. 
Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, 
E. (1953) Equations of state calculations by fast computing machine. J. Chern. 
Phys., 21, 1087-1091. 

Phillips, D. B. and Smith, A. F. M. (1995) Bayesian model comparison via jump 
diffusions. In Markov Chain Monte Carlo in Practice (eds W. R. Gilks, S. 
Richardson and D. J. Spiegelhalter), pp. 215-239. London: Chapman & Hall. 

Raftery, A. E. and Lewis, S. M. (1992) How many iterations ofthe Gibbs sampler? 
In Bayesian Statistics 4- (eds J. M. Bernardo, J. Berger, A. P. Dawid and A. 
F. M. Smith), pp. 641-649. Oxford: Oxford University Press. 

Raftery, A. E. and Lewis, S. M. (1995) Implementing MCMC. In Markov Chain 
Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and D. J. Spiegel-
halter), pp. 115-130. London: Chapman & Hall. 

Ritter, C. and Tanner, M. A. (1992) Facilitating the Gibbs sampler: the Gibbs 
stopper and the Griddy-Gibbs sampler. J. Am. Statist. Ass., 87, 861-868. 

Roberts, G. O. (1992) Convergence diagnostics ofthe Gibbs sampler. In Bayesian 
Statistics 4- (eds J. M. Bernardo, J. Berger, A. P. Dawid and A. F. M. Smith), 
pp. 775-782. Oxford: Oxford University Press. 

Roberts, G. O. (1995) Markov chain concepts related to samping algorithms. In 
Markov Chain Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and 
D. J. Spiegelhalter), pp. 45-57. London: Chapman & Hall. 

Spiegelhalter, D. J., Best, N. G., Gilks, W. R. and Inskip, H. (1995) Hepatitis B: 
a case study in MCMC methods. In Markov Chain Monte Carlo in Practice 
(eds W. R. Gilks, S. Richardson and D. J. Spiegelhalter), pp. 21-43. London: 
Chapman & Hall. 

Tierney, L. (1994) Markov chains for exploring posterior distributions (with dis-
cussion). Ann. Statist., 22,1701-1762. 

Tierney, L. (1995) Introduction to general state-space Markov chain theory. In 
Markov Chain Monte Carlo in Practice (eds W. R. Gilks, S. Richardson and 
D. J. Spiegelhalter), pp. 59-74. London: Chapman & Hall. 

Zeger, S. L. and Karim, M. R. (1991) Generalized linear models with random 
effects: a Gibbs sampling approach. J. Am. Statist. Ass., 86, 79-86. 