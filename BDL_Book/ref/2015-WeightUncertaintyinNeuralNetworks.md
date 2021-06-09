# 神经网络中的权重不确定性

Weight Uncertainty in Neural Networks
- 【B】Bayesian Deep Collaborative Matrix Factorization

其中前者是贝叶斯神经网络的代表之作，在其中，**权重不再是一个固定的数，而是一个概率分布**；后者讲述了贝叶斯深度学习在推荐系统上的一个应用。

-

## **【A】Weight Uncertainty in Neural Networks**

作者：Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra

来源：ICML2015

### **1 动机概述**

在经典的深度学习当中，神经网络的权重被当做一个固定的数来处理，这篇文章提出了一个非常新颖的框架，即神经网络的权重**不再是一个固定的数，而是一个概率分布**，使用该权重的时候相当于是从这个概率分布中进行抽样。

那么为什么需要使用分布而不是固定的数呢？文章提出了三个Motivation：

- （1）通过权重的compression cost来达到正则化的效果
- （2）更高质量的潜在表示
- （3）在Contextual bandits等强化学习任务中, 文章提出的Bayes by Backprop可以更好地学习如何权衡Exploration和Exploitation

而这么做的优势在于：可以表达特定观测的**Uncertainty**，也能作为一种**权重的正则化**方法，同时，当权重是一个分布的时候，每次采样出来的数自然是有区别的，这不就相当于在内部融合了**model averaging**嘛！

具体而言，本文中的神经网络的所有权重都是以一个分布的形式出现，和传统的神经网络有所不同，如下图所示：

![](https://pic4.zhimg.com/v2-368983796804ad6089786566f7ffa13b_b.jpg)

![](https://pic4.zhimg.com/80/v2-368983796804ad6089786566f7ffa13b_720w.jpg)

传统深度学习中神经网络的权重是一个固定的数值，而这篇文章中是一个分布，当然这就会引起新的问题，比如说如何去进行训练和优化？还能像通常的神经网络一样进行反向传播吗？

很显然，在一个复杂的神经网络中，精确的贝叶斯推断是intractable的，因此在具体方法上，文章对Bayesian Updates使用变分近似。

![](https://pic2.zhimg.com/v2-5794439d7a42ebd44131a29bb051abc5_b.jpg)

![](https://pic2.zhimg.com/80/v2-5794439d7a42ebd44131a29bb051abc5_720w.jpg)

### **2 具体方法**

这一部分，文章介绍了处理贝叶斯深度学习模型的具体方法，大概分为如下的几步：

**Unbiased Monte Carlo gradients**

下图显示了贝叶斯深度学习中损失函数的形式，主要就是两项构成，其中第一项和**参数先验**有关，称为**Complexity Cost**，第二项和数据有关，称为**Likelihood Cost**。

![](https://pic2.zhimg.com/v2-6d2838268f2a68fe932e7a699aa2ce35_b.jpg)

![](https://pic2.zhimg.com/80/v2-6d2838268f2a68fe932e7a699aa2ce35_720w.jpg)

那么我们如何去优化这个损失的式子？我们会发现，这个和传统深度学习的损失不一样呀！里面出现了期望项，**该如何去对一个期望求导呢？**文章使用了如下这个定理:

![](https://pic1.zhimg.com/v2-f1c2d4d354532f38522e0daf3a34a0b0_b.jpg)

![](https://pic1.zhimg.com/80/v2-f1c2d4d354532f38522e0daf3a34a0b0_720w.jpg)

  

这个定理讲的事情就是**一定条件下期望和求导可交换**，通过这种方式，我们就可以对损失函数进行优化了。注意，这个定理其实是高斯分布重参数化技巧的一个更一般化的版本。

另外，本文的创新点（或者说和以前文章不同之处）在于没有使用闭式的Complexity cost，这么做的好处是可以使用更多类型的先验和变分后验族。

**Gaussian variational posterior**

知道了我们应该如何对损失函数去做近似，接下来便讨论一下：**想要比较方便地去近似，我们需要什么样的假设**。

这里提了一种最为简单的假设：**变分后验为高斯分布**。在这种情况下，权重可以通过从![[公式]](https://www.zhihu.com/equation?tex=N%28%5Cmu%2C%5Csigma%5E2%29)的高斯分布中抽样得到，其中假设参数![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%3D%5Clog+%281%2B%5Cexp%28%5Crho%29%29)，则变分后验参数为![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta%3D%28%5Cmu%2C%5Crho%29)，另外我们可以在其中加入噪声，则优化的过程为：

![](https://pic3.zhimg.com/v2-62cff92cb97179aca281420fa81ae056_b.jpg)

![](https://pic3.zhimg.com/80/v2-62cff92cb97179aca281420fa81ae056_720w.jpg)

注意：在这个过程中，我们其实是通过通常的方式计算反向传播的梯度，只不过是用上述方式转化一下而已。

**Scale Mixture Prior**

明确了使用什么方式去近似，不如考虑下一个问题：**参数的先验是什么**。在这一部分，文章提出来使用两个高斯的Scale Mixture作为先验，他们的均值都为零，而方差不相同：

![[公式]](https://www.zhihu.com/equation?tex=P%28%5Cmathbf%7Bw%7D%29%3D%5Cprod_j%5Cpi+N%28%5Cmathbf%7Bw%7D_j%7C0%2C%5Csigma_1%5E2%29%2B%281-%5Cpi%29N%28%5Cmathbf%7Bw%7D_j%7C0%2C%5Csigma_2%5E2%29+%5C%5C)

这个类似经典的spike-and-slab先验，不过其中先验的参数在所有权重中共享。这种方式的好处在于：**无需在优化过程中更新先验的参数**。

**Minibatches and KL re-weighting**

最后大概就是落实到具体的优化过程了，文章提出该损失也可以通过minibatch优化：在优化的每个Epoch中训练数据随机分为M个相同大小的子集，每个子集的梯度为子集元素的平均。

### **3 与强化学习的联系：Contextual Bandits**

使用Weight Uncertainty有什么样的实际应用价值呢？文章举出了一个具体的例子：Contextual Bandits——最简单的一个强化学习任务：在每一步agent得到一个context ![[公式]](https://www.zhihu.com/equation?tex=x)，并从K个可能的行动中选取一个![[公式]](https://www.zhihu.com/equation?tex=a)，每个![[公式]](https://www.zhihu.com/equation?tex=a)会有不同的回报![[公式]](https://www.zhihu.com/equation?tex=r)。

该场景下，一个最主要的问题就是**agent不能知道它未选取的action回报是多少**，这便是一个经典的问题：**The absence of conuterfactual**，即我们无从得知未发生过的事情的信息。

**Thompson Sampling for Neural Networks**

我们将Contextual Bandits要解决的问题进行明确，在这个问题中，**需要学习的就是![[公式]](https://www.zhihu.com/equation?tex=P%28r%7Cx%2Ca%2C%5Cmathbf%7Bw%7D%29)**：其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D)是待学习参数。拿到这个式子的第一反应一般都是：上深度学习呀！把参数给直接学出来。

本文指出：参数![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D)自然可以用神经网络学习，然而如果仅仅根据观测和每步最高回报对应的action来进行拟合，则很可能出现**under-explore**的问题。

Thompson Sampling有助于解决exploitation和exploration权衡的问题，其核心也是贝叶斯的思想：使用一组参数并选择和这些参数相关的action——**如果这个参数“大概率会比较好”，那么它会更为频繁地被选中**。

具体而言：开始的时候，变分后验与先验分布非常接近，**所有action几乎同等概率被选取**。随着时间的推移，action被选取后，变分后验将开始收敛，许多参数的Uncertainty就开始逐渐降低，从而**action选取的随机性也会逐渐降低**，即开始慢慢集中于选择高回报期望的action。

-

## **【B】Bayesian Deep Collaborative Matrix Factorization**

作者：Teng Xiao, Shangsong Liang, Weizhou Shen, Zaiqiao Meng

来源：AAAI2019

### **1 动机概述**

这篇文章算是贝叶斯深度学习的一个应用，文章很有意思，顾名思义，模型就是Bayesian+Deep网络+Collaborative用在矩阵分解上。

我们都知道，推荐系统中，除了**用户-物品评分矩阵**之外，往往还有一些额外的信息可以利用，比如**用户的社交网络关系**，**物品自己的特征**。我们的目的其实是学习**用户和物品的潜在表示（Latent Vector）**，从而用于众多下游任务重。那么能否同时使用评分矩阵、用户网络和物品特征去学习用户和物品的潜在表示呢？

传统方法比如矩阵分解，往往会面临着如下问题：比如**稀疏性、忽略用户之间的interaction、只能捕捉到用户和物品之间的线性关系**等，当然一些更新一点的方法，比如hybrid MF（如CTR或CDL）能在一定程度上解决上述问题，然而它们使用的**点估计方法**往往会导致**高variance**以及**忽略了参数内在的uncertainty**。

文章的模型正是用来解决这些问题的，文章指出：user和item的附加信息能解决矩阵稀疏问题，通过结合贝叶斯概率的框架，能学到更robust和紧凑的潜在表示。

当然这个问题并不容易求解。因此，本文设计了一个高效的EM-style点估计算法去学习参数，并推导了Full Bayesian的后验估计。

文章主要有以下几个贡献点：

- （1）同时利用物品的文本信息（content），用户网络（social interaction）和用户物品评分矩阵学习用户和物品的隐向量
- （2）设计了高效的并行变分EM-style算法做贝叶斯点估计
- （3）设计了Full Bayesian的变分后验估计方法推断潜因子的后验分布

### **2 具体方法**

**基本模型**

文章的符号多且杂，因此在介绍文章的方法之前，先需要弄清楚其中一些符号的含义：

![](https://pic4.zhimg.com/v2-43adf86297463456a2e85e679a88e103_b.jpg)

![](https://pic4.zhimg.com/80/v2-43adf86297463456a2e85e679a88e103_720w.jpg)

该文章模型的核心就是下面这个概率图模型：其中灰色的为可观测变量，实线为生成过程，而虚线为推断过程：

![](https://pic1.zhimg.com/v2-9510f79a9dbc330eb23b0a2d89cc8a70_b.jpg)

![](https://pic1.zhimg.com/80/v2-9510f79a9dbc330eb23b0a2d89cc8a70_720w.jpg)

可以看到，我们的观测（即模型的输入）就是**评分矩阵![[公式]](https://www.zhihu.com/equation?tex=R)，用户网络![[公式]](https://www.zhihu.com/equation?tex=S)，物品特征![[公式]](https://www.zhihu.com/equation?tex=X)**，根据这个概率图模型，以下的式子成立：

![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cmathcal%7BO%7D%2C%5Cmathcal%7BZ%7D%29%3D%5Cprod_%7Bi%3D1%7D%5EN%5Cprod_%7Bj%3D1%7D%5EM%5Cprod_%7Bk%3D1%7D%5EN+p%28%5Cmathcal%7BO%7D_%7Bijk%7D%2C%5Cmathcal%7BZ%7D_%7Bijk%7D%29%3D%5Cprod_%7Bi%3D1%7D%5EN%5Cprod_%7Bj%3D1%7D%5EM%5Cprod_%7Bk%3D1%7D%5EN+p%28%5Cmathbf%7Bz%7D_j%29+p%28%5Cmathbf%7Bg%7D_k%29+p%28%5Cmathbf%7Bu%7D_i%29+p_%5Ctheta%28%5Cmathbf%7Bx%7D_j%7C%5Cmathbf%7Bz%7D_j%29+p%28%5Cmathbf%7Bv%7D_j%7C%5Cmathbf%7Bz%7D_j%29+p%28R_%7Bij%7D%7C%5Cmathbf%7Bu%7D_i%2C%5Cmathbf%7Bv%7D_j%29+p%28S_%7Bik%7D%7C%5Cmathbf%7Bu%7D_i%7C%5Cmathbf%7Bg%7D_k%29+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO%7D%3D%5C%7BR%2CS%2CX%5C%7D)为可观测变量，而![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BZ%7D%3D%5C%7BU%2CV%2CG%2CZ%5C%7D)为因变量（需要进行推断）。该算法的具体过程如下：

![](https://pic3.zhimg.com/v2-de2f32598180e6bd09777b0215e6cb7a_b.jpg)

![](https://pic3.zhimg.com/80/v2-de2f32598180e6bd09777b0215e6cb7a_720w.jpg)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_v%2C%5Clambda_u%2C%5Clambda_g%2C%5Clambda_q%2C%5Cpsi_1%3E%5Cpsi_2%3E0%2C+%5Cpsi_3%3E%5Cpsi_4%3E0)都为free parameter。![[公式]](https://www.zhihu.com/equation?tex=p_%5Ctheta%28%5Cmathbf%7Bx%7D_j%7C%5Cmathbf%7Bz%7D_j%29)代表物品的content information（![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bx%7D_j)由latent content vector ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D_j)由参数为![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)的神经网络生成），所有变量的先验假设都是特定均值和方差的高斯分布。

接下来文章介绍了优化这个模型的两种方法：Bayesian Point Estimation和Bayesian Posterior Estimation。

**Bayesian Point Estimation**

贝叶斯点估计的过程如下：

![](https://pic3.zhimg.com/v2-22fdcf204d66d921a095b4085f867fa2_b.jpg)

![](https://pic3.zhimg.com/80/v2-22fdcf204d66d921a095b4085f867fa2_720w.jpg)

注意，在这个复杂的模型中，传统的变分EM算法是难以推断![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BZ%7D)的后验的，因此文章借鉴VAE的想法：**使用变分分布![[公式]](https://www.zhihu.com/equation?tex=q_%5Cphi%28Z%7CX%2CR%29)去近似真实的后验![[公式]](https://www.zhihu.com/equation?tex=p%28Z%7C%5Cmathcal%7BO%7D%29)**。其中对于![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D_j)，有：

![[公式]](https://www.zhihu.com/equation?tex=q%28%5Cmathbf%7Bz%7D_j%29%3Dq_%5Cphi%28%5Cmathbf%7Bz%7D_j%7C%5Cmathbf%7Bx%7D_j%2CR_%7B.j%7D%29%3DN%28%5Cmu_j%2Cdiag%28%5Cdelta_j%5E2%29%29+%5C%5C)

均值![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_j)和方差![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta_j)都是推断网络（Inference Network）的输出。

对于公式（10）的ELBO，难以直接去优化，因此本文提出了一个迭代式的变分EM（VEM）算法去求解这个优化问题（最大化![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D_%7Bpoint%7D)）：

![](https://pic3.zhimg.com/v2-ac86bc132531e5f3a0438513afbe261a_b.jpg)

![](https://pic3.zhimg.com/80/v2-ac86bc132531e5f3a0438513afbe261a_720w.jpg)

先来看E-step，其中![[公式]](https://www.zhihu.com/equation?tex=R%2CS)就是之前定义过的用户-物品矩阵和用户关系网络（![[公式]](https://www.zhihu.com/equation?tex=C)和![[公式]](https://www.zhihu.com/equation?tex=E)的定义分别与![[公式]](https://www.zhihu.com/equation?tex=R)和![[公式]](https://www.zhihu.com/equation?tex=S)有关，在前面的算法步骤中提到过）。

显然，![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_v)定义了物品content information对物品隐向量的影响程度，![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_q)起到了平衡用户网络和用户物品矩阵的作用。因此![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_v), ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_q)可以看作**collaborative parameters**：**协调控制着物品特征，用户网络和用户-物品矩阵三方面信息的影响**。

再看M-step，其中![[公式]](https://www.zhihu.com/equation?tex=M)为物品的数量，由于期望一项不能求得解析解，因此使用蒙特卡洛方法进行近似处理：

![](https://pic3.zhimg.com/v2-ff3bbeed75fb9d1fd879744f5c33922e_b.jpg)

![](https://pic3.zhimg.com/80/v2-ff3bbeed75fb9d1fd879744f5c33922e_720w.jpg)

该算法的完整流程如下所示：

![](https://pic1.zhimg.com/v2-dc2b5c0ea0533ae2cfff606dc2b9678c_b.jpg)

![](https://pic1.zhimg.com/80/v2-dc2b5c0ea0533ae2cfff606dc2b9678c_720w.jpg)

**Bayesian Posterior Estimation**

前面的贝叶斯点估计方法可能会导致过于乐观的估计，以及估计量的高方差，并且其实忽略了参数内在的Uncertainty的问题。然而与基于CTR或者CDL的推荐系统不同，文章这个BDCFM模型还可以使用full Bayesian的后验估计！

这里文章对变分分布的假设和公式（7）相同，为Matrix-wise Independent。对于![[公式]](https://www.zhihu.com/equation?tex=p%28%5Cmathbf%7Bz%7D_j%29)我们引入变分分布![[公式]](https://www.zhihu.com/equation?tex=q_%5Cphi%28Z%7CX%29)来近似真实后验![[公式]](https://www.zhihu.com/equation?tex=p%28Z%7C%5Cmathcal%7BO%7D%29)，对于![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bz%7D_j)的假设依然为![[公式]](https://www.zhihu.com/equation?tex=q%28%5Cmathbf%7Bz%7D_j%29%3Dq_%5Cphi%28%5Cmathbf%7Bz%7D_j%7C%5Cmathbf%7Bx%7D_j%2CR_%7B.j%7D%29%3DN%28%5Cmu_j%2Cdiag%28%5Cdelta_j%5E2%29%29).

与算法1不同的是，在这里我们**没有假设![[公式]](https://www.zhihu.com/equation?tex=q%28%5Cmathbf%7Bu%7D_i%29%2Cq%28%5Cmathbf%7Bv%7D_j%29%2Cq%28%5Cmathbf%7Bg%7D_k%29)分布的具体形式**。优化的目标还是最大化变分下界![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D%28q%29)

![](https://pic2.zhimg.com/v2-4b550b27adc1b78d8b7e421775636a71_b.jpg)

![](https://pic2.zhimg.com/80/v2-4b550b27adc1b78d8b7e421775636a71_720w.jpg)

可以看到，优化的过程和贝叶斯点估计是相同的。最后就是如何去预测（用户对物品的评分）的问题了，毕竟辛辛苦苦得到的用户和物品的潜在表示![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bu%7D_i%2C%5Cmathbf%7Bv%7D_j)应该要用起来！

![[公式]](https://www.zhihu.com/equation?tex=R_%7Bij%7D%5E%7B%2A%7D%3D%28%5Cmathbf%7Bz%7D_i%2B%5Cmathbf%7Bk%7D_i%29%5ET%5Cmathbf%7Bu%7D_i%3D%5Cmathbf%7Bv%7D_j%5ET%5Cmathbf%7Bu%7D_i+%5C%5C)

对于新出现的物品，可以将![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon_j)设置为0，那么预测的分数就应该是：

![[公式]](https://www.zhihu.com/equation?tex=R_%7Bij%7D%5E%7B%2A%7D%3D%5Cmathbf%7Bz%7D_j%5ET%5Cmathbf%7Bu%7D_i+%5C%5C)

文章提出的模型同样具备根据不同场景改造的灵活性（比如序列中可以将神经网络变为RNN等）。