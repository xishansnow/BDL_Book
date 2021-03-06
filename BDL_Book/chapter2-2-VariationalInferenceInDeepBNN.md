# 第 7 章 深层 BNN 的变分推断

## 7.1 机器学习的当前趋势

机器学习目前有三大趋势：概率编程、深度学习和“大数据”。在PP内部，很多创新是使用变分推理使事物变得有尺度。在这篇博客中，我将展示如何使用PyMC3中的变分推理来㼿t一个简单的贝叶斯神经网络。我还将讨论如何将概率编程和深度学习联系起来，以便在未来的研究中开辟非常有趣的途径。

### （1）大规模概率规划

概率编程允许非常灵活地创建自定义概率模型，并且主要关注对数据的洞察和学习。这种方法本质上是贝叶斯的，所以我们可以指定先验来通知和约束我们的模型，并以后验分布的形式得到不确定性估计。使用MCMC抽样算法，我们可以从这个后验到非常㼿抽取样本，很好地估计这些模型。PyMC3和Stan是目前构建和估计这些模型的最先进的工具。然而，采样的一个主要缺点是它通常非常慢，特别是对于高维模型。这就是为什么最近，变分推理算法被开发出来，它们几乎与mcmc一样具有㼿伸缩性，但速度要快得多。这些算法不是从后验抽取样本，而是㼿t分布(例如正态分布)到后验，将抽样问题转化为优化问题。在PyMC3和Stan中实现了自动微分变分推理，并实现了一个主要用于变分推理的新软件包Edward。

不幸的是，当涉及到传统的ML问题(如Classi㼿阳离子或(非线性)回归)时，概率编程通常比更多的算法方法(如随机森林或梯度增强回归树)发挥第二㼿DDLE(在准确性和可扩展性方面)。

### （2）深度学习

现在，深度学习已经进入第三次复兴，它通过主导几乎所有的物体识别基准，在雅达利游戏中大获全胜，并在围棋中击败世界冠军李世石(Lee Sedol)，一再登上头条新闻。从统计学的角度来看，神经网络是非常好的非线性函数逼近器和表示学习器。虽然最为人所知的是Classi㼿阳离子，但它们已经扩展到带有自动编码器的无监督学习和各种其他有趣的方式(例如，递归网络，或估计多峰分布的MDN)。为什么它们工作得这么好？没有人真正知道，因为统计特性仍未完全了解。

深度学习的很大一部分好处在于训练这些极其复杂的模型的能力。这取决于几个支柱：速度：加快图形处理器的速度，从而实现更快的处理速度。软件：像Theano和TensorFlow这样的框架允许㼿灵活地创建抽象模型，然后这些抽象模型可以优化并编译到中央处理器或图形处理器上。学习算法：在数据子集上进行训练-随机梯度下降-允许我们在海量数据上训练这些模型。像辍学这样的技术避免了过多的㼿考试。架构：很多创新来自于改变输入层，比如卷积神经网络，或者输出层，比如MDN。

### （3）深度学习与概率规划的桥梁

一方面，我们有概率编程，它允许我们以一种非常有原则和广为人知的方式建立相当小的、有重点的模型，以洞察我们的数据；另一方面，我们有深度学习，它使用许多启发式算法来训练巨大的、高度复杂的模型，这些模型在预测方面令人惊叹。变分推理中的最新创新使得概率编程能够衡量模型复杂性和数据大小。因此，我们正处于能够将这两种方法结合起来，有望解锁机器学习中的新创新的边缘。有关更多动机，请参见Dustin Tran最近的博客帖子。

虽然这将允许概率编程应用于更广泛的有趣问题，但我相信这种联系也为深度学习的创新带来了巨大的希望。一些想法是：预测中的不确定性：正如我们将在下面看到的，贝叶斯神经网络将其预测中的不确定性告知我们。我认为不确定性在机器学习中是一个被低估的概念，因为它显然对现实世界的应用程序很重要。但它在训练中也可能有用。例如，我们可以根据模型最不确定的样本来特定地训练模型㼿。表示中的不确定性：我们还可以得到权重的不确定性估计，这可以让我们了解学习到的网络表示的稳定性。有先验的正则化：权重通常是L2正则化的，以避免过多的㼿正则化，这非常自然地成为权重系数㼿的高斯先验。然而，我们可以想象所有其他类型的先验条件，比如强制稀疏性的钉子和板条(这更像是使用L1范数)。具有知情先验的转移学习：如果我们想要在新的对象识别数据集上训练网络，我们可以通过将知情先验以从其他预先训练的网络(如GoogLeNet)检索的权重为中心来引导学习。分层神经网络：概率编程中一种非常强大的方法是分层建模，它允许将在子组中学到的内容汇集到整个群体中(请参阅我在PyMC3中关于分层线性回归的教程)。应用于神经网络，在层次化的数据集中，我们可以训练单独的神经网络来专门研究子组，同时仍然被告知总体人口的表示。例如，设想一个经过训练的网络，用于从汽车图片中对汽车模型进行分类。我们可以训练一个层次化的神经网络，训练一个子神经网络，将型号与单一制造商区分开来。人们的直觉是，来自某一制造商的所有汽车都有某些相似之处，因此培训专门经营品牌的个人网络是有意义的。然而，由于各个网络连接在更高层，它们仍然会与其他专门子网络共享有关对所有品牌有用的功能的信息。有趣的是，网络的不同层可以由层次的不同级别通知-例如，提取视觉线条的早期层在所有子网络中可以是相同的，而更高阶的表示将是不同的。分层模型将从数据中了解所有这些信息。其他混合架构：我们可以更自由地构建各种神经网络。例如，贝叶斯非参数可以用于㼿灵活地调整隐藏层的大小和形状，以便在训练期间针对手头的问题最佳地缩放网络结构。目前，这需要昂贵的超参数优化和大量的部落知识。

贝叶斯神经网络