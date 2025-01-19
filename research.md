---
layout: default
title: Research Overview - Damek Davis
---

# Research Overview

Learning algorithms work exceptionally well in practice, but we have yet to find a coherent mathematical foundation explaining when they work and how to improve their performance. The challenge is that most learning algorithms rely on fitting highly nonlinear models via simple nonconvex optimization heuristics, and except for a few exceptional cases, there is no guarantee they will find global optima. Despite this and NP-hardness, simple heuristics often succeed, and over the last few years, I have studied why and when they do.

I spend a lot of time thinking about neural networks. I am particularly interested in whether we can provide provable convergence guarantees to standard training algorithms or substantially improve existing methods. Deep networks fall outside the scope of classical optimization theory since they lead to problems that lack conventionally helpful notions of convexity or smoothness. Taking the inherent *nonsmooth* structure of neural networks seriously is crucial to understand these methods. I study this structure and the associated algorithms by using and developing tools from several disciplines, including [nonsmooth/variational analysis](https://link.springer.com/book/10.1007/978-3-642-02431-3), [tame geometry](https://epubs.siam.org/doi/10.1137/080722059), and [high-dimensional statistics](https://www.cambridge.org/core/books/highdimensional-statistics/8A91ECEEC38F46DAB53E9FF8757C7A4E). 

## Algorithms for nonsmooth optimization in machine learning
While neural networks are nonsmooth, they are not pathological — they are built from just a few simple components, like polynomials, exponentials, logs, max's, min's, and absolute values. The best model of such non-pathological functions available in optimization is the so-called *tame class,* a class which appears in several of my papers and precludes [cantor function](https://en.wikipedia.org/wiki/Cantor_function)-esque behavior. I have spent much time trying to uncover notions of beneficial "partial" smoothness in tame optimization problems to exploit this structure in algorithms. 

While tame problems comprise virtually all tasks of interest, they lack enough structure to endow simple iterative methods with global efficiency guarantees. A class with more structure, which I view as a stepping stone between convex functions and general neural networks, is the so-called *weakly convex* class. These are functions that differ from convex functions by a simple quadratic. This class is deceptively simple yet surprisingly broad. It includes, for example, all C^2 smooth functions (on compact sets) and all compositions of Lipschitz convex functions with smooth mappings: h(c(x)). These losses appear throughout data science, particularly in low-rank matrix recovery problems (e.g., matrix completion and sensing). 

My group has been working towards understanding the convergence of simple iterative methods, such as the stochastic subgradient method (SGD), on the tame and weakly convex problem classes. We have also been working towards designing methods that outperform SGD. 

I will briefly summarize some of the contributions of my group. For those interested, you can find a brief technical introduction to some of my papers in the expository note: 

[Subgradient methods under weak convexity and tame geometry](research/papers/ViewsAndNews-28-1.pdf) 
Damek Davis, Dmitriy Drusvyatskiy
SIAG/OPT Views and News (2020)

### An exponential speedup for "generic" tame problems
We developed the first first-order method that (locally) converges nearly linearly (i.e., exponentially fast) on "generic" tame problems. This result shows that we can exponentially(!) surpass the "speed limit" of gradient methods derived by Nemirovski and Yudin in the 80s -- if we wait a bit. The result applies to "almost every problem" in practice. We found this super surprising!

[A nearly linearly convergent first-order method for nonsmooth functions with quadratic growth](https://arxiv.org/abs/2205.00064) 
Damek Davis, Liwei Jiang
Foundations of Computational Mathematics (to appear) | [code](https://github.com/COR-OPT/ntd.py) | [Twitter thread](https://twitter.com/damekdavis/status/1682389849167233027?s=20)

### A superlinearly convergent method for "generic" tame equations
We developed the first algorithm that (locally) converges nearly superlinearly (i.e., double exponentially fast) on "generic" tame equations. 

[A superlinearly convergent subgradient method for sharp semismooth problems](https://arxiv.org/abs/2201.04611) 
Vasileios Charisopoulos, Damek Davis
Mathematics of Operations Research (2023) | [code](https://github.com/COR-OPT/SuperPolyak.py) | [Twitter thread](https://twitter.com/damekdavis/status/1596616542396944384)

### Training guarantees for SGD on tame and weakly convex functions
We showed that the stochastic subgradient method (e.g., backpropagation) converges to first-order critical points on virtually any neural network.

[Stochastic subgradient method converges on tame functions](https://arxiv.org/abs/1804.07795) 
Damek Davis, Dmitriy Drusvyatskiy, Sham Kakade, Jason D. Lee
Foundations of Computational Mathematics (2018) | [Talk](research/presentations/ICCOPT2019.pdf) 

We proved the first sample/computational efficiency guarantees for the stochastic subgradient method on the weakly convex class. 

[Stochastic model-based minimization of weakly convex functions](https://arxiv.org/abs/1803.06523) 
Damek Davis, Dmitriy Drusvyatskiy
SIAM Journal on Optimization (2018) | [blog](http://ads-institute.uw.edu//blog/2018/04/02/sgd-weaklyconvex/) 

[Proximally Guided Stochastic Subgradient Method for Nonsmooth, Nonconvex Problems.](https://arxiv.org/abs/1707.03505) 
Damek Davis, Benjamin Grimmer
SIAM Journal on Optimization (2018) [[code](https://github.com/COR-OPT/PGSG/blob/master/Interactive-PGSG.ipynb)]

### Avoidable saddle points in nonsmooth optimization
We developed the concept of an avoidable nonsmooth saddle point — nonoptimal points that algorithms may approach. The proper formulation of this concept is well-known in C^2 smooth optimization but was missing even for C^1 functions. We showed that both first-order and proximal methods do not converge to these points on "generic" tame problems:

[Talk: avoiding saddle points in nonsmooth optimization](research/presentations/OWOSNov2021.pdf) 
Updated (11/2021) | [video](https://www.youtube.com/watch?v=6BOFWQhxYZE)

[Proximal methods avoid active strict saddles of weakly convex functions](https://arxiv.org/abs/1912.07146) 
Damek Davis, Dmitriy Drusvyatskiy
Foundations of Computational Mathematics (2021)

[Escaping strict saddle points of the Moreau envelope in nonsmooth optimization](https://arxiv.org/abs/2106.09815) 
Damek Davis, Mateo Díaz, Dmitriy Drusvyatskiy
SIAM Journal on Optimization (2022)

[Active manifolds, stratifications, and convergence to local minima in nonsmooth optimization](https://arxiv.org/abs/2108.11832) 
Damek Davis, Dmitriy Drusvyatskiy, Liwei Jiang
Manuscript (2022)

### Asymptotic normality of SGD in nonsmooth optimization
We characterized the asymptotic distribution of the error sequence in stochastic subgradient methods, proving it is asymptotically normal with "optimal covariance" on "generic" tame problems.

[Asymptotic normality and optimality in nonsmooth stochastic approximation](https://arxiv.org/abs/2301.06632) 
Damek Davis, Dmitriy Drusvyatskiy, Liwei Jiang
Manuscript (2023)

### Low-rank matrix recovery: a stepping stone to neural networks
We achieved the first sample complexity optimal and computationally optimal methods for several low-rank matrix recovery based on *nonsmooth* weakly convex formulations. Nonsmoothness was crucial to establishing these rates since prior smooth formulations suffered from "poor conditioning."
 
[Composite optimization for robust rank one bilinear sensing](https://academic.oup.com/imaiai/advance-article-abstract/doi/10.1093/imaiai/iaaa027/5936039) 
Vasileios Charisopoulos, Damek Davis, Mateo Diaz, Dmitriy Drusvyatskiy
IMA Journal on Information and Inference (2020) [ [code](https://github.com/COR-OPT/RobustBlindDeconv) ]

[The nonsmooth landscape of phase retrieval](https://academic.oup.com/imajna/article-abstract/40/4/2652/5684995) 
Damek Davis, Dmitriy Drusvyatskiy, Courtney Paquette
IMA Journal on Numerical Analysis (2017) | [Talk](research/presentations/NonsmoothStatisticalAssumptions.pdf) 

 [Low-rank matrix recovery with composite optimization: good conditioning and rapid convergence](https://arxiv.org/abs/1904.10020) 
Vasileios Charisopoulos, Yudong Chen, Damek Davis, Mateo Díaz, Lijun Ding, Dmitriy Drusvyatskiy
Foundations of Computational Mathematics (2019) | [code](https://github.com/COR-OPT/CompOpt-LowRankMatrixRecovery) 

## Other selected work 
Besides my work on nonconvex learning algorithms, I also have worked on clustering and convex optimization algorithms. 

### Provable clustering methods and a potential statistical-to-computational gap
Clustering is a fundamental statistical problem of dividing a dataset into two or more groups. Our work on this talk topic focuses on the classical setting wherein both clusters follow a Gaussian distribution with identical covariance but distinct means. When the covariance matrix is known or "nearly spherical," there are efficient algorithms to perform the clustering and achieve the "Bayes-optimal error rate." When the covariance is unknown or poorly conditioned, no known algorithms achieve the Bayes-optimal rate. 

Our contribution to this topic is a surprising dichotomy for clustering with an unknown covariance matrix: on the one hand, the maximum likelihood estimator uncovers the correct clustering and achieves the Bayes-optimal error; on the other, we give evidence that no known algorithm can compute the maximum likelihood estimator unless one increases the number of samples by an order of magnitude. Thus, we conjecture that there is a [statistical-to-computational gap](https://arxiv.org/abs/1803.11132) for this classical statistical problem.

[Clustering a Mixture of Gaussians with Unknown Covariance](https://arxiv.org/abs/2110.01602) 
Damek Davis, Mateo Diaz, Kaizheng Wang
Manuscript (2021)

### Three-Operator Splitting and the complexity of splitting methods
I focused on a class of convex optimization algorithms called operator-splitting methods for my PhD thesis. An operator splitting method is a technique for writing the solution of a "structured" convex optimization problem as the fixed-point of a well-behaved nonlinear operator. Algorithmically, one then finds the fixed-point of the operator through, e.g., the classical [fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration). My best-known contributions to the topic include the (1) "Three-Operator-Splitting" method, which has been widely used throughout computational imaging, and (2) my work that established the convergence rates of several classical splitting methods, such as the Douglas-Rachford splitting method and Alternating Direction Method of Multipliers (ADMM).

[A Three-Operator Splitting Scheme and its Optimization Applications](https://link.springer.com/article/10.1007/s11228-017-0421-z) 
Damek Davis, Wotao Yin
Set-Valued and Variational Analysis (2017)

[Convergence rate analysis of several splitting schemes](http://arxiv.org/abs/1406.4834) 
Damek Davis, Wotao Yin
Splitting Methods in Communication and Imaging, Science and Engineering (2017) 