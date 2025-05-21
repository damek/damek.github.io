---
title: "Is AlphaEvolve problem B.1 hard?"
date: 2025-05-15
tags: [ai-for-math]
description: "Is AlphaEvolve problem B.1 hard? Yes"
---

Google released [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/). I tried to get a sense of whether the problems it solved were hard. I focused on one Problem B.1. 
> # B.1. First autocorrelation inequality
> For any function $f:\mathbb{R} \rightarrow \mathbb{R}$, define the *autoconvolution* of $f$, written $f*f$, as
> \begin{equation}
> f\*f (t) := \int_\mathbb{R} f(t-x) f(x)\ dx.
> \end{equation}
> Let $C_1$ denote the largest constant for which one has
> \begin{equation}
> \max_{-1/2 \leq t \leq 1/2} f\*f(t) \geq C_1 \left(\int_{-1/4}^{1/4} f(x)\ dx\right)^2
> \end{equation}
> for all non-negative $f: \mathbb{R} \rightarrow \mathbb{R}$.  This problem arises in additive combinatorics, relating to the size of Sidon sets.  It is currently known that
> \begin{equation}
> 1.28 \leq C_1 \leq 1.5098
> \end{equation}
> with the lower bound proven by [Cloninger and Steinerberger (2017)](https://www.ams.org/journals/proc/2017-145-08/S0002-9939-2017-13690-9/S0002-9939-2017-13690-9.pdf) and the upper bound achieved by [Matolcsi and Vinuesa (2010)](https://www.sciencedirect.com/science/article/pii/S0022247X10006001) via a step function construction. AlphaEvolve found a step function with 600 equally-spaced intervals on $[-1/4,1/4]$ that gives a better upper bound of $C_1 \leq 1.5053$.

- Read more about my attempts at solving the problem on [twitter](https://x.com/damekdavis/status/1924938162494717953). 
- See my code on [github](https://github.com/damek/alpha_evolve_problem_B1/blob/main/problemb1.ipynb).  
- View the [visualization](https://damek.github.io/alpha_evolve_problem_B1/).