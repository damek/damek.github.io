---
title: "Weak baselines"
date: 2025-05-04
description: "Be careful with empirical claims"
---

- Enjoyed this post on [weak baselines](https://x.com/lateinteraction/status/1918798801982160935). 

![weak baselines](/assets/figures/weak-baselines.png)

- Been making [similar points myself](https://x.com/damekdavis/status/1911512233491964219).

![benchmarking](/assets/figures/benchmarking-is-hard.png)

- I emphasized in my class on [benchmarking optimizers](https://damekdavis.com/STAT-4830/section/10/notes.md), that one shouldn't try to make such empirical claims unless one has the tuning budget of Google. 
- [My lecture](https://damekdavis.com/STAT-4830/section/11/notes.md) on the [deep learning tuning playbook](https://github.com/google-research/tuning_playbook) discussed how to choose  an initial baseline and then iteratively improve it: "hill climbing on baselines." 
- Theory papers should focus on finding the strongest baseline, too. If you look into the optimization literature, one might be surprised by how many do not improve any quantifiable metric, but instead develop a more "flexible and general method." This is a problem with methods driven research. Some better ways to set the baseline in theory of optimization include showing your method provably:
    - has a larger initialization region
    - accelerates over sota in a neighborhood of the solution
    - "works" for a class of problems where no principled method worked before
    - has improved sample complexity or rate of convergence.
    - reduces dependence on condition number while being implementable for specific empirical problems. 
    - does not require knowing certain unknowable problem parameters for implementation
    - ...