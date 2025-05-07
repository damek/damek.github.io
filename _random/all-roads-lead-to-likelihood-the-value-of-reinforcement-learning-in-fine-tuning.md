---
title: "All roads lead to likelihood: the value of reinforcement learning in fine-tuning"
date: 2025-05-07
tags: [reinforcement-learning, fine-tuning]
description: "When DPO and RLHF are the same and / or different"
---

The paper ["All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning"](https://arxiv.org/pdf/2503.01067) examines why online, two-stage fine-tuning procedures like RLHF often appear to outperform direct offline methods such as DPO in aligning language models with human preferences. The core of the paper establishes an algebraic equivalence between these approaches under specific assumptions about the reward model's structure, and then hypothesizes that observed empirical differences arise from a "generation-verification gap," where learning a simpler reward model (verifier) separately is easier than jointly learning a policy (generator) and its implicit reward.

The foundation for modeling preferences is the Bradley-Terry (BT) model, where the probability of preferring trajectory $\xi\_1$ over $\xi\_2$ is 

$$P(\xi_1 \succ \xi_2) = \sigma(r(\xi_1) - r(\xi_2)),$$

with $r(\xi)$ being a scalar reward for trajectory $\xi$ and $\sigma$ the sigmoid function. Offline DPO directly optimizes a policy $\pi$ by defining an implicit "local" reward 

$$r_\pi(\xi) = \sum_{h=0}^{H-1} \log \pi(a_h|s_h).$$

DPO then maximizes 

$$\sum \log \sigma(r_\pi(\xi^+) - r_\pi(\xi^-))$$

over preference pairs $(\xi^+, \xi^-)$ by adjusting $\pi$. In contrast, online RLHF first learns an explicit "global" reward model $r\_G(\xi)$ by maximizing 

$$\sum \log \sigma(r_G(\xi^+) - r_G(\xi^-))$$

over parameters of $r\_G$. Subsequently, it learns a policy $\pi^*$ using this $r\_G(\xi)$, where the principle of maximum entropy RL implies the resulting policy yields a trajectory distribution $P\_{\pi^\* }^\* (\xi) \propto \exp(r\_G(\xi))$.

The paper's first main result (Theorem 2.2) demonstrates an algebraic equivalence: if the global reward model $r\_G$ in RLHF is constrained to have the same functional form as DPO's local reward, i.e., 

$$r_G(\xi) = r_{\pi'}( \xi) = \sum_h \log \pi'(a_h|s_h)$$ 

for some policy $\pi'$, then the two approaches are identical. Under this constraint (which the paper denotes $R=R(\Pi)$), Stage 1 of RLHF optimizes 

$$\sum \log \sigma(r_{\pi'}( \xi^+) - r_{\pi'}( \xi^-))$$ 

by varying $\pi'$. This is precisely the DPO objective, so the learned reward model is $r\_{\pi\_{DPO}}(\xi)$, where $\pi\_{DPO}$ is the policy DPO would find. 

Stage 2 of RLHF then seeks a policy $\pi^\*$ such that $P\_{\pi^\* }^\* (\xi) \propto \exp(r\_{\pi\_{DPO}}(\xi))$. Substituting $r\_{\pi\_{DPO}}(\xi) = \sum\_h \log \pi\_{DPO}(a\_h\|s\_h)$, we get 

$$P_{\pi^*}^*(\xi) \propto \exp(\sum_h \log \pi_{DPO}(a_h|s_h)) = \prod_h \pi_{DPO}(a_h|s_h).$$

This implies $\pi^\* = \pi\_{DPO}$. Thus, when the reward model architecture is restricted in this way, RLHF simply rearranges the DPO optimization.

To explain the empirically observed advantage of online methods, the paper proposes the "generation-verification gap" hypothesis (H6). This posits that the true underlying reward structure $r^\* $ dictating preferences might be "simple" (belonging to a class $R\_{sim}$ that is easy to learn) and that an unconstrained global reward model in RLHF's Stage 1 can effectively learn this $r^\* $. If DPO's implicit reward $r\_\pi(\xi)$ struggles to represent this $r^\* $ with a policy $\pi$ that is also easy to find, or if the joint optimization of finding such a $\pi$ is inherently harder, then RLHF gains an advantage. RLHF decouples the problem: first learn $r^\* \in R\_{sim}$, then derive a policy for it. DPO attempts to find a policy $\pi$ whose structure $r\_\pi$ simultaneously captures $r^*$ and defines an optimal policy. A related result (Theorem 3.1) formalizes that if DPO's search were restricted to policies optimal for some $r \in R\_{sim}$, it would match RLHF's outcome.

The paper presents experiments where manipulating task or reward complexity (e.g., very short trajectories, or using a complex predefined reward like ROUGE-L) alters the performance gap between online and offline methods. These are interpreted as supporting H6 by showing that when the generation-verification gap is presumed to be small (verification is as hard as generation, or generation is trivially easy), the online advantage diminishes. 