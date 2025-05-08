---
title: "Getting the hang of policy gradients by reframing optimization as RL"
date: 2025-05-07
---


We can gain insight into Reinforcement Learning (RL) training mechanisms by taking a general optimization problem and reframing it as a "stateless RL problem." In this reframing, the task is to find optimal parameters for a probability distribution that generates candidate solutions. The distribution's parameters are adjusted based on rewards received for sampled solutions, aiming to increase the probability of generating high-reward solutions. This perspective isolates aspects of policy optimization from complexities such as state-dependent decision making, which allows for a focused study of certain RL mechanisms.

### From Optimization to a Stateless RL 

Consider the general problem of finding a vector $w$ that minimizes a loss function $L(w)$:

$$ \min_w L(w) $$

$L(w)$ can be a deterministic function of $w$. Alternatively, $L(w)$ might represent an expected loss, 

$$L(w) = E_{z \sim D_z}[\ell(w,z)]$$

where $\ell(w,z)$ is a loss computed for a specific data sample $z$ drawn from a distribution $D_z$.

To transform this into an RL problem, we shift from directly seeking an optimal $w$ to optimizing the parameters $\theta$ of a policy $\pi\_\theta(w)$. The policy $\pi\_\theta(w)$ is a probability distribution that generates candidate solutions $w$. The RL objective is then to find parameters $\theta$ that maximize the expected reward $J(\theta)$ obtained from these solutions:

$$ J(\theta) = E_{w \sim \pi_\theta(w)}[R(w)] $$

If the policy is $\pi\_\theta(w) = N(w\|\mu, \sigma^2)$ and the reward is $R(w) = -L(w)$, with $L(w)$ uniquely minimized at $w^\* $, then $J(\theta)$ is maximized as the policy mean $\mu$ approaches $w^\*$ and the standard deviation $\sigma$ approaches $0^+$. In this limit, the policy effectively concentrates its mass at $w^\* $.

In this construction:
*   The **policy** $\pi_\theta(w)$, parameterized by $\theta$, is the distribution used to sample candidate solutions $w$.
*   An **action** corresponds to sampling a solution $w$ from $\pi_\theta(w)$.
*   The environment is effectively **stateless** because the reward $R(w)$ for a sampled solution $w$ depends only on $w$.

The definition of the reward $R(w)$ is derived from the original optimization problem. If the goal is to minimize $L(w)$, one reward definition is $R(w) = -L(w)$. If the loss is stochastic, $\ell(w,z)$, then the reward can also be stochastic, for example, $R(w,z) = -\ell(w,z)$. In this stochastic reward case, the objective becomes $J(\theta) = E_{w \sim \pi_\theta(w)}[E_{z \sim D_z}[R(w,z)]]$.

### Example Reward Formulations

The way $R(w)$ (or $R(w,z)$) is defined based on $L(w)$ or $\ell(w,z)$ directly influences the information available to the learning algorithm. To explore these effects, we consider several reward structures:

1.  **True Reward:** $R(w) = -L(w)$.
    *   This reward provides direct information about $L(w)$ for each sampled $w$.
2.  **Unbiased Randomized Reward:** $R(w,z) = -\ell(w,z)$, where $z \sim D_z$.
    *   This reward uses individual data instances $z$. $R(w,z)$ is noisy, but $\mathbb{E}\_z[R(w,z)] = -L(w)$.
3.  **Batch Reward (Proxy for Hacking):** $R(w) = -\frac{1}{\|S\_{train}\|} \sum\_{z\_i \in S\_{train}} \ell(w,z\_i)$. $S\_{train}$ is a fixed data batch.
    *   If $S\_{train}$ is unrepresentative, optimizing this $R(w)$ can illustrate reward hacking.
4.  **Discrete Rewards (Sparsity):** $R(w) = \mathbb{1}\_{-L(w) > \text{tol}}$, where $\text{tol}$ is a threshold.
    *   This yields a binary signal. If the condition is met infrequently, the reward is sparse.
5.  **Sparse Continuous Rewards:** $R(w) = -L(w) \cdot \mathbb{1}\_{-L(w) > \text{tol}}$.
    *   This reward is also sparse but provides magnitude information once the performance threshold is crossed.

### The Policy Gradient in a Stateless Setting

The Policy Gradient Theorem (PGT) provides a method for computing $\nabla\_\theta J(\theta)$. (A PGT derivation is in a [previous post](/random/basic-facts-about-policy-gradients)). For the stateless problem, the policy gradient is:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{w \sim \pi_\theta(w), z \sim D_z}[ \nabla_\theta \log \pi_\theta(w) R(w,z) ] \quad (*) $$

(If $R(w)$ is deterministic, omit expectation over $z$). A stochastic estimate of $\nabla\_\theta J(\theta)$ is $\hat{g}\_t = \nabla\_\theta \log \pi\_\theta(w\_t) R(w\_t, z\_t)$, using samples $w\_t \sim \pi\_{\theta\_t}(w)$ and $z\_t \sim D\_z$. Policy parameters can then be updated via stochastic gradient ascent:

$$ \theta \leftarrow \theta + \alpha \hat{g}_t $$

#### Properties of the Policy Gradient Estimator

The estimator $\hat{g}\_t = \nabla\_\theta \log \pi\_\theta(w\_t) R(w\_t, z\_t)$ is an unbiased estimate of $\nabla\_\theta J(\theta)$. The variance of this estimator, $\text{Var}(\hat{g}\_t) = \mathbb{E}[(\nabla\_\theta \log \pi\_\theta(w) R(w,z))^2] - (\nabla\_\theta J(\theta))^2$, can be large. This variance impacts learning stability and speed.

For example, consider a policy $\pi\_\theta(w) = N(w \| \mu, \sigma^2 I\_d)$ for $w \in \mathbb{R}^d$. Let parameters be $\theta = (\mu, \psi)$, where $\psi = \log\sigma$ ensures $\sigma = e^\psi > 0$. The score function for $\mu$ is 

$$\nabla_\mu \log \pi_\theta(w) = (w-\mu)/\sigma^2.$$

The variance of the $\mu$-component of $\hat{g}\_t$, $\hat{g}\_{\mu,t}$, involves $E[\|(w-\mu)/\sigma^2\|^2 R(w,z)^2]$. The term $E[\|w-\mu\|^2/\sigma^4]$ contributes a factor scaling as $d/\sigma^2$. Thus, $\text{Var}(\hat{g}\_{\mu,t})$ can scale proportionally to $d/\sigma^2$ multiplied by terms related to $R(w,z)$. This $1/\sigma^2$ dependence shows that as $\sigma \to 0$ (exploitation), $\text{Var}(\hat{g}\_{\mu,t})$ can increase if $R(w,z)$ does not also diminish appropriately as $w \to \mu$.

The score for $\psi=\log\sigma$ is 

$$\nabla_{\psi} \log \pi_\theta(w) = (\|w-\mu\|^2/\sigma^2 - d),$$

where $d$ is the dimension of $w$. The variance of its gradient estimate also depends on $R(w,z)$ and $\sigma$.

Note that an interesting consequence of optimization $L$ via the PGT approach is that $R(w,z)$ can be non-differentiable with respect to $w$. Indeed, we only need to calculate the gradient $\nabla_\theta$ for policy parameters. This flexibility is exchanged for managing the variance of $\hat{g}_t$. Well, that and the $J$ is generally nonconvex in $\sigma$, even if $L$ is convex.

#### Aside: Connection to Gaussian Smoothing
If the policy is $w \sim N(\mu, \sigma_0^2 I_d)$ with fixed $\sigma_0^2$ (so $\theta = \mu$), and $R(w) = -L(w)$, then 

$$J(\mu) = E_{w \sim N(\mu, \sigma_0^2 I_d)}[-L(w)] = -L_{\sigma_0}(\mu),$$

where 

$$L_{\sigma_0}(\mu) = E_{w \sim N(\mu, \sigma_0^2 I_d)}[L(w)]$$

is the *Gaussian-smoothing* of the function $L(\cdot)$ evaluated at $\mu$. The policy gradient $\nabla\_\mu J(\mu)$ is then $-\nabla\_\mu L\_{\sigma_0}(\mu)$. Thus, PGT here performs stochastic gradient descent on a smoothed version of $L$. This links PGT to zeroth-order optimization methods.

### Variance of the gradient, stepsize selection, and gradient clipping

Recall that we wish to maximize $J(\theta)$ via the stochastic gradient ascent update $\theta \leftarrow \theta + \alpha \hat{g}_t$. Two primary considerations for SGA are the variance of $\hat{g}_t$ and the selection of stepsize $\alpha$.

#### Variance Reduction
**Baselines:** In the stateless setting, 

$$V^{\pi_\theta} = E_{w \sim \pi_\theta(w)}[R(w)]$$

is the expected reward under the policy. The advantage is 

$$A^{\pi_\theta}(w') = R(w') - V^{\pi_\theta}.$$

Subtracting a baseline $b_t \approx V^{\pi_\theta}$ from the reward:

$$ \hat{g}_t = \nabla_\theta \log \pi_\theta(w_t) (R(w_t, z_t) - b_t) $$

One way to estimate $V^{\pi\_\theta}$ online is using an exponential moving average of rewards. This provides $b_t$. The centered term $(R(w_t,z_t) - b_t)$ can yield a lower variance $\hat{g}_t$.

**Batch Gradient Estimator:** One can average $\hat{g}\_t$ over a mini-batch of $N\_s$ independent samples $w\_j \sim \pi\_\theta(w)$ (each with its own $R(w\_j)$ or $R(w\_j, z\_j)$). This forms $\bar{g}\_t = \frac{1}{N\_s} \sum\_{j=1}^{N\_s} \hat{g}\_{t,j}$. In this case, 

$$\text{Var}(\bar{g}_t) = \text{Var}(\text{single sample } \hat{g}_t)/N_s.$$

This reduces variance at the cost of $N_s$ reward evaluations per policy update.

#### Stepsize Selection and Convergence

The objective $J(\theta)$ is generally non-convex. For non-convex $J(\theta)$, convergence rate analysis focuses on metrics like $E[\|\nabla J(\theta_k)\|^2] \to 0$ (or to a noise floor for constant $\alpha$). In more restricted settings, for example, if $J(\theta)$ is (locally) strongly convex around an optimum $\theta^\*$, metrics like $E[\|\theta_k-\theta^*\|^2]$ or $J(\theta^\*) - J(\theta_k)$ can be analyzed. The stepsize (or learning rate [^1]) $\alpha$ affects convergence. (if none of this is familiar to you, see my lecture notes on [stochastic gradient descent for mean estimation](https://damek.github.io/STAT-4830/section/6/notes.html))

1.  **Constant Stepsize:** For a constant $\alpha$, $\theta\_k$ oscillates around a region where $\nabla J(\theta) \approx 0$. A convergence metric $M\_k(\alpha)$ (e.g., $E[\|\nabla J(\theta\_k)\|^2]$ for non-convex or $E[\|\theta\_k-\theta^\*\|^2]$ for locally convex) usually scales as:

    $$ M_k(\alpha) \approx \frac{C_0 \cdot (\text{Initial Error})}{\text{poly}(k) \cdot \alpha} + \frac{C_1 \cdot \alpha \cdot \text{Var}(\text{single sample } \hat{g}_t)}{N_s} $$

    where $N_s$ is the batch size for $\hat{g}_t$ ($N_s=1$ if no batching). As $k \to \infty$, the first term (bias reduction) vanishes, leaving the second term (noise floor). A larger $\alpha$ speeds initial progress but gives a higher noise floor.
2.  **Diminishing Stepsize:** For $M_k(\alpha_k) \to 0$, $\alpha_k$ must diminish, for instance, satisfying the Robbins-Monro conditions: $\sum_{k=0}^\infty \alpha_k = \infty$ and $\sum_{k=0}^\infty \alpha_k^2 < \infty$.

There are of course issues with these bounds when we take $\sigma$ to zero, since the variance explodes. To actually achieve convergence, we would need to increase the batch size sufficiently fast. *Or*, we could 

#### Clip the gradients

Gradient clipping replaces $\hat{g}\_t$ with $c\frac{\hat{g}\_t}{\|\|\hat{g}\_t\|\|}$ if $\|\|\hat{g}\_t\|\| > c$ for a user specified constant $c$. Since the gradient of the score function explodes as $\sigma$ tends to zero, this becomes necessary.

### Illustrative Example: Policy Gradient for a Quadratic Loss

We apply these concepts to an agent (everything's an agent now!) learning to sample $w$ to minimize 

$$L(w) = \frac{1}{2}(w - w^\ast)^2$$

for a target $w^\ast$.

A stochastic version of the loss is 

$$\ell(w,z) = \frac{1}{2}(w - w^\ast)^2 + z,$$

where $z \sim N(0, \sigma_z^2)$.

The agent's policy $\pi\_\theta(w)$ is $N(w\|\mu, \sigma^2)$, for scalar $w$ (i.e., $d=1$). The learnable parameters are $\theta = (\mu, \psi)$, where $\psi = \log\sigma$. This parameterization ensures $\sigma = e^\psi > 0$ when $\psi$ is optimized without constraints.

The score functions for this policy (with $d=1$) are:

$$\begin{aligned}
\nabla_\mu \log \pi_\theta(w) &= \frac{w-\mu}{\sigma^2} \\
\nabla_\psi \log \pi_\theta(w) &= \frac{(w-\mu)^2}{\sigma^2} - 1
\end{aligned}$$

*(Continuing from the previous draft)*

### Experiments: Illustrating Reward Design and SGD Effects

We present results from numerical experiments applying the policy gradient approach to the quadratic loss $L(w) = \frac{1}{2}(w - w^\ast)^2$ with target $w^\ast=5.0$. The policy is $\pi\_\theta(w) = N(w\|\mu, \sigma^2)$ parameterized by $\theta = (\mu, \psi)$ where $\psi=\log\sigma$. All experiments start from an initial policy $\mu\_0=0.0$, $\sigma\_0=2.0$ (i.e., $\psi\_0=\log 2$) and use gradient clipping with a maximum norm of 10.0. We examine the five reward formulations (R1-R5) described earlier. For R1 and R2, we investigate the effects of learning rate, baselines, diminishing stepsizes, and batching. For R3, R4, and R5, we show results from a single representative configuration.

#### R1: True Reward

The reward is $R(w) = -L(w) = -\frac{1}{2}(w - w^\ast)^2$. This provides direct, dense feedback based on the true objective. All experiments for R1 run for $10^4$ episodes.

**Learning Rate Comparison (Set A):**
We compare three constant learning rates: $\alpha=0.01$ (High), $\alpha=0.001$ (Medium), $\alpha=0.0001$ (Low). All runs use an EMA baseline and batch size $N_s=1$.

![Figure R1-LR](/assets/figures/R1_True-Learning-Rate-Comparison.png)
*Figure 1: R1 (True Reward) - Effect of Learning Rate. Comparison of policy mean $\mu$ and standard deviation $\sigma$ convergence for different constant learning rates ($\alpha$). All runs use EMA baseline, $N_s=1$, $10^4$ episodes.*

*Description:* Figure 1 displays the evolution of $\mu$ and $\sigma$. The highest learning rate ($\alpha=0.01$) converges $\mu$ rapidly to $w^\ast=5.0$, reaching it within about 2000 episodes. The final $\sigma$ approaches approximately $0.07$. This run exhibits visible oscillation in $\mu$ around $w^\ast$, indicating a noise floor. The medium learning rate ($\alpha=0.001$) converges $\mu$ slower, reaching $w^\ast$ near episode 8000, with $\sigma$ converging towards $0.24$. The oscillations are less pronounced than with $\alpha=0.01$. The lowest learning rate ($\alpha=0.0001$) shows very slow progress; after $10^4$ episodes, $\mu$ is only around $2.3$ and $\sigma$ is still above 1.1. These results illustrate the trade-off with constant stepsizes: faster convergence with larger $\alpha$ comes at the cost of a higher asymptotic noise floor.

**Baseline Comparison (Set B):**
We compare using an EMA baseline versus no baseline ($b_t=0$) for the medium learning rate ($\alpha=0.001$), with $N_s=1$.

![Figure R1-Baseline](/assets/figures/R1_True-Baseline-Comparison.png)
*Figure 2: R1 (True Reward) - Effect of Baseline. Comparison of runs with EMA baseline vs. no baseline ($b_t=0$). Both use $\alpha=0.001$, $N_s=1$, $10^4$ episodes.*

Description:* Figure 2 compares learning with and without an EMA baseline. In this deterministic reward setting, both runs converge $\mu$ towards $w^\ast=5.0$. The run without a baseline shows significant initial instability in $\sigma$, which increases substantially before decreasing. The final $\sigma$ without baseline is approximately $0.31$, compared to $0.24$ with the EMA baseline. The EMA baseline run shows a smoother, monotonic decrease in $\sigma$. This suggests the baseline helps stabilize learning, particularly for the policy variance parameter $\sigma$, even when rewards are deterministic.

**Stepsize Schedule Comparison (Set C):**
We compare a diminishing learning rate schedule ($\alpha_k = 0.01 / (1 + k/K_{decay})$ with $K_{decay}=2000$) against a constant learning rate $\alpha=0.01$. Both use EMA baseline, $N_s=1$.


![Figure R1-Stepsize](/assets/figures/R1_True-Stepsize-Schedule-Comparison.png)
*Figure 3: R1 (True Reward) - Stepsize Schedule Comparison. Compares constant $\alpha=0.01$ vs. a diminishing schedule starting at $\alpha_0=0.01$. Both use EMA baseline, $N_s=1$, $10^4$ episodes.*

*Description:* Figure 3 compares constant versus diminishing stepsizes. Both runs converge $\mu$ quickly to $w^\ast=5.0$. The constant stepsize ($\alpha=0.01$) run shows persistent oscillations in $\mu$ around $w^\ast$, indicative of the noise floor, with $\sigma$ settling near $0.07$. The diminishing stepsize run also converges quickly initially. As $\alpha_k$ decreases, the oscillations in $\mu$ are visibly reduced, and $\sigma$ continues to decrease, reaching approximately $0.12$ by the end and appearing to still trend downwards. This illustrates that a diminishing stepsize schedule satisfying the Robbins-Monro conditions can mitigate the noise floor and allow parameters to converge more precisely to the optimal values.

**Batch Gradient Estimator Comparison (Set D):**
We compare using a single sample ($N_s=1$) versus a batch of $N_s=10$ samples to estimate the gradient. Both use the medium learning rate ($\alpha=0.001$) and EMA baseline.

![Figure R1-Batching](/assets/figures/R1_True-Batch-Gradient-Estimator-Comparison.png)
*Figure 4: R1 (True Reward) - Batch Gradient Comparison. Compares gradient estimates using $N_s=1$ vs. $N_s=10$ samples per update. Both use $\alpha=0.001$, EMA baseline, $10^4$ episodes.*

*Description:* Figure 4 shows the effect of mini-batching. The run using $N_s=10$ samples per gradient estimate exhibits smoother trajectories for both $\mu$ and $\sigma$ compared to the $N_s=1$ run. The final $\sigma$ is slightly lower for $N_s=10$ ($\approx 0.22$) compared to $N_s=1$ ($\approx 0.24$). This smoothing effect is due to the reduction in the variance of the gradient estimate $\bar{g}_t$ when averaged over more samples ($\text{Var}(\bar{g}_t) = \text{Var}(\hat{g}_t)/N_s$). While requiring more computation per update, batching provides a more reliable gradient estimate.

#### R2: Randomized Reward

The reward is $R(w,z) = -\ell(w,z) = -(\frac{1}{2}(w - (w^\ast+z))^2)$, where $z \sim N(0, 1)$. This reward is an unbiased estimate of $-L(w)$ plus a noise term. All experiments for R2 run for $10^4$ episodes.

**Learning Rate Comparison (Set A):**
Comparison of constant LRs: $\alpha=0.01, 0.001, 0.0001$. All use EMA baseline, $N_s=1$.

![Figure R2-LR](/assets/figures/R2_Randomized-Learning-Rate-Comparison.png)
*Figure 5: R2 (Randomized Reward) - Effect of Learning Rate. Comparison for different constant learning rates ($\alpha$). All use EMA baseline, $N_s=1$, $10^4$ episodes.*

*Description:* Figure 5 shows the learning rate effect with stochastic rewards. The noise in $R(w,z)$ significantly impacts performance. The high learning rate ($\alpha=0.01$) converges $\mu$ to the vicinity of $w^\ast=5.0$ quickly but results in large, persistent oscillations (high noise floor). The final $\mu$ is $4.86$ and $\sigma$ is $0.33$. The medium LR ($\alpha=0.001$) converges $\mu$ more slowly but with much smaller oscillations, reaching $\mu \approx 4.94$ and $\sigma \approx 0.33$. The low LR ($\alpha=0.0001$) converges very slowly, with $\mu \approx 2.2$ after $10^4$ episodes. The noise floor effect, scaling with $\alpha \cdot \text{Var}(\hat{g}\_t)$, is clearly exacerbated by the stochasticity of the reward compared to R1.

**Baseline Comparison (Set B):**
Comparison of EMA baseline vs. no baseline ($b_t=0$) for medium LR ($\alpha=0.001$), $N_s=1$.


![Figure R2-Baseline](/assets/figures/R2_Randomized-Baseline-Comparison.png)
*Figure 6: R2 (Randomized Reward) - Effect of Baseline. Comparison of runs with EMA baseline vs. no baseline ($b_t=0$). Both use $\alpha=0.001$, $N_s=1$, $10^4$ episodes.*

*Description:* Figure 6 highlights the importance of the baseline for stochastic rewards. The run using the EMA baseline shows relatively stable convergence of $\mu$ towards $w^\ast=5.0$ (final $\mu \approx 4.94$) and $\sigma$ decreasing to about $0.33$. The run without a baseline exhibits significant instability. $\sigma$ increases dramatically initially and remains high and erratic. $\mu$ also fails to converge stably (final $\mu \approx 5.04$, but highly variable). The baseline effectively centers the noisy rewards $R(w,z)$, providing a more stable advantage signal $(R(w,z)-b_t)$ and enabling effective learning.

**Stepsize Schedule Comparison (Set C):**
Comparison of diminishing $\alpha_k$ vs. constant $\alpha=0.01$. Both use EMA baseline, $N_s=1$.


![Figure R2-Stepsize](/assets/figures/R2_Randomized-Stepsize-Schedule-Comparison.png)
*Figure 7: R2 (Randomized Reward) - Stepsize Schedule Comparison. Compares constant $\alpha=0.01$ vs. a diminishing schedule starting at $\alpha_0=0.01$. Both use EMA baseline, $N_s=1$, $10^4$ episodes.*

*Description:* Figure 7 compares stepsize schedules with noisy rewards. The constant LR ($\alpha=0.01$) run converges $\mu$ to the vicinity of $w^\ast$ but maintains substantial oscillations due to the noise floor (final $\mu \approx 4.86$, $\sigma \approx 0.33$). The diminishing LR run also converges $\mu$ quickly initially. As $\alpha_k$ decays, the oscillations in $\mu$ decrease significantly, allowing $\mu$ to settle nearer to $w^\ast$ (final $\mu \approx 5.07$). $\sigma$ also decreases more effectively under the diminishing schedule (final $\sigma \approx 0.23$). This demonstrates that diminishing stepsizes are required to mitigate the noise floor and achieve more precise convergence with stochastic rewards.

**Batch Gradient Estimator Comparison (Set D):**
Comparison of $N_s=1$ vs. $N_s=10$. Both use medium LR ($\alpha=0.001$) and EMA baseline.


![Figure R2-Batching](/assets/figures/R2_Randomized-Batch-Gradient-Estimator-Comparison.png)
*Figure 8: R2 (Randomized Reward) - Batch Gradient Comparison. Compares gradient estimates using $N_s=1$ vs. $N_s=10$ samples per update. Both use $\alpha=0.001$, EMA baseline, $10^4$ episodes.*

*Description:* Figure 8 shows the impact of batching with stochastic rewards. Using $N_s=10$ yields significantly smoother learning curves for both $\mu$ and $\sigma$ compared to $N_s=1$. The final parameters are $\mu \approx 5.03, \sigma \approx 0.23$ for $N_s=10$, compared to $\mu \approx 4.94, \sigma \approx 0.33$ for $N_s=1$. Reducing gradient variance through batching leads to more stable updates and potentially better convergence within the same number of *episodes* (parameter updates), albeit at higher computational cost per episode.

#### R3: Hacking Reward (Fixed Batch Proxy)

The reward is $R(w) = -\frac{1}{10} \sum_{z_i \in S_{train}} \frac{1}{2}(w - (w^\ast + z_i))^2$, where $S_{train}$ is a fixed batch of 10 noise samples $z_i \sim N(0,1)$. We run a single configuration: High STD LR ($\alpha=0.01$), EMA baseline, $N_s=1$, Clip=10, for $10^4$ episodes. The plot target is the true $w^\ast=5.0$.


![Figure R3-Single](/assets/figures/R3_Hacking-Single-Run.png)
*Figure 9: R3 (Hacking Reward) - Single Run. Agent optimizes reward based on a fixed batch $S_{train}$ of 10 noise samples. Uses $\alpha=0.01$, EMA baseline, $N_s=1$, $10^4$ episodes. Target line is true $w^\ast=5.0$.*

*Description:* Figure 9 illustrates reward hacking. The agent optimizes the reward function defined by the fixed batch $S_{train}$. The policy mean $\mu$ converges rapidly and stably (final $\mu \approx 5.45$), and $\sigma$ decreases effectively (final $\sigma \approx 0.07$). However, the converged value of $\mu$ is significantly different from the true optimum $w^\ast=5.0$. This occurs because the minimum of the sample-average loss (defined by $S_{train}$) is likely shifted from $w^\ast$. The agent has successfully optimized the proxy reward provided, demonstrating reward hacking where the solution converges precisely to the minimizer of the available, potentially flawed, objective.

#### R4: Discrete Sparse Reward

The reward is $R(w) = 1$ if $(w-w^\ast)^2 < 0.25$, else $0$. We run a single configuration: High STD LR ($\alpha=0.01$), EMA baseline, $N_s=1$, Clip=10, for $5 \times 10^4$ episodes (Note: plot x-axis).


![Figure R4-Single](/assets/figures/R4_DiscreteSparse-Single-Run.png)
*Figure 10: R4 (Discrete Sparse Reward) - Single Run. Reward is 1 if $|w-w^\ast|<0.5$, else 0. Uses $\alpha=0.01$, EMA baseline, $N_s=1$, $5 \times 10^4$ episodes.*

*Description:* Figure 10 shows learning with a discrete, sparse reward. The agent initially makes slow progress, as samples $w$ from the initial policy $N(0, 2^2)$ rarely fall within the rewarding region $(4.5, 5.5)$. $\mu$ only begins moving significantly towards $w^\ast=5.0$ around episode 15,000. During this initial phase, $\sigma$ increases, likely promoting exploration. Once rewards become more frequent as $\mu$ nears $w^\ast$, $\sigma$ begins to decrease, and $\mu$ converges close to $w^\ast$ (final $\mu \approx 5.01$, $\sigma \approx 0.11$). The baseline value remains near zero until rewards become consistent, after which it rises towards the average reward (which is less than 1, as not all samples land in the region). This illustrates the exploration challenge posed by sparse rewards.

#### R5: Sparse Continuous Reward (Shifted)

The reward is $R(w) = \max(0, -0.5(w-w^\ast)^2 + 0.125)$, positive only if $\|w-w^\ast\|<0.5$. We run a single configuration: VSparse LR ($\alpha=0.1$), EMA baseline, $N_s=1$, Clip=10, for $5 \times 10^4$ episodes (Note: plot x-axis).


![Figure R5-Single](/assets/figures/R5_SparseContinuous-Single-Run.png)
*Figure 11: R5 (Sparse Continuous Reward, Shifted) - Single Run. Reward is $\max(0, -0.5(w-w^\ast)^2 + 0.125)$. Uses $\alpha=0.1$, EMA baseline, $N_s=1$, $5 \times 10^4$ episodes.*

*Description:* Figure 11 shows learning with the sparse continuous (and shifted non-negative) reward. Similar to R4, learning is initially slow due to sparsity, with $\mu$ starting to move towards $w^\ast$ around episode 20,000. $\sigma$ again increases initially to explore. Once the rewarding region is found, the shaped reward provides magnitude information. Combined with the high learning rate ($\alpha=0.1$), this leads to rapid convergence of $\mu$ to $w^\ast=5.0$ (final $\mu \approx 5.00$) and a very effective decrease in $\sigma$ (final $\sigma \approx 0.01$). The baseline tracks the average positive reward obtained within the region. This suggests that shaped rewards, even if sparse, can lead to precise convergence once discovered, and that higher learning rates may be effective in exploiting the signal within the sparse region for this reward type.


[1] I think I've completely given up and no longer care whether I say stepsize or learning rate. 