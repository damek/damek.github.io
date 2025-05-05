---
title: "the ladder mechanism for ml competitions"
date: 2025-05-04
description: "how to adapt models to the test set without exponentially inflating generalization bounds"
---

## tldr

in the worst case, adapting models based on test set performance [exponentially inflates generalization bounds](https://x.com/damekdavis/status/1918019744348639453); the [ladder mechanism](https://arxiv.org/pdf/1502.04585) shows how to avoid this if we precommit to an adaptation rule. 

## the problem
machine learning competitions use leaderboards to rank participants. participants submit models $f_1, \dots, f_k$ iteratively. they receive scores $R_1, \dots, R_k$ based on performance on a held-out test set $S$. participants use this feedback to create subsequent submissions $f_t$. this interaction creates an adaptive data analysis scenario where the analyst's choices depend on information derived from the test set $S$. this adaptivity poses a challenge: standard statistical guarantees about model performance can fail. the empirical loss $R_S(f_t)$ computed on $S$ might not accurately reflect the true generalization loss $R_D(f_t)$, because $f_t$'s dependence on $S$ through past scores means $f_t$ is not independent of $S$. participants might overfit the holdout set, leading to unreliable leaderboards. this work investigates this problem and introduces the ladder mechanism to maintain leaderboard reliability under such adaptive analysis.

### formalizing the problem and goal

#### the adaptive setting

the setup involves a holdout set $S = \{(x\_i, y\_i)\}\_{i=1}^n$ drawn i.i.d. from a distribution $D$. we compute empirical loss $R\_S(f) = \frac{1}{n} \sum_{i=1}^n l(f(x\_i), y\_i)$ and aim to understand the true loss $R\_D(f) = \mathbb{E}\_{(x,y) \sim D}[l(f(x), y)]$. the interaction proceeds sequentially: an analyst strategy $A$ generates $f\_t$ based on history $(f\_1, R\_1, \dots, f\_{t-1}, R\_{t-1})$, and a leaderboard mechanism $L$ computes score $R\_t$ using $f\_t$, the history, and the sample $S$. the core difficulty is that $f\_t$'s dependence on $S$ (via $R\_{<t}$) breaks the independence assumption needed for standard statistical bounds.

#### fixing the protocol

to analyze this adaptive process, the framework assumes the interaction protocol $(A, L)$ is fixed before the specific sample $S$ is drawn. the protocol includes the analyst's deterministic algorithm $A$ for choosing $f_t$ and the host's mechanism $L$ for computing $R_t$. this fixed protocol $(A, L)$ defines a conceptual tree $T$ representing all possible interaction histories. the set $F$ comprises all classifiers $f_t$ appearing at any node in this tree $T$. importantly, $F$ is determined by the protocol $(A, L, k)$ and is fixed *before* the specific sample $S$ is used for evaluation.

#### leaderboard accuracy

the objective shifts from accurately estimating each $R_D(f_t)$ to ensuring the reported score $R_t$ accurately reflects the best true performance achieved up to step $t$. this is measured by the leaderboard error:

$$
\text{lberr}(R_1, \dots, R_k) \overset{\text{def}}{=} \max_{1 \le t \le k} \left| \min_{1 \le i \le t} R_D(f_i) - R_t \right|
$$

the aim is to design a mechanism $L$ that minimizes this error against any analyst strategy $A$.

### the ladder mechanism

the ladder mechanism is proposed as a simple algorithm for $L$. it controls the flow of information to the analyst by using a pre-defined step size $\eta > 0$.

the algorithm proceeds as follows:
1.  initialize the best score $R_0 = \infty$.
2.  for each round $t = 1, \dots, k$:
    *   receive classifier $f\_t$ from the analyst $A$.
    *   compute the empirical loss $R\_S(f\_t)$ using the holdout set $S$.
    *   apply the decision rule: if $R\_S(f\_t) < R\_{t-1} - \eta$, then update the reported score $R\_t \leftarrow [R\_S(f\_t)]_\eta$. otherwise, maintain the previous score $R\_t \leftarrow R\_{t-1}$. (here, $[x]\_\eta$ rounds $x$ to the nearest multiple of $\eta$.)
    *   report the score $R\_t$ back to the analyst $A$.

the intuition behind the ladder is that it only reveals progress, i.e., a new, lower score, if the observed empirical improvement surpasses the threshold $\eta$. this quantization prevents the analyst from reacting to small fluctuations in $R\_S(f)$ that might be specific to the sample $S$, thus limiting the leakage of information about $S$.

### theoretical upper bound

the ladder mechanism comes with a provable guarantee on its leaderboard accuracy.

**theorem 3.1:** for any deterministic analyst strategy $A$, the ladder mechanism $L$ using a step size $\eta = C \left( \frac{\log(kn)}{n} \right)^{1/3}$ (for a constant $C$) ensures that, with high probability over the draw of sample $S$ (of size $n$), the leaderboard error is bounded:

$$
\text{lberr}(R_1, \dots, R_k) \le O\left( \left( \frac{\log(kn)}{n} \right)^{1/3} \right)
$$

this result is significant because the error depends only logarithmically on the number of submissions $k$. this implies robustness against prolonged adaptive interaction, contrasting with naive methods where error might grow polynomially with $k$.

### proof argument: uniform convergence over the function set $F$

the proof establishes that the empirical loss $R\_S(f)$ converges uniformly to the true loss $R\_D(f)$ over the entire set $F$ of functions potentially generated by the fixed protocol $(A, L)$.

-  *bounding $\|F\|$ via compression (claim 3.2):* the argument first bounds the size of $F$. the limited information release by the ladder mechanism restricts the complexity of the interaction tree $T$. any node (representing a history and function $f_t$) in $T$ can be uniquely identified using a limited number of bits, $B$. this implies $\|F\| \le \|T\| \le 2^B$.
    *   *encoding scheme:* to identify a node, we encode its path from the root. a path is characterized by the sequence of score updates. an "update step" at index $j$ occurs if $R\_j < R\_{j-1} - \eta$. the number of such steps along any path is at most $N\_d \approx 1/\eta$. the number of possible score values $R\_j$ at an update step is also bounded by $N\_v \approx 1/\eta$. encoding the indices and values of these update steps requires a certain number of bits. claim 3.2 shows that $B = (1/\eta + 2)\log(4k/\eta)$ bits suffice.
    *   *bound:* this yields $\|F\| \le 2^B$, where $B = O(\frac{1}{\eta}\log(k/\eta))$.

-  *uniform convergence argument:* since $F$ is fixed before $S$ is observed, we can apply a uniform convergence argument using a union bound.
    *   the probability that *any* function in $F$ has a large deviation is bounded:

        $$ \mathbb{P}\left\{ \exists f \in F : |R_S(f) - R_D(f)| > \epsilon \right\} \le \sum_{f \in F} \mathbb{P}\left\{ |R_S(f) - R_D(f)| > \epsilon \right\} $$
        
    *   for each *fixed* $f \in F$, hoeffding's inequality gives

        $$
        \mathbb{P}\{|R_S(f) - R_D(f)| > \epsilon\} \le 2e^{-2\epsilon^2 n}.
        $$

    *   combining these yields the overall failure probability bound:

        $$ \mathbb{P}\{\text{large deviation in } F\} \le |F| \cdot 2e^{-2\epsilon^2 n} \le 2 \cdot 2^B e^{-2\epsilon^2 n} $$

- *balancing error terms:* we need this probability to be small. this requires the exponent to be sufficiently negative, essentially $2\epsilon^2 n \gtrsim B \approx \frac{1}{\eta}\log(k/\eta)$. the total leaderboard error comprises statistical error $\epsilon$ and mechanism threshold error $\eta$, totaling approximately $\epsilon + \eta$. to minimize this sum subject to the constraint $2\epsilon^2 n \approx \frac{1}{\eta}\log(k/\eta)$, we balance the terms, leading to $\epsilon \approx \eta$. substituting back into the constraint yields $2\eta^3 n \approx \log(k/\eta)$. this resolves to $\eta \approx \epsilon \approx \left( \frac{\log(k/\eta)}{n} \right)^{1/3}$. choosing $\eta = O\left( \left( \frac{\log(kn)}{n} \right)^{1/3} \right)$ makes the failure probability small.

-  *implication for leaderboard error:* the uniform convergence ensures that, with high probability, $\|R\_S(f) - R\_D(f)\| \le \epsilon$ holds simultaneously for all $f \in F$. since the actual sequence $f\_1, \dots, f\_k$ must belong to $F$, this guarantee applies to them. this property, combined with the ladder's update rule (which relates $R\_t$ to $R\_S(f)$ for some $f$ that triggered an update), allows proving that the reported score $R\_t$ stays within $\approx \epsilon + \eta$ of the true best score $\min\_{i \le t} R\_D(f\_i)$. this bounds the leaderboard error as stated in the theorem.

### theoretical lower bound

the paper also establishes a fundamental limit on the accuracy achievable by *any* leaderboard mechanism.

**theorem 3.3:** for any mechanism $L$, there exist scenarios (distributions $D$, classifiers $f_i$) where the expected worst-case leaderboard error is bounded below:

$$
\inf_L \sup_D \mathbb{E}_S[\text{lberr}(R(S))] \ge \Omega\left( \sqrt{\frac{\log k}{n}} \right)
$$

this lower bound highlights a gap between the ladder's $n^{-1/3}$ performance and the optimal $n^{-1/2}$ dependence. the proof utilizes a reduction to high-dimensional mean estimation combined with fano's inequality.

### practical stuff

#### parameter-free ladder mechanism

a practical challenge is setting the step size $\eta$ without prior knowledge of $k$ and $n$. a parameter-free variant addresses this by adapting the threshold dynamically.
*   it maintains the best score $R_{t-1}$ and the loss vector $l_{prev}$ of the submission that achieved it.
*   for a new submission $f_t$ with loss vector $l_t$, it computes the sample standard deviation $s = \text{std}(l_t - l_{prev})$.
*   it uses an adaptive threshold $\eta_t = s/\sqrt{n}$.
*   the update rule becomes: update $R\_t \leftarrow [R\_S(f\_t)]_{1/n}$ if $R\_S(f\_t) < R\_{t-1} - \eta\_t$. otherwise, $R\_t \leftarrow R\_{t-1}$.
this approach uses a heuristic related to a paired t-test to gauge statistical significance based on observed variance.

#### the boosting attack

this attack demonstrates the weakness of leaderboard mechanisms that reveal empirical scores with high precision.
*   it targets mechanisms where the reported score $R\_t$ is very close to $R\_S(f\_t)$.
*   the attack proceeds by submitting $k$ random classifiers $u\_i$ and observing their scores $l\_i \approx R\_S(u\_i)$. it identifies a subset $I$ of classifiers that performed slightly better than chance on $S$ (e.g., $l\_i \le 1/2$). a final classifier $u^*$ is formed by taking a majority vote over this subset $\{u\_i : i \in I\}$.
*   if the score precision is high (error significantly less than $1/\sqrt{n}$), the attack can yield $u^*$ with $R\_S(u^\* ) \ll 1/2$ but $R\_D(u^\* ) \approx 1/2$. this creates a large leaderboard error, $\Omega(\sqrt{k/n})$. the ladder mechanism resists this attack because its threshold $\eta \approx n^{-1/3}$ is much coarser than the $n^{-1/2}$ precision needed for the attack to reliably identify the slightly advantageous random classifiers. it simply doesn't provide precise enough feedback.