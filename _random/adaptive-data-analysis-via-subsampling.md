---
title: "Adaptive data analysis via subsampling"
date: 2025-05-05
tags: [adaptive-data-analysis]
description: "A general recipe for building estimators based on adaptive queries"
---

## TLDR

Our previous post discussed the [ladder mechanism](/random/the-ladder-mechanism-for-ml-competitions), which allows adapting models based on test set accuracy by quantizing feedback. Here, we explore an alternative: ensuring statistical validity for *general adaptive queries* by relying on subsampling and bounded query outputs, requiring no explicit mechanism.

## Recap: adapting models with the ladder mechanism

In the [post on the ladder mechanism](/random/the-ladder-mechanism-for-ml-competitions), we saw how an analyst could iteratively submit models $f_1, \dots, f_k$ for evaluation on a holdout set $S$. The mechanism worked like this:
1.  analyst submits $f_t$.
2.  host computes loss $R_S(f_t)$ on the *full* dataset $S$.
3.  host compares $R_S(f_t)$ to the previously best reported score $R_{t-1}$.
4.  host reports $R_t$: either $R_{t-1}$ (no significant improvement) or a new score $[R_S(f_t)]_\eta$ (quantized improvement), based on a threshold $\eta$.

The guarantee was that the best reported score $R_t$ reliably tracks the true best performance $\min_{i \le t} R_D(f_i)$ (low leaderboard error), even for a very large number of submissions $k$. This provides a safe way to adapt model choices based on leaderboard feedback.

## Beyond leaderboards: handling arbitrary adaptive queries

The ladder mechanism focuses specifically on tracking the minimum loss value. What if an analyst wants to ask more general questions about the data adaptively? For instance, calculating various statistics, exploring correlations, or testing different hypotheses sequentially, where the choice of the next question depends on previous answers? Can we ensure these answers remain representative of the underlying distribution $D$ without introducing bias from the adaptive interaction?

## Subsampling + bounded output suffices

["Subsampling suffices for adaptive data analysis"](https://arxiv.org/abs/2302.08661) (by [Guy Blanc](https://web.stanford.edu/~gblanc/)) shows that this is possible *without* an explicit mechanism like the ladder, provided the analyst's queries $\varphi_t$ naturally adhere to two conditions:

1.  **Subsampling input:** each query $\varphi_t$ operates not on the full dataset $S$, but on a random subsample $S' \subset S$ of size $w_t$, drawn uniformly without replacement ($w_t$ should be significantly smaller than $n/2$).
2.  **Bounded output:** each query $\varphi\_t$ outputs a value from a finite set $Y\_t$ of limited size $r\_t = \|Y\_t\|$.

The insight is that the combination of noise inherent in using a small random subsample $S'$ and the information coarsening from having only $r_t$ possible outputs inherently limits how much the answer $y_t = \varphi_t(S')$ can reveal about the specific dataset $S$.

## The interaction model

This approach assumes the following interaction flow:

1.  **Pre-interaction phase:**
    *   analyst possesses prior knowledge and potentially a training dataset $S_{train}$ ($\text{state}_0$).

2.  **Interactive phase (queries $\varphi_t$ on holdout $S$):**
    *   for $t=1$ to $T$:
        *   analyst, based on $\text{state}_0$ and responses $y\_{<t}$, chooses a query function $\varphi\_t: X^{w\_t} \to Y\_t$.
        *   a random subsample $S'$ of size $w\_t$ is drawn from $S$.
        *   response $y\_t = \varphi\_t(S')$ is computed. *(we use $y\_t \sim \varphi\_t(S)$ to denote this process)*.
        *   analyst receives $y\_t$.
    *   the total number of queries $T$ is limited polynomially by the cumulative "cost".

3.  **Post-interaction phase (choosing test query $\psi$):**
    *   analyst now has $\text{state}_0$ and the transcript $y = (y\_1, \dots, y\_T)$.
    *   based *only* on this information, analyst selects a final set $\Psi$ of $m = \|\Psi\|$ test queries, $\psi: X^w \to [0,1]$.
    *   the number $m$ can be exponentially large.

The goal is to guarantee that these adaptively chosen test queries $\psi$ generalize.

## Main results: bounding information leakage and bias

The theory quantifies the success of this approach by first bounding the information leakage during the interactive phase.

**Theorem 2 (subsampling queries reveal little information):** Let $S \sim D^n$. Let $y = (y\_1, \dots, y\_T)$ be the responses from an adaptive sequence of $T$ subsampling queries, where query $\varphi\_t$ uses subsample size $w\_t$ and range $Y\_t$ with $\|Y\_t\|=r\_t$. The mutual information between the dataset and the responses is bounded by:

$$ I(S; y) \le \frac{4 E[\sum_{t=1}^T w_t(r_t - 1)]}{n} $$

where the expectation $E[\cdot]$ is over the analyst's adaptive choices of $\varphi_t$. The term $w_t(r_t-1)$ represents the "information cost" of query $t$.

This low mutual information translates into guarantees about the generalization error of the final test queries $\psi$. We define the error for a test query $\psi: X^w \to [0,1]$ as:

$$ \text{error}(\psi, S, D) := \frac{1}{w} \min\left(\Delta, \frac{\Delta^2}{\text{Var}_{T \sim D^w}[\psi(T)]}\right) \quad \text{where} \quad \Delta := |\mu_\psi(S) - \mu_\psi(D)| $$

Here $\mu\_\psi(S) = E\_{T \sim S^{(w)}}[\psi(T)]$ is the empirical mean on $S$ (average over all size-$w$ subsamples without replacement) and $\mu\_\psi(D) = E\_{T \sim D^w}[\psi(T)]$ is the true mean.

**Theorem 9 (generalization bound - expected error):** In the interaction described above, let the analyst choose a set $\Psi$ of $m = \|\Psi\|$ test queries based on the responses $y$. The expected maximum error over these test queries is bounded:

$$ E_{S, \text{analyst}}\left[\sup_{\psi \in \Psi} \text{error}(\psi, S, D)\right] \le O\left(\frac{E[\sum_{t=1}^T w_t|Y_t|] + \ln m}{n^2} + \frac{\ln m}{n}\right) $$

**Theorem 10 (generalization bound - high probability for $w=1$):** If all interactive queries use subsample size $w\_t=1$ and the total output complexity $\sum\_{t=1}^T \|Y\_t\| \le b$ is bounded, then for any failure probability $\delta > 0$,

$$ \Pr\left[\sup_{\psi \in \Psi} \text{error}(\psi, S, D) \ge O\left(\ln(m/\delta)\left(\frac{b}{n^2} + \frac{1}{n}\right)\right)\right] \le \delta $$

These bounds show that the error is small if $n$ is large relative to the cumulative "cost" of the interactive queries and the complexity (number or log-number) of final test queries.

## Proof sketch: stability -> information -> bias

The proof rigorously connects subsampling to low bias via algorithmic stability and mutual information.

1.  **ALMOKL stability:** the core concept is *average leave-many-out kl (ALMOKL) stability*. an algorithm $M$ (representing $\varphi_t(S)$) is $(m, \epsilon)$-ALMOKL stable if removing $m$ random points from $S$ changes its output distribution by at most $\epsilon$ in average kl divergence, compared to a simulator $M'$ running on the smaller dataset $S_J$.

    $$ E_{J \sim \binom{[n]}{n-m}} [d_{KL}(M(S) || M'(S_J))] \le \epsilon \quad (\text{definition 5.1}) $$

    The simulator $M'$ needs careful construction to handle potential support mismatches. The paper uses:

    $$ M'_{\varphi}(S_J) := \begin{cases} \text{Unif}(Y) & \text{wp } \alpha = \frac{1}{|Y| + (n-m)/w} \\ \varphi(S_J) & \text{wp } 1 - \alpha \end{cases} \quad (\text{eq. 8}) $$

2.  **Subsampling implies stability (lemma 6.1):** this is a key technical lemma. It shows that a query $\varphi: X^w \to Y$ processed via subsampling without replacement from $S$ is $(m, \epsilon)$-ALMOKL stable for:

    $$ \epsilon \le \frac{w(|Y|-1)}{n-m+1} $$

    *Proof idea:* The proof involves comparing sampling without replacement (for $\varphi(S)$ and $\varphi(S_J)$) to sampling *with* replacement. This transition is non-trivial and uses theorem 6, a generalization of hoeffding's reduction theorem applied to u-statistics and convex functions. Once reduced to the with-replacement setting, the kl divergence can be bounded using properties of sums of independent variables (related to binomial distributions) and bounds on inverse moments (fact 6.2). The specific choice of $\alpha$ in $M'$ simplifies this final step.

3.  **Stability implies low mutual information (theorem 12):** this step connects the per-query stability to the total information leakage across all $T$ adaptive queries. it uses an amortized analysis based on m-conditional entropy $H_m(S) = E\_J[H(S\_J \| S\_{-J})]$.
    *   **lemma 5.3:** shows the expected drop in $H_m(S)$ after observing $y\_t$ is bounded by $\text{stab}\_m(\varphi\_t)$, crucially holding even when $S$ is conditioned on $y\_{<t}$ and is no longer product.
    *   **lemma 5.4:** relates the total drop in $H_m(S)$ to the mutual information $I(S; y)$, provided the initial $S$ was product.
    *   combining these yields $I(S; y) \le \frac{n}{m} \sum E[\text{stab}_m(\varphi_t)]$. using $m=n/2$ and Lemma 6.1 gives Theorem 2.

4.  **Low mutual information implies low bias (theorem 15):** this uses established information-theoretic arguments. If $I(S; y)$ is small, then any test query $\psi$ chosen based *only* on $y$ cannot be too statistically dependent on the specifics of $S$. theorem 15 gives a quantitative bound:

    $$ E\left[\sup_{\psi \in \Psi(y)} \text{error}(\psi, S, D)\right] \le O\left(\frac{I(S; y) + E_y[\ln|\Psi(y)|]}{n}\right) $$

    Combining this with the bound on $I(S;y)$ from theorem 2/12 yields theorem 9. Theorem 10 requires an additional boosting argument (section 8 of the paper) leveraging a generalized direct product theorem.

## Application: statistical query (SQ) mechanism

Blanc's framework yields a simple mechanism for answering [statistical queries](https://dl.acm.org/doi/10.1145/293347.293351). A statistical query is defined by a function $\varphi: X \to [0,1]$, and the goal is to estimate its true mean $\mu\_\varphi(D) = E\_{x \sim D}[\varphi(x)]$. Suppose an analyst makes $T$ adaptive choices of such SQ functions $\varphi\_1, \dots, \varphi\_T$.

*   **Mechanism:** for each submitted sq $\varphi_t$:
    1.  draw $k$ points $x\_1, \dots, x\_k$ uniformly i.i.d. from the dataset $S$.
    2.  for each point $x\_i$, sample a binary vote $v\_i \sim \text{bernoulli}(\varphi\_t(x\_i))$.
    3.  output the average vote $y\_t = (\sum\_{i=1}^k v\_i) / k$ as the answer to $\varphi\_t$.

*   **Analysis:** consider the process of generating a single vote $v\_i$. this involves sampling one point $x\_i$ from $S$ (subsample size $w=1$) and producing a binary output $v\_i \in \{0,1\}$ (range size $r=2$). this fits the "subsampling suffices" framework with $w=1, r=2$. the mechanism essentially performs $k$ such base queries to answer one sq $\varphi\_t$. since $w=1$, the high-probability bound (theorem 10) is applicable.

*   **Guarantee (Theorem 3):** let the analyst adaptively choose $T$ sq functions $\varphi\_1, \dots, \varphi\_T$. if the mechanism uses $k = O(\ln(T/\delta)/\tau^2)$ votes per query, and the dataset size $n$ satisfies the conditions implied by theorem 10 (specifically $n \ge O(\sqrt{T \cdot k \cdot \ln(1/\delta)} / \tau)$ roughly), then with probability at least $1-\delta$, all answers satisfy:

    $$ |y_t - \mu_{\varphi_t}(D)| \le \max(\tau \cdot \text{std}_{\varphi_t}(D), \tau^2) $$

where $\text{std}\_{\varphi\_t}(D) = \sqrt{\text{Var}\_{x \sim D}[\varphi\_t(x)]}$. this mechanism runs in time $O(k)$ per query $\varphi\_t$, which is sublinear in $n$ if $k \ll n$.

## The application to leaderboards is not so clear

The [subsampling suffices" framework](https://arxiv.org/abs/2302.08661) relies on queries using small random subsamples ($w\_t \ll n/2$) and having bounded outputs ($\|Y\_t\|=r\_t$). Could we design a leaderboard mechanism based on this principle, perhaps yielding results comparable to the original ladder mechanism? Consider a mechanism where the loss $L\_t = R\_{S'}(f\_t)$ is computed on a subsample $S'$ of size $w\_t$, quantized to $r\_t$ levels to produce $y\_t$, and $y\_t$ possibly informs a displayed score $R\_t$. There are couple of issues you run into when you try to apply the analysis from [Blanc's paper](https://arxiv.org/abs/2302.08661) to this set up.

First, the number of adaptive queries is more limited. Blanc's guarantees require the total information cost, bounded by $B\_{total} = E[\sum w\_t r\_t]$, to be controlled relative to $n$ (e.g., $B\_{total} \lesssim n^2$ for theorem 9's expected error bound). If each query involves computing a loss on a subsample of size $w\_t$ and quantizing to $r\_t \approx 1/\epsilon$ levels, the total number of adaptive queries $T$ is limited such that $T \cdot w\_t / \epsilon \lesssim n^2$. This imposes a polynomial limit on $T$. this contrasts with the original ladder mechanism, whose analysis supports a potentially exponential number of submissions $k$.

Second, the subsample loss $L\_t$ is an inherently noisier estimate of the true loss $R\_D(f\_t)$ compared to the full-sample loss $R\_S(f\_t)$ used by the original ladder mechanism. The standard deviation of $L\_t$ scales as $1/\sqrt{w\_t}$, compared to $1/\sqrt{n}$ for $R\_S(f\_t)$. Since $w\_t \ll n$, this higher variance makes it harder to reliably discern true performance differences between submissions using the subsample estimate.

Third, there is a trade-off between precision and the number of queries. Blanc's framework requires the interactive query output $y\_t$ to belong to a finite set $Y\_t$ of size $r\_t$. If the subsample loss $L\_t = R\_{S'}(f\_t)$ is continuous, an explicit step is needed to map $L\_t$ to a finite set $Y\_t$ (e.g., rounding or binning). This quantization step introduces an error ($\|y\_t - L\_t\| \le \epsilon/2$ if rounding to precision $\epsilon = 1/r\_t$). Furthermore, the size $r\_t$ creates a trade-off: increasing precision (larger $r\_t$) reduces quantization error but tightens the constraint on the number of allowed queries $T$ ($T w\_t r\_t \lesssim n^2$).

Finally, the guarantee targets differ. Blanc's theorems provide bounds on the maximum error of post-hoc test queries $\psi$, chosen based on the interaction transcript $y$. These bounds ensure that conclusions drawn *after* the interaction generalize. The ladder mechanism specifically bounds the leaderboard error $\max\_{1 \le t \le k} \|R\_t - \min\_{1 \le i \le t} R\_D(f\_i)\|$, ensuring the reported best score tracks the true best performance throughout the interaction. Defining a post-hoc test query $\psi$ whose error (comparing $\mu\_\psi(S)$ to $\mu\_\psi(D)$) directly corresponds to or bounds the leaderboard error term (comparing $R\_t$ to $R\_D(f\_{i^*(t)})$) is not straightforward, as they compare different quantities ($R\_t$ is an output, $R\_D$ is a property of inputs). The guarantees address different aspects of reliability.    