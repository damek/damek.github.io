---
title: "What is kv cache?"
date: 2025-05-03
description: "A simple concept with a complicated name"
---

$$
\begin{aligned}
\textbf{(1) Projections:}\quad &
q = W_Q x_{t+1},\;
k = W_K x_{t+1},\;
v = W_V x_{t+1};\\[2mm]
%
\textbf{(2) Cache append:}\quad &
K_{1:t+1} = \begin{bmatrix} K_{1:t}\\ k \end{bmatrix},\;
V_{1:t+1} = \begin{bmatrix} V_{1:t}\\ v \end{bmatrix};\\[2mm]
%
\textbf{(3) New logit:}\quad &
s_{t+1} = \frac{1}{\sqrt{d}}\;q k^{\top},\qquad
e_{t+1} = e^{s_{t+1}};\\[2mm]
%
\textbf{(4) Incremental softmax:}\quad &
Z_{t+1} = Z_{t} + e_{t+1},\\
&\alpha_i^{(t+1)} = \alpha_i^{(t)}\frac{Z_{t}}{Z_{t+1}}\;\;(1\le i\le t),\qquad
\alpha_{t+1}^{(t+1)} = \frac{e_{t+1}}{Z_{t+1}};\\[2mm]
%
\textbf{(5) Output:}\quad &
z_{t+1} = \sum_{i=1}^{t+1} \alpha_i^{(t+1)} v_i
      = \alpha^{(t+1)} V_{1:t+1}.
\end{aligned}
$$
