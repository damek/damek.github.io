---
title: "multi head, multi query, and grouped query attention"
date: 2025-05-03
---


$$
X\in\mathbb{R}^{T\times d},\qquad d = H\,d_k,\qquad 
{\rm softmax}(S)_{ij} = \exp(S_{ij}) \big/ \textstyle\sum_{m}\exp(S_{im}).
$$

###   multi-head attention (mha) 
the total number of K/V heads is H
$$
\begin{aligned}
(1)\;& Q_h = X W_{Q,h},\; K_h = X W_{K,h},\; V_h = X W_{V,h}; \\[2pt]
(2)\;& S_h = \tfrac{1}{\sqrt{d_k}}\,Q_h K_h^{\top}; \\[2pt]
(3)\;& \alpha_h = {\rm softmax}(S_h); \\[2pt]
(4)\;& Z_h = \alpha_h V_h; \\[2pt]
(5)\;& Z = \bigl[Z_1\;\Vert\;\dots\;\Vert\;Z_H\bigr]\,W_O. \\[4pt]
\text{Cache size: }&2\,H\,T\,d_k \quad(\text{keys+values}). 
\end{aligned}
$$

###  multi-query attention (mqa)
the total number of K/V heads is 1
$$
\begin{aligned}
(1)\;& Q_h = X W_{Q,h},\qquad K = X W_K,\qquad V = X W_V; \\[2pt]
(2)\;& S_h = \tfrac{1}{\sqrt{d_k}}\,Q_h K^{\top}; \\[2pt]
(3)\;& \alpha_h = {\rm softmax}(S_h),\qquad Z_h = \alpha_h V; \\[2pt]
(4)\;& Z = \bigl[Z_1\;\Vert\;\dots\;\Vert\;Z_H\bigr]\,W_O. \\[4pt]
\text{Cache size: }&2\,T\,d_k \quad(\text{$H$-fold reduction}). 
\end{aligned}
$$

##   grouped-query attention (gqa)
the total number of K/V heads is G, where 1 < G < H
$$
\begin{aligned}
(1)\;& \text{partition heads into groups }g=1,\dots,G\ \text{of size }H/G; \\[2pt]
(2)\;& Q_h = X W_{Q,h},\;
          K_g = X W_{K,g},\;
          V_g = X W_{V,g}\quad(h\in g); \\[2pt]
(3)\;& S_h = \tfrac{1}{\sqrt{d_k}}\,Q_h K_g^{\top}; \\[2pt]
(4)\;& \alpha_h = {\rm softmax}(S_h),\qquad Z_h = \alpha_h V_g; \\[2pt]
(5)\;& Z = \bigl[Z_1\;\Vert\;\dots\;\Vert\;Z_H\bigr]\,W_O. \\[4pt]
\text{Cache size: }&2\,G\,T\,d_k \quad(\text{$H/G$-fold reduction}). 
\end{aligned}
$$

###  purpose
smaller #K/V heads $\;\Rightarrow\;$ smaller [KV-cache](../what-is-kv-cache) $\;\Rightarrow\;$ lower memory-bandwidth during autoregressive decoding, hence higher tokens per second. quality degrades monotonically with the reduction factor; $G$ is a hardware–quality dial.
