---
title: "Modded-NanoGPT Walkthrough II: Muon Optimizer, Model Architecture, and Parallelism"
date: 2025-05-22
updated: 2025-06-02
tags: [pytorch, transformers, optimization, distributed]
description: "Part II: Muon optimizer, GPT architecture details, and distributed training in modded-nanogpt."
---

In [Part I](/random/modded-nanogpt-walkthrough-i) of this walkthrough, we covered the initial setup, compiler configurations, and custom FP8 operations within the `modded-nanogpt` repository's `train_gpt.py` script. This second part continues the walkthrough of `train_gpt.py`. We will look at the Muon optimizer, GPT model architecture, and the distributed training strategies.

### Table of Contents

- [The Muon Optimizer: Iterative Orthogonalization for Updates](#the-muon-optimizer-iterative-orthogonalization-for-updates)
  - [A. `zeropower_via_newtonschulz5`: Orthogonalizating the gradient](#a-zeropower_via_newtonschulz5-orthogonalizating-the-gradient)
  - [B. The `Muon` Optimizer Class](#b-the-muon-optimizer-class)
- [GPT Model Architecture: Component Details](#gpt-model-architecture-component-details)
  - [A. Core Building Blocks](#a-core-building-blocks)
  - [C. The `GPT` Model Assembly](#c-the-gpt-model-assembly)
- [Parallelism and Distributed Training](#parallelism-and-distributed-training)
  - [B. Distributed Data Loading](#b-distributed-data-loading)
  - [C. Setting the Stage: Hyperparameters and Environment](#c-setting-the-stage-hyperparameters-and-environment)
  - [D. Model, Optimizers, and Schedules](#d-model-optimizers-and-schedules)
  - [E. Pre-computation and Synchronization: Warmup and Gradient Overlap](#e-pre-computation-and-synchronization-warmup-and-gradient-overlap)
  - [F. The Main Training Loop](#f-the-main-training-loop)
  - [G. Finalization](#g-finalization)

### The Muon Optimizer: Iterative Orthogonalization for Updates

The `train_gpt.py` script introduces a custom optimizer called `Muon`, that is specifically used with the matrix layers of the transformer model. (For the nonmatrix layers, they use an Adam method.) In short, Muon replaces the matrix blocks of the gradient[^0] with a new matrix with better conditioning and the same row/column space. This is achieved by applying an iterative algorithm called the Newton-Schulz.

Why do they do this? From my read of the literature (up to June 02, 2025), there has been no strong theoretical justification for doing so. Although we can realize it as a variant of gradient descent in a block spectral norm, we don't know why it's good to do gradient descent in the spectral norm for transformer models. 🤷

#### A. `zeropower_via_newtonschulz5`: Orthogonalizating the gradient

The function `zeropower_via_newtonschulz5` applies Newton-Schulz to an input matrix $G$. Classically, the method was designed to do the following: 

> If $G$ has a singular value decomposition (SVD) $G = U \Sigma V^T$, this iteration (when properly initialized) converges quadratically to a matrix $G' \approx U I' V^T$. In this expression, $I'$ is a diagonal matrix with entries of 1 where $\Sigma$ had non-zero singular values, and 0 otherwise. This process yields an (approximately) orthogonal matrix with the same row and column space as $G$.

The method in the code is slightly different. It instead modifies the iteration so that the singular values near zero become larger more quickly, but the limiting singular values (empirically) reach the interval between .5 and 1.5. This seems to work OK.

```python
@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
```
Walking through the code, the operations are as follows: The input tensor `G`, representing a gradient update, is first cast to `bfloat16` precision. If the input matrix `G` has more rows (`G.size(-2)`) than columns (`G.size(-1)`), it is transposed. Let `X` be this potentially transposed matrix. The iteration then computes `A = X @ X.mT`. The dimensions of `A` are `X.size(-2) x X.size(-2)`. The initial transposition ensures `X.size(-2)` is the smaller of `G`'s original two dimensions. This makes the intermediate matrix `A` (and subsequent products like `A@A`) smaller, reducing computational cost.

Next, `X` is normalized by its spectral norm. The code approximates this using `X.norm(dim=(-2, -1), keepdim=True)`, and adds a small epsilon `1e-7` for numerical stability. This normalization puts $X$ into the region of quadratic convergence for the (classical) Newton-Schulz iteration.

The core of the function is the iterative application of a quintic formula:

$$ X_{k+1} = a X_k + (b(X_k X_k^T) + c(X_k X_k^T)^2) X_k $$

The constants $a, b, c$ are `(3.4445, -4.7750, 2.0315)`. This iteration runs for a specified number of `steps` (the default `ns_steps` for Muon is 5[^1]). Finally, if `X` was initially transposed, it is transposed back. The `@torch.compile` decorator is used to optimize this function into efficient GPU kernels.

#### B. The `Muon` Optimizer Class

The `Muon` class, defined by inheriting from `torch.optim.Optimizer`, implements the custom update rule for 2D matrix parameters.


```python
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() 
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
```
The `__init__` method groups parameters by their total number of elements (`p.numel()`). For each unique element count (`current_param_size`), it pre-allocates an `update_buffer` tensor of shape `(world_size, current_param_size)`. This grouping ensures that when `dist.all_gather_into_tensor` is called for this `update_buffer`, all GPUs contribute an input tensor of the same size, a requirement for the all gather operation.

The `step()` method is called after gradients are globally averaged. It processes parameters in `param_groups`. The loop `for base_i in range(len(params))[::self.world_size]` iterates over starting indices for parameter chunks. `base_i` takes values `0, world_size, 2*world_size...`. Each GPU (`self.rank`) processes parameter `p = params[base_i + self.rank]`.

For example, if `world_size = 8` and `len(params) = 20`:
*   `base_i = 0`: GPUs 0-7 process `params[0]` through `params[7]`.
*   `base_i = 8`: GPUs 0-7 process `params[8]` through `params[15]`.
*   `base_i = 16`: GPUs 0-3 process `params[16]` through `params[19]`. GPUs 4-7 execute the `else` branch.

If a GPU has a valid parameter `p` with (averaged) gradient `g`:
1.  *Momentum Accumulation*: The momentum buffer `buf` for $W_t$ (parameter `p`) is updated:

    $$ \text{buf}_t = m \cdot \text{buf}_{\text{prev}} + (1-m) \cdot \nabla L(W_t) $$

    via `buf.lerp_(g, 1 - group["momentum"])`.
2.  *Effective Gradient Calculation*: The effective gradient $g\_{\text{eff}}$ is set. If Nesterov, 

    $$ g_{\text{eff}} = (1-m) \cdot \nabla L(W_t) + m \cdot \text{buf}_t $$

    via `g.lerp_(buf, group["momentum"])`. Else, $g\_{\text{eff}} = \text{buf}\_t$.
3.  *Orthogonalization*: $g\_{\text{eff}}$ is processed by `zeropower_via_newtonschulz5` and flattened to $g\_{\text{ortho}}$.

If a GPU has no new parameter for the current `base_i` (e.g., GPUs 4-7 when `base_i=16` in the example), `g` is set to `update_buffer_views[self.rank]`. This ensures all ranks contribute a correctly-sized tensor to `dist.all_gather_into_tensor`. This tensor `g` (either $g_{\text{ortho}}$ or the placeholder) is then gathered asynchronously into `update_buffer` via `handle = dist.all_gather_into_tensor(...)`.

The `update_prev()` function applies the updates. It calls `handle.wait()` to ensure `all_gather` is complete. `params_world` slices the parameters processed in the current `base_i` chunk. For each parameter $W\_t$ (`p_world`) in this chunk and its corresponding gathered $g\_{\text{ortho_gathered}}$ (`g_world` from `update_buffer_views`), the update 

$$W_{t+1} = W_t - (\eta \cdot \alpha_{\text{shape}}) \cdot g_{\text{ortho_gathered}}$$ 

is applied. Here, $\eta$ is `group["lr"]` and $\alpha_{\text{shape}} = \sqrt{\max\left(1, \frac{\text{rows}}{\text{cols}}\right)}$ is a shape-dependent scaling factor.


### GPT Model Architecture: Component Details

The model implemented in `train_gpt.py` is a decoder-only Transformer, with several specific architectural choices.

#### A. Core Building Blocks

1.  **Normalization: `norm()`**
    ```python
    def norm(x: Tensor):
        return F.rms_norm(x, (x.size(-1),))
    ```
    This `norm` function applies Root Mean Square Layer Normalization (RMSNorm). Note that it has no trainable parameters! It normalizes the input tensor `x` over its last dimension. For a vector $x \in \mathbb{R}^n$ (representing features along the last dimension), the operation is:

    $$ \text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^n x_j^2 + \epsilon}} $$

    The `F.rms_norm` function adds a small epsilon in case $x$ is near zero. This normalization appears in several places within the model architecture. The `eps` argument in `F.rms_norm` is not specified, so it defaults to `torch.finfo(x.dtype).eps`. This is the smallest representable positive number such that `1.0 + eps != 1.0` for the given `dtype` of `x`.


2.  **`CastedLinear`**
    ```python
    class CastedLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
            super().__init__(in_features, out_features, bias=False)
            self.use_fp8 = use_fp8
            self.x_s = x_s
            self.w_s = w_s
            self.grad_s = grad_s

        def reset_parameters(self) -> None:
            std = 0.5 * (self.in_features ** -0.5)
            bound = (3 ** 0.5) * std
            with torch.no_grad():
                self.weight.uniform_(-bound, bound)

        def forward(self, x: Tensor):
            if self.use_fp8 and self.training:
                _x = x.flatten(0, -2)
                out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
                return out.reshape(*x.shape[:-1], -1)
            else:
                return F.linear(x, self.weight.type_as(x))
    ```
    The `CastedLinear` layer is a custom linear layer, inheriting from `nn.Linear`, designed to optionally use FP8 precision for its matrix multiplication. Its `forward` pass uses the custom `mm_op` (discussed in Part I) if `self.use_fp8` is true and the model is in training mode. This `mm_op` performs matrix multiplication using FP8 with specified scaling factors (`self.x_s`, `self.w_s`, `self.grad_s`). If these conditions are not met (e.g., during evaluation or if FP8 is disabled), it falls back to a standard `F.linear` operation. This layer does not use a bias term.

    The `reset_parameters` method defines a custom weight initialization. The standard deviation is calculated as $\text{std} = 0.5 \cdot (\text{in\_features})^{-0.5}$. The weights $W$ are then initialized from a uniform distribution $U[-\sqrt{3} \cdot \text{std}, \sqrt{3} \cdot \text{std}]$.

3.  **`Rotary` Embeddings**
    ```python
    class Rotary(nn.Module):
        def __init__(self, dim: int, max_seq_len: int):
            super().__init__()
            angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
            angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
            t = torch.arange(max_seq_len, dtype=torch.float32)
            theta = torch.einsum("i,j -> ij", t, angular_freq)
            self.cos = nn.Buffer(theta.cos(), persistent=False)
            self.sin = nn.Buffer(theta.sin(), persistent=False)

        def forward(self, x_BTHD: Tensor):
            assert self.cos.size(0) >= x_BTHD.size(-3)
            cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
            x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), 3).type_as(x_BTHD)
    ```
    This module implements Rotary Position Embeddings (RoPE). RoPE is a method to incorporate positional information into the self-attention mechanism by applying position-dependent rotations to the query and key vectors. The idea is that the dot product of two vectors rotated by angles $\theta_m$ and $\theta_n$ respectively, will depend on their relative angle $\theta_m - \theta_n$. This allows attention scores to reflect the relative positions of tokens.

    In the `forward` method, an input tensor `x_BTHD` (e.g., a query or key vector for each head, with shape Batch size, Sequence length, Number of attention heads, Dimension per head) has its last dimension (Dim_per_head, $D_h$) divided into pairs of features. Each pair $(x_1, x_2)$ at sequence position `pos` is rotated:

    $$ \begin{pmatrix} x'_1 \\ x'_2 \end{pmatrix}_{pos} = \begin{pmatrix} \cos \theta_{pos,j} & \sin \theta_{pos,j} \\ -\sin \theta_{pos,j} & \cos \theta_{pos,j} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}_{pos} $$
        
    The `__init__` method precomputes the `cos` and `sin` values for these rotations. It calculates angles $\theta_{pos, j} = \text{pos} \cdot \text{angular\_freq}_j$. A "half-truncate RoPE" modification is used here: `angular_freq` is constructed such that only the first `dim//4` frequency components are non-zero (where `dim` is `head_dim`), meaning rotations are applied to only half of the features within each head.

4.  **`CausalSelfAttention`**
    ```python
    class CausalSelfAttention(nn.Module):
        def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = head_dim
            hdim = num_heads * head_dim
            std = 0.5 * (dim ** -0.5)
            bound = (3 ** 0.5) * std
            self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.rotary = Rotary(head_dim, max_seq_len)
            self.c_proj = CastedLinear(hdim, dim)
            self.c_proj.weight.detach().zero_() 

        def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
            B, T = x.size(0), x.size(1)
            assert B == 1, "Must use batch size = 1 for FlexAttention"
            q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
            q, k = norm(q), norm(k)
            q, k = self.rotary(q), self.rotary(k)
            if ve is not None:
                v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
            else: 
                v = self.lambdas[0] * v
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
            y = y.contiguous().view(B, T, self.num_heads * self.head_dim) 
            y = self.c_proj(y)
            return y
    ```
    This module implements multi-head causal self-attention. "Causal" means that for any given token in a sequence, its representation can only be influenced by preceding tokens and itself, not by future tokens. This makes sense the model we're training can only generate text one token at a time.

    In `__init__`: A single weight tensor, `self.qkv_w` (shape: `(3, num_heads * head_dim, model_dim)`), is initialized to project the input into Query (Q), Key (K), and Value (V) spaces for all attention heads simultaneously. Learnable scalar parameters, `self.lambdas`, are prepared for later mixing "value embeddings" (`ve`) into the V tensors. The final output projection layer, `self.c_proj` (an instance of `CastedLinear`), has its weight matrix zero-initialized. This zero-initialization means the `c_proj` layer initially outputs a zero tensor, so at the start of training, the attention mechanism's output (after this projection) does not add to the residual path.

    The `forward` method defines works as follow: The input `x` to this attention module must have a batch size of one (`B == 1`). This requirement stems from `flex_attention`'s use with `create_blockmasks`. The `create_blockmasks` function generates sequence-dependent attention masks by identifying document boundaries (via token ID 50256) within each specific input sequence. Processing one long sequence at a time simplifies applying these unique masks, which incorporate document structure and sliding window logic. The overall training still processes multiple distinct sequences in parallel across GPUs through data parallelism.

    1.  **QKV Projection**: The input `x` is linearly projected using the flattened `self.qkv_w`. If $X \in \mathbb{R}^{B \times T \times \text{dim}}$ and $W_{QKV}$ is the appropriately reshaped `qkv_w`, this computes $X W_{QKV}^T$. The result is then viewed and chunked to separate Q, K, and V, each having shape (Batch size, Sequence length, Number of attention heads, Dimension per head).

    2.  **Q/K Preparation**: The Q and K tensors are first normalized using `norm()` (RMSNorm, implementing QK Norm) and then Rotary Position Embeddings (RoPE) are applied via `self.rotary()`.

    3.  **Value Modification**: The V tensor is potentially augmented. If `ve` (token value embeddings, derived from the input sequence) are provided, they are mixed into V using the learnable `self.lambdas`: $V_{new} = \lambda_0 V_{orig} + \lambda_1 ve$.

    4.  **Attention Calculation**: The Q, K, and V tensors, currently shaped (Batch size, Sequence length, Number of heads, Dimension per head), are transposed to (Batch size, Number of heads, Sequence length, Dimension per head) because this layout is expected by the `flex_attention` function. `flex_attention` then computes the attention output using these transposed Q, K, V, the provided `block_mask`, and a fixed `scale=0.12` for the dot products. Conceptually, for each head, we compute:
    
        $$ \text{Output}_h = \text{softmax}\left(\frac{Q_h K_h^T}{0.12} + M_h\right) V_h $$
    where $M_h$ is the attention mask for that head derived from `block_mask`. 

    5.  **Output Processing**: The output `y` from `flex_attention` (initially with layout Batch, Heads, SeqLen, HeadDim) is transposed back via `y.transpose(1, 2)`, resulting in a (Batch size, Sequence length, Number of heads, Dimension per head) layout. This transpose operation typically makes the tensor's underlying memory non-contiguous because it changes the stride information without reordering the actual data elements. The subsequent `.view(B, T, self.num_heads * self.head_dim)` operation reshapes `y` by collapsing the "Number of heads" and "Dimension per head" into a single feature dimension. Such a reshaping, which changes how elements are grouped across multiple original dimensions, requires the tensor's data to be contiguous in memory. Therefore, `.contiguous()` is called on `y` to create a new tensor with its data laid out sequentially if it isn't already. This allows the `.view()` operation to correctly reinterpret the tensor's shape. The reshaped tensor is then passed through `self.c_proj`.

6.  **`MLP`**
    ```python
    class MLP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            hdim = 4 * dim
            self.c_fc = CastedLinear(dim, hdim)
            self.c_proj = CastedLinear(hdim, dim)
            self.c_proj.weight.detach().zero_()

        def forward(self, x: Tensor):
            x = self.c_fc(x)
            x = F.relu(x).square() 
            x = self.c_proj(x)
            return x
    ```
    This is a two-layer MLP. The hidden dimension `hdim` is 4 times the input/output dimension `dim`. It uses `CastedLinear` layers, so FP8 computation is possible. The projection layer `c_proj` is zero-initialized. The activation function is ReLU-squared: $\text{act}(z) = (\text{ReLU}(z))^2$.

6.  **`Block`**
    ```python
    class Block(nn.Module):
        def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
            super().__init__()
            self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
            self.mlp = MLP(dim)
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

        def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
            x = self.lambdas[0] * x + self.lambdas[1] * x0
            if self.attn is not None:
                x = x + self.attn(norm(x), ve, block_mask)
            x = x + self.mlp(norm(x))
            return x
    ```
    The `Block` module defines one layer of the Transformer. It combines an attention mechanism and an MLP.

    A modification to the standard Transformer block is the input mixing stage: `x_mixed = self.lambdas[0] * x + self.lambdas[1] * x0`. Here, `x` is the output from the preceding layer (or the initial embedding for the first block), and `x0` is the initial normalized token embedding of the input sequence, which is passed as an argument to every block. These two tensors are combined using learnable scalar weights `self.lambdas`. This provides each block direct access to the initial input representation.

    The attention sublayer (`self.attn`) is not present for the 8th layer (`layer_idx == 7`).

    The sequence of operations within a block can be represented as:
    1.  Input mixing: $x_{\text{mixed}} = \lambda_0 x_{\text{in}} + \lambda_1 x_0$
    2.  Attention path (if `self.attn` is active): 
    
        $$x_{\text{attn_out}} = x_{\text{mixed}} + \text{Attention}(\text{norm}(x_{\text{mixed}}), ve, \text{mask})$$

        If attention is skipped, 

        $$x_{\text{attn_out}} = x_{\text{mixed}}.$$

    3.  MLP path: $x_{\text{out}} = x_{\text{attn\_out}} + \text{MLP}(\text{norm}(x_{\text{attn\_out}}))$

    Normalization (`norm()`) is applied before the attention and MLP components.

#### C. The `GPT` Model Assembly
```python
class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() 
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)
        x = x0 = norm(self.embed(input_seq)[None]) 
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        return loss
```
The `GPT` class's `__init__` method assembles the model's layers. It initializes a standard token embedding layer (`self.embed`). A distinct feature is `self.value_embeds`: three separate `nn.Embedding` layers. These generate embeddings from the input sequence, which are later mixed into the Value (`V`) tensors within specific attention layers, providing an alternative pathway for token-specific information to influence attention outputs. The core of the model is `self.blocks`, a stack of `Block` modules. The final projection to logits is handled by `self.lm_head`. This is a `CastedLinear` instance using FP8 precision and specific scaling factors for its matrix multiplication; its weight is zero-initialized. The vocabulary size for this head is padded to the nearest multiple of 128 using `next_multiple_of_n(vocab_size, n=128)`. Padding vocabulary size to a multiple of a power of two (like 64 or 128) can improve GPU kernel efficiency, [a point Andrej Karpathy](https://x.com/karpathy/status/1621578354024677377){:target="_blank"} noted can yield significant speedups by enabling more optimized computation paths. 
> The most dramatic optimization to nanoGPT so far (~25% speedup) is to simply increase vocab size from 50257 to 50304 (nearest multiple of 64). This calculates added useless dimensions but goes down a different kernel path with much higher occupancy. Careful with your Powers of 2.

`self.skip_weights` are learnable parameters, initialized to ones, for U-Net style skip connections between layers; there are `num_layers // 2` such weights, as `num_layers` is asserted to be even.

The `create_blockmasks` method generates attention masks for `flex_attention`. It defines a `BLOCK_SIZE` of 128 tokens. Token ID 50256 is used to delimit documents via `docs = (input_seq == 50256).cumsum(0)`, assigning a document ID to each token. The `document_causal` function, passed as `mask_mod` to `BlockMask.from_kv_blocks`, then ensures that attention scores are computed only between tokens within the same document, in addition to enforcing causality. This method also implements sliding window attention, where `sliding_window_num_blocks` dynamically sets the attention span. It produces two `BlockMask` objects, `long_bm` and `short_bm`, corresponding to different window sizes (a main window and a halved window), allowing layers to have varied attention scopes.

The `forward` method defines the data flow through the assembled model: Value embeddings (`ve_for_layers`) are computed from `input_seq` using each of the three embedding layers in `self.value_embeds`, yielding three distinct sets of value embeddings: $VE_0, VE_1, VE_2$. These are then distributed to the Transformer blocks according to the pattern shown below for a 12-layer model:
```
Layer Index | Value Embedding Used
-----------------------------------
Block 0     |        VE_0
Block 1     |        VE_1
Block 2     |        VE_2
Block 3     |        None
Block 4     |        None
Block 5     |        None  <-- Middle layers (len(blocks)-6 = 12-6 = 6 layers)
Block 6     |        None
Block 7     |        None  <-- (Note: This layer also skips attention)
Block 8     |        None
Block 9     |        VE_0  <-- Third to last
Block 10    |        VE_1  <-- Second to last
Block 11    |        VE_2  <-- Last
```
The code `ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]` implements this assignment. This pattern applies distinct, learned value-modifying signals from `self.value_embeds` primarily to the initial and final stages of processing within the network stack. Attention masks (`long_bm`, `short_bm`) are generated. A fixed pattern then assigns either a long or short window mask to each layer in `self.blocks`. The `input_seq` is embedded and normalized to produce `x0`; this `x0` (the initial token representation) is passed to every `Block` for input mixing. A U-Net style skip connection mechanism is implemented. This structure creates long-range shortcuts by connecting outputs from earlier layers to inputs of later, symmetrically corresponding layers. Let `num_encoder_layers = num_layers // 2`.
```
Input x (from previous layer or initial embedding x0)
  |
  V
Block 0  --> Store output_0 (skip_connections_stack.append(x))
  |
  V
...
  |
  V
Block (num_encoder_layers - 1) --> Store output_(num_encoder_layers-1)
  |
  V
--------------------------------------------------------------------
  |  (Now in "decoder" part, using stored outputs)
  V
Input to Block (num_encoder_layers) = x_prev + skip_weights[0] * output_(num_encoder_layers-1) <-- pop()
  |
  V
Block (num_encoder_layers)
  |
  V
...
  |
  V
Input to Block (num_layers - 1) = x_prev + skip_weights[num_encoder_layers-1] * output_0 <-- pop()
  |
  V
Block (num_layers - 1)
  |
  V
Final Output x
```
For the first `num_encoder_layers`, the output `x` of each block is stored. For the subsequent `num_encoder_layers`, before processing its input, each block receives an added component: an output from a symmetrically corresponding earlier layer (retrieved via `skip_connections_stack.pop()`) scaled by a learnable `self.skip_weights`.

After processing through all blocks, the final `x` is normalized. Logits are computed by `self.lm_head` (an FP8 `CastedLinear` layer) and cast to float. A logit softcapping function is then applied: `logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))`. This technique was apparently taken from Gemma 2. Finally, the cross-entropy loss is computed between the predicted logits and the `target_seq`.

### Parallelism and Distributed Training

The `train_gpt.py` script achieves its performance on 8 H100 GPUs through a sophisticated distributed training strategy. This strategy primarily employs data parallelism, where each GPU processes a unique shard of data. However, a key optimization is introduced to overlap gradient computation with the communication required for their synchronization. Furthermore, the Muon optimizer internally uses a parameter-sharded approach for its update calculations after global gradients are available.

The overall process for a single training iteration involves these main stages of parallelism and synchronization:

1.  **Data-Parallel Gradient Computation with Overlapped Communication**:
    *   Each GPU processes its unique data shard through its local model copy, executing the forward pass and initiating the backward pass (`model(...).backward()`).
    *   During the backward pass, as gradients for individual parameters (or groups of parameters called "buckets") are computed, a **post-accumulate grad hook** (`_gradient_hook`) is triggered.
    *   This hook immediately launches an **asynchronous `dist.all_reduce` operation** (with `op=dist.ReduceOp.AVG`) for the bucket whose gradients are now ready. This allows the communication (synchronization) of these gradients to begin while other gradients for preceding layers are still being computed.
    *   After `model(...).backward()` returns, the script calls `wait_for_gradients()`. This function ensures all launched asynchronous `all_reduce` operations for all buckets have completed. At this point, every GPU holds an identical copy of the globally averaged gradient for every model parameter.

2.  **Optimizer Parameter Updates**: With the globally averaged gradients available, the optimizers update the model parameters.
    *   **Adam Optimizer (`optimizer1`)**: Parameters managed by Adam are updated by each GPU using the averaged gradients, maintaining synchronization.
    *   **Muon Optimizer (`optimizer2`)**: For parameters managed by Muon (e.g., hidden matrices), each GPU uses the globally averaged gradients as input to Muon's `step()` method. Within this step:
        1.  Parameters are processed in shards. Each GPU computes momentum-adjusted and then orthogonalized updates for its assigned parameters within the current shard, using the averaged gradients.
        2.  These locally derived orthogonalized updates are then collected from all GPUs into a common buffer using `dist.all_gather_into_tensor`.
        3.  Finally, each GPU applies the relevant gathered orthogonalized update (from the common buffer) to its local copy of the parameters in that shard, using Muon's learning rate and scaling.


The following diagram illustrates this for one training step:

```
Per Training Step:
+-------------------------------------------------------------------------------------------------+
|                                       All GPUs (Rank 0 to N-1)                                  |
+-------------------------------------------------------------------------------------------------+
| 1. Data Loading & Local Computation (Data Parallelism):                                         |
|    GPU_i: Loads unique data_shard_i.                                                            |
|    GPU_i: Computes loss_i = model(inputs_i, targets_i, ...).                                    |
|-------------------------------------------------------------------------------------------------|
| 2. Backward Pass & Asynchronous Gradient Averaging (Overlapped):                                |
|    GPU_i: Initiates loss_i.backward().                                                          |
|    As gradients for a parameter bucket become available:                                        |
|      Hook triggers: dist.all_reduce(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)          |
|      // Computation of other gradients continues while this bucket syncs.                       |
|    After backward() call completes:                                                             |
|      wait_for_gradients() // Ensures all async all_reduces are finished.                        |
|    // Result: p.grad is now identical (averaged_grad) on all GPUs.                              |
|-------------------------------------------------------------------------------------------------|
| 3. Parameter Update Phase (Sequential Optimizers, using averaged_grad):                         |
|    a. Adam Optimizer Step (optimizer1.step()):                                                  |
|       GPU_i: Updates its local copy of Adam-managed parameters using averaged_grad.             |
|       // Parameters remain synchronized.                                                        |
|                                                                                                 |
|    b. Muon Optimizer Step (optimizer2.step()):                                                  |
|       // For Muon-managed parameters, using globally averaged_grad as input:                    |
|       // Internal Muon processing happens in shards of these parameters:                        |
|       For each shard_s of Muon_params:                                                          |
|         GPU_i: Processes its assigned p_s_i from shard_s:                                       |
|           - Applies momentum to averaged_grad for p_s_i.                                        |
|           - Orthogonalizes the result --> local_ortho_update_s_i.                               |
|         All GPUs (for shard_s):                                                                 |
|           dist.all_gather_into_tensor(update_buffer_s, [local_ortho_update_s_0, ...])           |
|           // update_buffer_s now contains all ortho_updates for parameters in shard_s.          |
|         GPU_i (in Muon's update_prev for shard_s):                                              |
|           handle.wait()                                                                         |
|           Updates its local copy of p_s_i using its corresponding slice from update_buffer_s.   |
|       // Parameters remain synchronized.                                                        |
+-------------------------------------------------------------------------------------------------+
```

We will now examine the specific code sections that implement these distributed operations, starting with the data loading.

#### B. Distributed Data Loading

1.  **`_load_data_shard()`**
    ```python
    def _load_data_shard(file: Path):
        header = torch.from_file(str(file), False, 256, dtype=torch.int32) 
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2]) 
        with file.open("rb", buffering=0) as f: 
            tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) 
            f.seek(256 * 4) 
            nbytes = f.readinto(tokens.numpy()) 
            assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
        return tokens
    ```
    This function, `_load_data_shard`, serves as a helper for reading a single binary data shard into CPU memory. Its design incorporates integrity checks for the data file and employs several I/O optimizations. It is called by the data generator responsible for feeding batches to each GPU process.

    The function begins by reading a 256-integer header from the file using `torch.from_file`. This header, created during data preprocessing, contains a magic number (20240520) and a version (1), which are asserted to match expected values, ensuring file format compatibility. The header also specifies the number of tokens in the shard.

    For file I/O, the file is opened with `buffering=0`. Standard Python file operations can involve an internal buffer. Setting `buffering=0` makes Python interact more directly with the operating system's I/O for reads. For large, sequential reads of an entire file shard, this approach can avoid an intermediate copy between the OS buffer, Python's internal buffer, and the final destination.

    A `torch.uint16` tensor, `tokens`, is pre-allocated in pinned memory (`pin_memory=True`) to hold all tokens from the shard. Pinned memory is not paged out to disk by the OS. This allows the GPU's Direct Memory Access (DMA) engine to perform asynchronous data transfers from this CPU RAM to GPU VRAM, which requires stable physical memory addresses.

    After skipping the header bytes (`f.seek(256 * 4)`), data is read directly into the `tokens` tensor's memory using `f.readinto(tokens.numpy())`. This reads into a pre-allocated NumPy view sharing memory with the PyTorch tensor, avoiding the creation of an intermediate bytes object. An assertion then verifies that the correct number of bytes was read. The function returns the populated `tokens` tensor, which resides in pinned CPU RAM. The file is automatically closed by the `with` statement.

2.  **`distributed_data_generator()`**
    ```python
    def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        assert batch_size % world_size == 0
        local_batch_size = batch_size // world_size
        file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
        tokens, pos = _load_data_shard(next(file_iter)), 0
        while True:
            if pos + batch_size + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
            pos += batch_size
            yield inputs, targets
    ```
    Each GPU process runs its own instance of `distributed_data_generator`. This generator's purpose is to continuously supply its GPU with unique (input, target) token pairs for training, ensuring that across all GPUs, the entire dataset is processed in a coordinated, sharded manner. Each GPU process instantiates this generator once (as train_loader before the main training loop begins) and then calls next() on it in each training step to obtain a batch.

    The data is assumed to be organized into multiple binary shard files (e.g., `fineweb_train_001.bin`, `fineweb_train_002.bin`, ...). The generator first lists all such files. The `batch_size` argument refers to the *global batch size* across all GPUs. `local_batch_size` is the portion of this global batch that each individual GPU will handle.

    Initially, each generator loads the first data shard file into a CPU memory buffer (`tokens`) using `_load_data_shard`. `pos` tracks the starting position of the *next global batch* to be read from this `tokens` buffer.

    Inside the main `while True` loop, the generator prepares a batch for its specific GPU (`rank`).
    It first checks if the current `tokens` buffer has enough data remaining for the next global batch. If not (`pos + batch_size + 1 >= len(tokens)`), it discards the exhausted shard and loads the next one from `file_iter`, resetting `pos` to 0.

    Then, it carves out its designated slice for the current global batch. Imagine the `tokens` buffer for the current shard as a long tape of token IDs. `pos` marks where the current global batch begins on this tape. Each GPU calculates its own starting point within this global batch segment:
    `my_slice_start = pos + (rank * local_batch_size)`.
    It reads `local_batch_size + 1` tokens from this point to form its local buffer `buf`. The `+1` is needed to create the input-target pair: `inputs` are `buf[:-1]` and `targets` are `buf[1:]`. These are then sent to the GPU asynchronously.

    Consider a `world_size = 4` and a global `batch_size = 1024` tokens. `local_batch_size` would be 256.
    If `pos = 0` in the current shard `tokens`:
    *   GPU 0 (`rank=0`): reads `tokens[0 : 256+1]`
    *   GPU 1 (`rank=1`): reads `tokens[256 : 512+1]`
    *   GPU 2 (`rank=2`): reads `tokens[512 : 768+1]`
    *   GPU 3 (`rank=3`): reads `tokens[768 : 1024+1]`

    Visually, for one global batch from a shard:
    ```
    Shard `tokens`: [---------------------------------------------------------------------...]
                       ^ pos (start of current global batch)
                       |
    Global Batch:      [ GPU0_data | GPU1_data | GPU2_data | GPU3_data ]
                       <----------------- batch_size ----------------->
    ```
    Each GPU's generator independently takes its slice. After yielding its batch, each generator instance advances its *local* `pos` by the *global* `batch_size`. This prepares it to look for the *next* global batch segment in the current shard on its next call. Because all generators advance `pos` by the same global amount and use their `rank` to offset, they continue to pick up distinct, contiguous portions of the overall data stream defined by the sequence of shard files.

#### C. Setting the Stage: Hyperparameters and Environment

With the data loading mechanism understood, the script next establishes the fixed configuration for the training run and prepares the multi-GPU environment. This setup is crucial for reproducibility and coordinated parallel execution.

1.  **`Hyperparameters` Dataclass**
    ```python
    @dataclass
    class Hyperparameters:
        train_files = "data/fineweb10B/fineweb_train_*.bin" 
        val_files = "data/fineweb10B/fineweb_val_*.bin" 
        val_tokens = 10485760 
        train_seq_len = 48*1024 
        val_seq_len = 4*64*1024 
        num_iterations = 1770 
        cooldown_frac = 0.4 
        vocab_size = 50257
        val_loss_every = 125 
        save_checkpoint = False
    args = Hyperparameters()
    ```
    A `dataclass` is used to group fixed training parameters. This includes paths to training and validation data shards, the total number of validation tokens to use, sequence lengths for training and validation, the total number of training iterations, the fraction of training for learning rate cooldown, vocabulary size, validation frequency, and a flag for checkpoint saving. Using a dataclass provides a structured way to access these settings throughout the script via the `args` instance.

2.  **Distributed Environment Initialization**
    ```python
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8 
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0)
    ```
    The command `torchrun --standalone --nproc_per_node=8 train_gpt.py` initiates the distributed training by launching eight separate instances of the `train_gpt.py` script. Each instance, now an independent Python process, must first discover its role within the collective and establish communication with its peers. This section of code orchestrates that transformation.

    Each process queries its environment, set up by `torchrun`, to learn its unique global `RANK` (from 0 to 7), the total `WORLD_SIZE` (8), and its `LOCAL_RANK` which determines the specific GPU it will command. With `torch.cuda.set_device(device)`, each process claims its designated GPU.

    The call `dist.init_process_group(backend="nccl", ...)` is where these initially isolated processes formally join a communication group. By using the `nccl` backend, they enable high-speed data exchange directly between their NVIDIA GPUs. Before proceeding to any collective work like model weight synchronization, `dist.barrier()` ensures every process has successfully initialized and reached this common checkpoint. This prevents any process from starting operations prematurely, for instance, rank 0 attempting to broadcast model weights before other ranks are prepared to receive them. Finally, one process, `rank == 0`, is designated as the `master_process`, typically responsible for singular tasks like writing logs, to ensure clarity and avoid redundant output from all eight workers. Through these steps, eight independent script executions become a synchronized team.
    
3.  **Logging Setup**

    At the very beginning of the script (lines 3-4), the script's own source code is read into the `code` variable:
    ```python
    with open(sys.argv[0]) as f:
        code = f.read() 
    ```
    This `code` is later logged by the master process for exact reproducibility of experiments.
    ```python
    logfile = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)
    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    print0(code)
    print0("="*100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    def nvidia_smi():
        import subprocess  
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("="*100)
    ```
    A unique `run_id` is generated for logging. The `print0` function ensures that print statements are executed only by the `master_process` and are written to a uniquely named log file. The script logs its own source code, Python and PyTorch versions, and the output of `nvidia-smi` to fully document the execution environment.

#### D. Model, Optimizers, and Schedules

This phase constructs the GPT model, defines how different sets of its parameters will be optimized, and establishes schedules for dynamically adjusting the learning rate and attention window size during training.

1.  **Model Instantiation and Initial Synchronization**
    ```python
    model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=12, num_heads=6, model_dim=768,
                           max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    ```
    Each GPU process instantiates the `GPT` model and moves it to its GPU. The script then casts the parameters of `nn.Embedding` layers to `bfloat16` precision as part of the lower-precision training strategy. To ensure all model replicas begin with identical weights, `dist.broadcast(param.detach(), 0)` is called for every parameter, copying values from rank 0 to all other ranks.

2.  **Optimizer Setup**
    ```python
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    ```
    The script employs a dual-optimizer strategy, assigning different types of model parameters to either an Adam or a Muon optimizer. First, it categorizes the model's parameters: `hidden_matrix_params` capture the 2D (or higher-dimensional) weights within the Transformer `blocks` (excluding embeddings). Other parameters, such as `embed_params`, `scalar_params` (those with fewer than 2 dimensions), and the `head_params` (the output layer's weight), are grouped separately. The RMSNorm function used in this model does not have learnable parameters.

    These distinct parameter groups are then assigned: `optimizer1`, an `torch.optim.Adam` instance, manages the `head_params`, `embed_params`, and `scalar_params`, each with its own learning rate. The `fused=True` argument for Adam instructs PyTorch to use an optimized, single GPU kernel for its update step, combining multiple element-wise operations to reduce launch overhead. `optimizer2`, an instance of the `Muon` optimizer, is dedicated to the `hidden_matrix_params`. For later use by the learning rate scheduler, the initial learning rate for each parameter group is stored as `group["initial_lr"]`.


3.  **Learning Rate and Attention Window Schedules**

    ```python
    def get_lr(step: int):
        x = step / args.num_iterations 
        assert 0 <= x < 1
        if x < 1 - args.cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / args.cooldown_frac
            return w * 1.0 + (1 - w) * 0.1

    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    def get_window_size_blocks(step: int):
        x = step / args.num_iterations 
        assert 0 <= x <= 1
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)
    ```
    To guide the training process dynamically, the script implements two scheduling functions that adjust hyperparameters based on the current training `step`.

    The `get_lr(step)` function controls the learning rate. For an initial phase of training (until `step / args.num_iterations` reaches `1 - args.cooldown_frac`), it maintains the learning rate multiplier at 1.0 (using the `initial_lr` stored for each parameter group). For the remaining `args.cooldown_frac` portion of training, the multiplier linearly decays from 1.0 down to 0.1. 

    The `get_window_size_blocks(step)` function dynamically adjusts the attention window size for `flex_attention`. As training progresses (indicated by `x = step / args.num_iterations`), the target `window_size` (in tokens) increases linearly from a small initial value (effectively 128 tokens, due to `next_multiple_of_n`) up to a maximum derived from `1728 * 128` tokens (specifically `next_multiple_of_n(1728, n=128)` blocks). This "attention window warmup"[^2] strategy starts the model with smaller, computationally less expensive attention contexts, allowing it to first learn local dependencies. As the model learns, its contextual reach is gradually expanded, enabling it to process longer-range interactions. The actual number of blocks is returned by `get_window_size_blocks_helper`, which is decorated with `@lru_cache(1)`. This cache stores the result for a given `window_size` (in tokens), avoiding re-computation and re-creation of the tensor if the effective `window_size` (after rounding by `next_multiple_of_n`) remains the same across several steps.

4.  **Model Compilation**
    ```python
    model: nn.Module = torch.compile(model, dynamic=False)
    ```
    To maximize the model's execution speed on the GPU, the script employs `torch.compile(model, dynamic=False)`. TThis command invokes PyTorch's TorchInductor backend (the default JIT compiler for GPUs) to transform the Python-defined GPT model into optimized code. By specifying `dynamic=False`, the script signals to the compiler that the tensor shapes encountered during training will be largely static. This allows the compiler to apply more aggressive optimizations, such as fusing multiple operations into single GPU kernels and generating code specialized for the exact operations and shapes used. This compilation process introduces an initial overhead when the model is first executed, with the aim of improving subsequent runtime performance through these optimized kernels.

#### E. Pre-computation and Synchronization: Warmup and Gradient Overlap

This part of the script prepares the GPU kernels for optimal performance and implements a mechanism to overlap gradient computation with the communication needed for synchronization across GPUs.

1.  **Kernel Warmup (Lines 513-525)**
    ```python
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    for _ in range(warmup_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
        model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
        for opt in optimizers: 
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state
    ```
    Before starting the main training, the script performs a brief warmup phase. It first saves the initial states of the model and optimizers using `copy.deepcopy`. Then, for `warmup_steps` (10), it executes the core training operations—forward pass, backward pass, and optimizer steps—using random dummy data. The primary purpose of these operations is to trigger and finalize any JIT compilations by `torch.compile` and to ensure necessary CUDA kernels are compiled and cached by the GPU driver. By running these core codepaths, the script front-loads these one-time compilation overheads. To ensure these warmup iterations do not influence the actual training trajectory or benchmark timings, the script restores the model and optimizer states from the `initial_state` saved at the beginning of this phase.


2.  **Overlap Communication Setup**
    ```python
    def create_buckets(params, bucket_size_mb=25):
    """Group parameters into buckets of approximately bucket_size_mb MB each"""
        buckets = []
        current_bucket = []
        current_size = 0

        # Sort parameters by size (largest first) for better bucketing
        sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

        for param in sorted_params:
            param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

            if current_size + param_size_mb > bucket_size_mb and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size_mb
            else:
                current_bucket.append(param)
                current_size += param_size_mb

        if current_bucket:
            buckets.append(current_bucket)

        return buckets  

    # Create buckets for all parameters
    all_params = [p for p in model.parameters() if p.requires_grad]
    param_buckets = create_buckets(all_params)
    # ... (print bucket info) ...
    # Bucket state tracking
    bucket_ready_count = [0] * len(param_buckets)
    bucket_handles = [None] * len(param_buckets)
    param_to_bucket = {}

    # Map each parameter to its bucket index
    for bucket_idx, bucket in enumerate(param_buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx
    ```
    To accelerate distributed training, the script implements a mechanism to overlap gradient synchronization with the backward pass computation. This is achieved by preparing parameters for bucketed communication and then using PyTorch's gradient hooks.

    First, `create_buckets` organizes the model's trainable parameters into "buckets," each approximately 25MB in size. This bucketing strategy groups multiple smaller gradient tensors together for collective communication. Performing fewer `all_reduce` operations on these larger, aggregated buckets is generally more efficient than many operations on individual small gradients, as it amortizes the fixed overhead of launching communication calls. A mapping, `param_to_bucket`, stores the bucket index for each parameter.

    With parameters bucketed, the script registers `_gradient_hook` for every trainable parameter using `param.register_post_accumulate_grad_hook()`. The autograd engine invokes this hook for a parameter immediately after its gradient is fully computed during `model.backward()`.

    The `_gradient_hook` function then manages the readiness of gradient buckets:
    ```python
    def _gradient_hook(param: Tensor):
        """Called when a parameter's gradient is ready"""
        if param.grad is None:
            return
        bucket_idx = param_to_bucket[param]
        bucket_ready_count[bucket_idx] += 1
        if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
            bucket_grads = [p.grad for p in param_buckets[bucket_idx]]
            if len(bucket_grads) == 1:
                handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
            else:
                handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)
            bucket_handles[bucket_idx] = handle
    
    # Register hooks for all parameters
    print0("Registering bucketed gradient hooks...")
    for param in all_params:
        param.register_post_accumulate_grad_hook(_gradient_hook)

    def wait_for_gradients():
        """Wait for all gradient reductions to complete and reset bucket state"""
        for handle in bucket_handles:
            if handle is not None:
                handle.wait()
        for i in range(len(bucket_ready_count)): # Reset for next iteration
            bucket_ready_count[i] = 0
            bucket_handles[i] = None
    ```
    When `_gradient_hook` is called for a specific `param`, it first determines `bucket_idx`, the index of the bucket containing this `param`. It then increments `bucket_ready_count[bucket_idx]`. This counter tracks how many parameters within that particular bucket have had their gradients computed in the current backward pass. The logic for triggering communication lies in the condition: `if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx])`. This checks if the number of gradients now ready in this bucket equals the total number of parameters originally assigned to this bucket. If they match, the bucket is considered "full" (all its gradients are available), and an asynchronous `dist.all_reduce` operation is initiated for all gradients in that bucket. The `async_op=True` flag allows this communication to proceed in the background. The handle returned by the `all_reduce` call is stored in `bucket_handles[bucket_idx]`. The hook itself does not return a value to the autograd engine; its action is this conditional launch of an `all_reduce`.

    Finally, the `wait_for_gradients()` function, called after `model.backward()` completes, iterates through all stored `bucket_handles` and calls `handle.wait()` on each. This step ensures all launched asynchronous gradient synchronizations are finished before the optimizers apply updates. The bucket state (counters and handles) is then reset for the next training iteration.

    This setup allows the `all_reduce` for gradients of later layers (computed earlier in the backward pass) to begin and potentially overlap significantly with the computation of gradients for earlier layers, hiding communication latency and improving step time.

    Note: I discuss this bucketing strategy [in my lecture notes.](https://damek.github.io/STAT-4830/section/12/notes.html#61-data-parallelism-dp){:target="_blank"}

#### F. The Main Training Loop

This is where all components are brought together to iteratively train the model.

1.  **Initialization**
    ```python
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ```
    With model, optimizers, and the distributed environment established, the script prepares for the main training iterations. Each GPU process instantiates `distributed_data_generator` as its `train_loader`, creating a generator to stream its assigned data shards. To measure the subsequent training duration accurately, `training_time_ms` is initialized. The call to `torch.cuda.synchronize()` makes the CPU wait until all previously launched CUDA operations on the GPU have completed. Following this synchronization, the timer `t0 = time.perf_counter()` is started, ensuring the measured training time reflects core model computation.

2.  **Per-Step Loop**
    The script loops for `args.num_iterations + 1` steps.
    ```python
    train_steps = args.num_iterations
    for step in range(train_steps + 1):
        last_step = (step == train_steps)
        # ... (Validation, Training sections) ...
    ```

    *   **Validation Section**:
        This section is executed on the `last_step` or every `args.val_loss_every` steps (if `args.val_loss_every > 0`).
        ```python
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0) # Accumulate training time
            model.eval() # Switch model to evaluation mode
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
            val_loss = 0
            with torch.no_grad(): # Disable gradient calculations for validation
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step)) # Accumulate loss
            val_loss /= val_steps # Average loss
            del val_loader # Free memory
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # Average loss across GPUs
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} ...", console=True) # Log
            model.train() # Switch model back to training mode
            torch.cuda.synchronize()
            t0 = time.perf_counter() # Restart training timer
        ```
        When validation is due, the script first synchronizes CUDA operations and updates the total `training_time_ms`, effectively pausing the training timer. It then transitions the model to evaluation mode via `model.eval()`, which disables behaviors like dropout. A new `val_loader` is instantiated to serve data from the validation set.

        Within a `torch.no_grad()` context to prevent gradient computation, the script iterates `val_steps` times, accumulating the loss from the model's predictions on validation batches. After processing all validation batches, it calculates the average `val_loss` for the current GPU and then deletes `val_loader` to free resources. To obtain a global validation loss, `dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)` averages the `val_loss` values computed independently by each GPU. The `master_process` then logs this global validation loss and current timing metrics. Finally, the script switches the model back to training mode with `model.train()` and, after another `torch.cuda.synchronize()`, restarts the training timer `t0` to resume measuring only the training computation time.
    *   **Checkpointing on Last Step**:
        ```python
        if last_step:
            if master_process and args.save_checkpoint:
                # ... (save model and optimizer states) ...
            break 
        ```
        If it's the `last_step`, and if `args.save_checkpoint` is true, the `master_process` saves the model's `state_dict`, the `optimizers`' `state_dict`s, and the source `code` to a checkpoint file. The `break` statement then exits the training loop, as the last step is only for validation and checkpointing.

    *   **Training Section**:
        This is the core training operation for each step.
        ```python
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()
        wait_for_gradients() 

        for opt in optimizers: 
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups: 
            frac = min(step / 300, 1) 
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        
        for opt in optimizers: 
            opt.step()
        
        model.zero_grad(set_to_none=True) 
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms ...", console=True)
        ```
        The script first feeds a batch of `inputs` and `targets` to the model. The `model(...)` call computes the loss, and `backward()` initiates the gradient calculation. During this backward pass, gradient hooks trigger asynchronous `all_reduce` operations, overlapping communication with computation.

        Once `backward()` completes, `wait_for_gradients()` ensures all GPUs possess identical, averaged gradients. The script then adapts to the current training stage by adjusting optimizer hyperparameters: it sets the learning rate for all parameter groups via `get_lr(step)` and applies a momentum warmup for the Muon optimizer over the initial 300 steps.

        With updated hyperparameters and synchronized gradients, `opt.step()` is called for both the Adam and Muon optimizers, directing them to update their respective model parameters. Finally, `model.zero_grad(set_to_none=True)` clears gradients for the next step, and the master process logs the step's timing.

#### G. Finalization
```python
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()
```
After the training loop completes, the `master_process` logs the peak CUDA memory allocated and reserved during the run. `dist.destroy_process_group()` then cleans up the distributed training environment, releasing resources.


[^0]: More specifically, the momentum estimate of the gradient.
[^1]: God, I love quadratic convergence.
[^2]: This is a really cool technique that I hadn't seen before. You can really learn a lot of subtle hidden details from reading source code.