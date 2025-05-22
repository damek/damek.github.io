---
title: "Modded-NanoGPT Walkthrough II: Muon Optimizer, Model Architecture, and Parallelism"
date: 2025-05-22
tags: [pytorch, transformers, optimization, distributed]
description: "Part II: Muon optimizer, GPT architecture details, and distributed training in modded-nanogpt."
---

In [Part I](/random/modded-nanogpt-walkthrough-I) of this walkthrough, we covered the initial setup, compiler configurations, and custom FP8 operations within the `modded-nanogpt` repository's `train_gpt.py` script. This second part continues the walkthrough of `train_gpt.py`. We will look at the Muon optimizer, we will dissect the specific GPT model architecture implemented, and finally, we will look at the distributed training strategies that enable its fast performance on multiple GPUs.

### The Muon Optimizer: Iterative Orthogonalization for Updates (Lines 105-215)

The `train_gpt.py` script introduces a custom optimizer called `Muon`, that is specifically used with the matrix layers of the transformer model. (For the nonmatrix layers, they use an Adam method.) In short, Muon replaces the matrix blocks of the gradient[^0] with a new matrix with better conditioning and the same row/column space. This is achieved by applying an iterative algorithm called the Newton-Schulz.

Why do they do this? From my read of the literature (up to May 22, 2025), there has been no strong theoretical justification for doing so. Although we can realize it as a variant of gradient descent in a block spectral norm, we don't know why it's good to do gradient descent in the spectral norm for transformer models. 🤷

**A. `zeropower_via_newtonschulz5`: Orthogonalizating the gradient (Lines 107-126)**

The function `zeropower_via_newtonschulz5` applies Newton-Schulz to an input matrix $G$. Classically, the method was designed to do the following: 

> If $G$ has a singular value decomposition (SVD) $G = U \Sigma V^T$, this iteration (when properly initialized) converges quadratically to a matrix $G' \approx U I' fV^T$. In this expression, $I'$ is a diagonal matrix with entries of 1 where $\Sigma$ had non-zero singular values, and 0 otherwise. This process yields an (approximately) orthogonal matrix with the same row and column space as $G$.

The method in the code is slightly different. It instead modifies the method so that (1) the singular values near zero become larger more quickly, but the limiting singular values (empirically) reach the interval between .5 and 1.5. This seems to work OK.

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

**B. The `Muon` Optimizer Class (Lines 128-215)**

The `Muon` class is defined by inheriting from `torch.optim.Optimizer`. It applies a specific update rule to 2D matrix parameters.

For a given 2D parameter matrix $W\_t$ at step $t$, and its gradient $\nabla L(W\_t)$ (computed across a batch of data), the Muon optimizer computes the updated matrix $W\_{t+1}$ through the following stages:

1. *Momentum.* A momentum buffer, $\text{buf}\_t$, is updated using the current gradient and the previous buffer state $\text{buf}\_{t-1}$. Let $m$ be the momentum factor (e.g., `group["momentum"]`). The update is:

    $$ \text{buf}_t = m \cdot \text{buf}_{t-1} + (1-m) \cdot \nabla L(W_t) $$

    This is achieved by the line `buf.lerp_(g, 1 - group["momentum"])` where `buf` initially holds $\text{buf}_{t-1}$ and `g` is $\nabla L(W_t)$.

2. *Nesterov.* The gradient used for the update, $g_{\text{eff}}$, is determined based on whether Nesterov momentum is used.
    *   If Nesterov momentum is enabled (`group["nesterov"]` is true):

    $$ g_{\text{eff}} = (1-m) \cdot \nabla L(W_t) + m \cdot \text{buf}_t $$

    This is achieved by `g = g.lerp_(buf, group["momentum"])`.
    *   If Nesterov momentum is not enabled:

    $$ g_{\text{eff}} = \text{buf}_t $$

3.  *Newton-Schulz.* The effective gradient $g\_{\text{eff}}$ is then processed by the Newton-Schulz iteration to produce an orthogonalized version, $g\_{\text{ortho}}$:

    $$ g_{\text{ortho}} = \texttt{zeropower_via_newtonschulz5}(g_{\text{eff}}, \text{steps}) $$

4.  *Update.* The final parameter update incorporates this $g_{\text{ortho}}$. Let $\eta$ be the learning rate (`group["lr"]`). The learning rate is scaled by a factor $\alpha_{shape}$ dependent on the parameter's dimensions (rows $R$ and columns $C$):

    $$ \alpha_{shape} = \sqrt{\max\left(1, \frac{R}{C}\right)} $$

    The parameter $W_t$ is updated to $W_{t+1}$ as:

    $$ W_{t+1} = W_t - (\eta \cdot \alpha_{shape}) \cdot g_{\text{ortho}} $$

    This is implemented by `p_world.add_(g_world.view_as(p_world), alpha= -group["lr"] * scale_factor)`.

The `__init__` method of the `Muon` class sets up standard optimizer attributes like learning rate and momentum. A notable detail is that parameters are grouped by their total number of elements (`p.numel()`). For each distinct size, an `update_buffer` tensor of shape `(world_size, size)` is created on the CUDA device using `bfloat16` precision. `update_buffer_views` then provides per-rank slices into this shared buffer. This structure is likely to facilitate the efficient gathering (`dist.all_gather_into_tensor`) of the processed $g_{ortho}$ updates from all GPUs if Muon itself shards the processing of its parameter list across GPUs.

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
The `step()` method, decorated with `@torch.no_grad()` to disable gradient tracking for the update operations, performs the parameter update. It iterates through parameters in chunks, distributing the processing across `world_size` GPUs. The loop `for base_i in range(len(params))[::self.world_size]` iterates with a step size equal to `world_size`. Each GPU, identified by `self.rank`, processes the parameter `params[base_i + self.rank]`.

For each parameter `p` and its gradient `g` (which should have already been averaged across GPUs *before* this `optimizer.step()` call is made):
1.  The momentum buffer `buf` is updated using the current gradient `g` and the previous buffer state. The line `buf.lerp_(g, 1 - group["momentum"])` implements the update $ buf_t = buf_{t-1} \cdot m + g \cdot (1-m) $.
2.  If Nesterov momentum is enabled (`group["nesterov"]` is true), the effective gradient `g` is further adjusted using the updated buffer: `g.lerp_(buf, group["momentum"])` implements $ g_{eff} = g_{orig} \cdot (1-m) + buf_t \cdot m $. Otherwise, $g_{eff}$ is simply `buf`.
3.  This effective gradient $g_{eff}$ is then processed by `zeropower_via_newtonschulz5` to obtain an orthogonalized update, which is subsequently flattened.

If a GPU has no parameter assigned to it in the current chunk (i.e., `base_i + self.rank >= len(params)`), it still contributes to the collective communication. In this case, its contribution `g` is taken from its view of the `update_buffer` (`update_buffer_views[self.rank]`).

The `update_prev()` helper function is called to apply updates from the *previous* chunk of parameters once their synchronized orthogonalized gradients are available.

The line `handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)` gathers the locally computed orthogonalized update `g` from the current GPU along with updates from all other GPUs into the shared `update_buffer`. The `async_op=True` flag allows this communication to happen in the background. `params_world` is a slice of the parameters corresponding to the updates currently being gathered.

Inside `update_prev()`:
*   `handle.wait()` ensures that the `all_gather` operation for the current chunk's updates is complete.
*   Then, each parameter `p_world` in the processed shard (`params_world`) is updated. It uses its corresponding orthogonalized update `g_world` (obtained from `update_buffer_views`). The update rule `p_world.add_(g_world.view_as(p_world), alpha=effective_lr)` subtracts the scaled orthogonalized gradient from the parameter. The `effective_lr` includes the group's learning rate and the scaling factor based on the parameter's aspect ratio: `-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5`.

### GPT Model Architecture: Component Details (Lines 217-389)
The model implemented in `train_gpt.py` is a decoder-only Transformer, with several specific architectural choices.

*(Mathematical Detail Level: RMSNorm - Medium; RoPE - High; Attention mechanism - High; Logit Softcapping - Medium)*

**A. Core Building Blocks**

1.  **Normalization: `norm()` (Lines 217-218)**
    ```python
    def norm(x: Tensor):
        return F.rms_norm(x, (x.size(-1),))
    ```
    This function applies Root Mean Square Layer Normalization (RMSNorm). For an input vector $x \in \mathbb{R}^n$, RMSNorm is defined as:
    $$ \text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^n x_j^2 + \epsilon}} \cdot \gamma_i $$
    Here, $\gamma_i$ represents a learnable scale parameter, and $\epsilon$ is a small constant for numerical stability. `F.rms_norm` handles these details. This normalization variant omits the mean-centering step of standard LayerNorm, which can sometimes simplify computation.

2.  **`CastedLinear` (Lines 220-236)**
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
    The `CastedLinear` layer provides optional FP8 support. Its `reset_parameters` method implements a custom weight initialization. The standard deviation for initialization is $\text{std} = 0.5 \cdot (\text{in\_features})^{-0.5}$. Weights $W$ are drawn from a uniform distribution $U[-\sqrt{3} \cdot \text{std}, \sqrt{3} \cdot \text{std}]$. The `forward` pass utilizes the custom `mm_op` (discussed in Part I) for FP8 matrix multiplication if `use_fp8` is true and the model is in training mode. Otherwise, it defaults to a standard `F.linear` operation.

3.  **`Rotary` Embeddings (Lines 238-255)**
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
    This module implements Rotary Position Embeddings (RoPE). A "half-truncate RoPE" modification is used: `angular_freq` is constructed such that its latter half is zero. This means that rotations are only applied to the first `dim//4` pairs of features (i.e., the first `dim//2` features if `dim` is the feature dimension being rotated, typically `head_dim`). The rotation angles $\theta$ are computed as an outer product of positions `t` and `angular_freq`: $\theta_{pos, j} = \text{pos} \cdot \text{angular\_freq}_j$. The `cos` and `sin` of these angles are precomputed and stored as non-persistent buffers.
    In the `forward` method, an input tensor `x_BTHD` (e.g., query or key, with shape Batch, Time, Heads, Dim_per_head) is split into two halves, $x_1$ and $x_2$, along its last dimension (the head dimension $D_h$). The rotation is applied as:
    $$ x'_{1} = x_1 \cos \theta_{pos} - x_2 \sin \theta_{pos} $$
    $$ x'_{2} = x_1 \sin \theta_{pos} + x_2 \cos \theta_{pos} $$
    This corresponds to multiplying a complex number $x_1 + i x_2$ by $e^{i\theta_{pos}}$. (Note: the code snippet in the scoping doc had `+ x2 sin` and `-x1 sin`, which corresponds to rotation by $-\theta$. The actual effect on relative attention scores is similar.)

4.  **`CausalSelfAttention` (Lines 257-287)**
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
    In `__init__`: A merged weight tensor `self.qkv_w` of shape `(3, num_heads * head_dim, model_dim)` is used for Q, K, V projections. Learnable `lambdas` are initialized for mixing value embeddings (`ve`) into `v`. The output projection `c_proj` has its weight zero-initialized.
    In `forward`: The assertion `B == 1` indicates that `flex_attention` is used here with an effective batch size of 1 sequence. Q, K, V are derived from the input `x` using `qkv_w`. For an input $X \in \mathbb{R}^{B \times T \times \text{dim}}$, and $W_{QKV}$ (reshaped `qkv_w`), the projection $X W_{QKV}^T$ yields a tensor that is then viewed and chunked into $Q, K, V$, each of shape $\mathbb{R}^{B \times T \times \text{num\_heads} \times \text{head\_dim}}$.
    $Q$ and $K$ are normalized using `norm()` (QK Norm). RoPE is applied to $Q$ and $K$. If `ve` (value embeddings) are provided, they are mixed into $V$ using the learnable `lambdas`. The `flex_attention` function is called; note that $Q, K, V$ are transposed to the shape `(B, num_heads, T, head_dim)` which is a common convention for attention implementations. A custom `scale=0.12` is used for scaling dot products, instead of the typical $1/\sqrt{d_k}$. The `block_mask` argument controls the attention pattern (causality, sliding window). The output `y` is transposed back, made contiguous, reshaped, and finally passed through `c_proj`.

5.  **`MLP` (Lines 289-299)**
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

6.  **`Block` (Lines 301-311)**
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
    This module defines a single Transformer block. Attention is skipped if `layer_idx == 7` (the 8th layer). Learnable `lambdas` are used to mix the block's current input `x` with `x0`, which is the initial token embedding passed through all layers. The computation flow is: input mixing, then an optional attention sublayer followed by a residual connection, and an MLP sublayer followed by a residual connection. Normalization (`norm(x)`) is applied to the input of both the attention and MLP sublayers (pre-norm style).

**C. The `GPT` Model Assembly (Lines 316-389)**
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
The `GPT` class `__init__` method sets up:
*   `self.embed`: a standard token embedding layer.
*   `self.value_embeds`: a list of three separate `nn.Embedding` layers. These provide embeddings from the `input_seq` that are subsequently mixed into the `v` tensor in specific attention layers.
*   `self.blocks`: a list of `Block` modules, forming the stack of Transformer layers.
*   `self.lm_head`: the final linear layer projecting to logits. It is a `CastedLinear` instance configured to use FP8, with specific scaling factors (`x_s`, `w_s`, `grad_s`). Its weight is zero-initialized. The vocabulary size is padded to the nearest multiple of 128 using `next_multiple_of_n`.
*   `self.skip_weights`: learnable parameters for U-Net style skip connections between layers, initialized to ones. There are `num_layers // 2` such weights, as `num_layers` is asserted to be even.

The `create_blockmasks` method is responsible for generating the attention masks for `flex_attention`. It defines a `BLOCK_SIZE` of 128. It identifies document boundaries based on the token ID 50256, which presumably acts as a document separator. The function computes masks that enforce causality and respect document structure, while also implementing sliding window attention. The `sliding_window_num_blocks` input argument dynamically controls the size of this attention window. The method returns two `BlockMask` objects, `long_bm` and `short_bm`, which represent different attention window configurations (e.g., a standard window and a halved window).

In the `forward` method:
*   `ve` (value embeddings) are computed from `input_seq` using the `self.value_embeds` layers. These are then assigned to specific layers in a "012 ... 012" pattern: the first three and last three layers receive these value embeddings, while intermediate layers receive `None`.
*   `long_bm` and `short_bm` attention masks are created by calling `self.create_blockmasks`. A specific sequence of these masks (`block_masks`) is then assigned to the layers, determining whether each layer uses a long or short attention window.
*   `x = x0 = norm(self.embed(input_seq)[None])`: the `input_seq` is embedded, normalized, and an unsqueezed batch dimension is added (as `B=1` is assumed). `x0` stores this initial embedding for use in each `Block`'s input mixing step.
*   The U-Net style skip connection logic is implemented:
    *   For the first `n = num_layers // 2` layers (`i < n`), the output `x` of each block is stored in a list `skip_connections`.
    *   For the subsequent `n` layers (`i >= n`), the output from a corresponding earlier layer is popped from `skip_connections` (LIFO order means layer `n-1`'s output is popped first, then `n-2`'s, etc.), multiplied by a learnable `skip_weight`, and added to the current `x` before it's processed by the block. The indexing `skip_weights[i - n]` ensures that layer `n` (the first layer in the second half) is combined with the output of layer `n-1` (the last layer of the first half) using `skip_weights`, and so on, forming symmetric connections.
*   After processing through all blocks, the final `x` is normalized.
*   `logits = self.lm_head(x).float()`: logits are computed using the FP8-enabled `lm_head` and then cast to float.
*   Logit softcapping is applied: `logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))`. This formula, $L_{capped} = 30 \cdot \sigma(L_{orig} / (7.5 \sqrt{d_{model}}))$, where $\sigma$ is the sigmoid function, serves to cap the logits. This is similar to applying a scaled tanh function, $30 \cdot \text{tanh}(L_{orig} / (15 \sqrt{d_{model}}))$.
*   Finally, the cross-entropy loss is computed between the predicted logits and the `target_seq`.

### Parallelism and Distributed Training

*(Mathematical Detail Level: Medium for describing operations like all_reduce)*

The script is designed for distributed training, primarily using data parallelism.

**A. Overall Training Parallelism Strategy**
*(An ASCII diagram illustrating the flow of data and gradient synchronization will be beneficial here in the blog post, similar to the one drafted in the scoping document.)*
The general flow for each training step across multiple GPUs is:
1.  **Data Loading**: Each GPU (`GPU_i`) receives a unique shard of the current data batch (`data_shard_i`) from the `distributed_data_generator`.
2.  **Forward Pass**: Each `GPU_i` computes `loss_i = model(inputs_i, targets_i, ...)` using its local copy of the model and its data shard.
3.  **Backward Pass**: Each `GPU_i` computes `loss_i.backward()`, resulting in local gradients (`local_grad_i`) for all model parameters.
4.  **Gradient Averaging**: For every parameter `p` in the model, `dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)` is called. This averages the gradients across all GPUs. After this step, `p.grad` is identical on all GPUs.
5.  **Optimizer Steps**:
    *   `optimizer1.step()` (Adam): Each `GPU_i` updates its local copy of parameters handled by Adam (embeddings, scalars, head parameters) using the globally averaged gradients.
    *   `optimizer2.step()` (Muon): For parameters handled by Muon (hidden matrices), the `Muon.step()` logic is executed. This involves each `GPU_i` processing its assigned shard of these parameters, applying momentum and orthogonalization to the globally averaged gradients for that shard, then using `dist.all_gather_into_tensor` to collect all orthogonalized updates, and finally applying the relevant updates to its local parameter copies.

**B. Distributed Data Loading (Lines 391-418)**

1.  **`_load_data_shard()` (Lines 393-401)**
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
    This function reads a single data shard from a binary file. It checks a header for a magic number and version compatibility. It allocates a `torch.uint16` tensor in pinned memory (`pin_memory=True`) for faster CPU-to-GPU transfers and reads data directly into the tensor's underlying NumPy array using `f.readinto()`.

2.  **`distributed_data_generator()` (Lines 403-418)**
    ```python
    def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        assert batch_size % world_size == 0
        local_batch_size = batch_size // world_size
        file_iter = iter(files) 
        tokens, pos = _load_data_shard(next(file_iter)), 0
        while True:
            if pos + batch_size + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            buf_start = pos + rank * local_batch_size
            buf_end = pos + (rank + 1) * local_batch_size + 1 # Shard for this rank, +1 for target
            buf = tokens[buf_start:buf_end] 
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) 
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) 
            pos += batch_size 
            yield inputs, targets
    ```
    This generator handles loading and distributing data shards. It finds all files matching `filename_pattern`. The `batch_size` argument appears to represent the global batch size, which is then divided by `world_size` to get `local_batch_size` for each GPU. It iterates through files, loading a new one when the current `tokens` tensor is exhausted. Each GPU (identified by `rank`) gets its unique slice of data: `tokens[pos + rank * local_batch_size : pos + (rank + 1) * local_batch_size + 1]`. It takes `local_batch_size + 1` tokens to form `inputs` (all but the last token) and `targets` (all but the first token). These are transferred to the CUDA device asynchronously using `non_blocking=True`. The global position `pos` is advanced by the global `batch_size`.

**C. Environment Setup (Lines 430-438)**
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
This section initializes the distributed training environment. It retrieves `RANK` (global rank of the current process), `WORLD_SIZE` (total number of processes), and `LOCAL_RANK` (rank of the process on the current node) from environment variables, which are typically set by a launcher like `torchrun`. It asserts that `world_size` is 8. It then sets the current CUDA device for the process and initializes the PyTorch distributed process group using the `nccl` backend, which is optimized for NVIDIA GPU communication. `dist.barrier()` ensures all processes synchronize at this point. The `master_process` flag is set to true only for rank 0, which is conventionally used for tasks like logging or saving checkpoints.

**D. Specific Distributed Operations (from main training script part)**

1.  **Initial Model Sync (Line 463)**
    ```python
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    ```
    After the model is instantiated on each GPU, this loop iterates through all model parameters. For each parameter, `dist.broadcast(param.detach(), 0)` sends the parameter tensor from rank 0 to all other ranks in the distributed group. This ensures that all model replicas start with the exact same initial weights. `param.detach()` is used because we are broadcasting the tensor data, not its computation history.

2.  **Gradient Synchronization (Line 547)**
    ```python
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    ```
    This loop is executed after `loss.backward()` has computed gradients locally on each GPU (based on its shard of data). For each parameter, `dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)` performs an all-reduce operation. The `op=dist.ReduceOp.AVG` specifies that the gradients from all GPUs should be averaged. After this operation, the `.grad` attribute of each parameter on every GPU will hold the same averaged gradient value. This is the core mechanism of data parallelism.

3.  **Validation Loss Sync (Line 532)**
    ```python
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    ```
    During validation, each GPU computes a local `val_loss` based on its shard of the validation data. This line averages these local validation losses across all GPUs. The result is that `val_loss` on every GPU becomes the same global average validation loss.

This concludes the walkthrough of the Muon optimizer, model architecture, and distributed training aspects of `train_gpt.py`. The remaining sections of the script, starting from line 420, orchestrate the main training loop, handle logging, define hyperparameters, and manage learning rate schedules, integrating the components discussed.

[^0]: More specifically, the momentum estimate of the gradient.
[^1]: God, I love quadratic convergence.