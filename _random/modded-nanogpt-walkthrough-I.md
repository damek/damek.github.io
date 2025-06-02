---
title: "Modded-NanoGPT Walkthrough I: initial setup, compiler config, and custom FP8 operations"
date: 2025-05-13
tags: [pytorch, transformers, optimization]
description: "Part 1 of a two part series on the modded-nanogpt repo"
---

The [`modded-nanogpt` repository](https://github.com/KellerJordan/modded-nanogpt/) is a sort of "speedrunning" exercise, designed to train a GPT-2 scale model (~124M parameters) to a target validation loss comparable to [Karpathy's `nanoGPT`](https://github.com/karpathy/nanoGPT) in a significantly reduced time. The best reported figure was about 3 minutes on 8xH100 GPUs. This is a two part series that gives a walkthrough of the [`train_gpt.py` script](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py) from the repo, focusing on the code's mechanisms for parallelism, numerical precision, architectural choices. Part I discusses the initial setup, compiler config, and custom FP8 operations. [Part II](/random/modded-nanogpt-walkthrough-ii) discusses the optimizer, parallelism, attention mechanisms, and the `GPT` class.

> I am mainly writing this to summarize my points of confusion when I read the codebase in March. It is based on an extremely long conversation I had with ChatGPT 4.5 (I was using this as an opportunity to see how the model behaved / understand the repo). I then fed that conversation to Gemini 2.5 Pro and had it help me scope a walkthrough. Writing is by default bad with LLMs, so I went through extensive rounds of feedback and reorganization. It was the only way I could write a piece this long on this topic. But I learned a lot! 

### Table of Contents

- [Initial Configuration and Environment](#initial-configuration-and-environment)
- [Custom FP8 Operations](#custom-fp8-operations)
- [Autograd Integration](#autograd-integration)
- [Up next](#up-next)


### Initial Configuration and Environment

The script begins by importing standard Python modules. An interesting thing I hadn't thought of doing before: the script logs it's own source code.
```python
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
# ... other standard imports ...
```
`sys.argv` is the path to the script itself. Reading and storing its content in the variable `code` (which is later logged if `master_process`) allows a given training run's log to be precisely associated with the exact code version that produced it. This is good practice for reproducibility in experiments and benchmarks.

**CUDA Environment Settings**

Two lines configure aspects of the CUDA environment:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
This environment variable tunes PyTorch's CUDA memory allocator. PyTorch can use `cudaMallocAsync`, an asynchronous memory allocation backend. This allocator can manage GPU memory in segments. Setting `expandable_segments:True` allows these segments to grow if a tensor allocation request slightly exceeds the capacity of existing free blocks but could be accommodated by expanding an existing segment. This can reduce the need for the allocator to request entirely new, potentially large, memory segments from the CUDA driver, which can be a synchronous and costly operation. For Transformer models, activation tensor sizes can vary, for example, due to dynamic batching, variable sequence lengths (if not strictly padded to a maximum), or intermediate tensors in attention mechanisms. Expandable segments can help manage this by reducing memory fragmentation and allocation overhead.

```python
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
```
This line performs a minimal GPU operation that also engages the autograd engine. Its purpose is to ensure the CUDA context within the PyTorch process is fully initialized. On some systems, or with specific CUDA driver and PyTorch version combinations, the first complex GPU operation can trigger latent initialization overheads or, in rare cases, issues. This small, preemptive operation helps ensure the CUDA runtime is "warmed up" before more substantial computations begin.

**Core PyTorch Imports and Compiler Configuration**
The script imports `flex_attention` from `torch.nn.attention.flex_attention`, a PyTorch component that enables more control over attention patterns. It's useful for optimizing performance of attention patterns that are not standard, like sparse or block-wise attention.

A configuration line for `torch.compile`'s Inductor backend is commented out:
```python
#torch._inductor.config.coordinate_descent_tuning = True
```
`torch.compile`  can JIT-compile `nn.Module`s or functions into optimized executables. Inductor is its default GPU backend, translating PyTorch operations into Triton or CUDA C++ code for GPU kernels. A GPU kernel is a function executed in parallel by many GPU cores. Inductor performs optimizations like operator fusion (merging multiple operations into a single kernel to reduce launch overheads and memory traffic). The `coordinate_descent_tuning=True` flag instructs Inductor to perform an extensive search for optimal kernel parameters (e.g., tile sizes, loop unrolling factors) using coordinate descent. While this could speed up the code, the tuning process itself is time-intensive (the comment suggests 30 minutes). It is disabled here, likely to prioritize faster iteration during development and for the "speedrun" context, relying on Inductor's default heuristics.

### Custom FP8 Operations: Numerical Aspects

While torch.compile can optimize standard PyTorch operations, achieving maximum performance on specific hardware like H100 GPUs can sometimes involve more direct control over numerical precision. This script takes such a step by defining custom operations for matrix multiplication using 8-bit floating-point (FP8) numbers. Matrix multiplications are computationally intensive and ubiquitous in [Transformer models](https://damek.github.io/STAT-4830/section/12/notes.html#2-transformers-anatomy-of-a-large-model) forming the core of:
1.  **Self-Attention:** Projections to Query (Q), Key (K), and Value (V) vectors ($XW_Q, XW_K, XW_V$), and the output projection ($(\text{Attention})W_O$).
2.  **Feed-Forward Networks (MLP):** Typically two linear layers ($XW_1$, $XW_2$).
3.  **Embedding/Output Layer:** The final projection to vocabulary logits ($XW_{LM\_{head}}$).

This script defines custom operations to perform some of these matrix multiplications using 8-bit floating-point (FP8) numbers. The goal is to leverage the reduced memory footprint and potentially faster computation offered by FP8 on compatible hardware like H100 GPUs. We will see later that the `CastedLinear` module, used for the LM head and potentially other linear layers, employs these custom FP8 functions.

**A. FP8 Formats and Scaling**

PyTorch tensors are in FP32 by default, which represents each number using 32 bits of precision. Often in transformer training, we use FP8 arithmetic, which only uses 8 bits per number. This change can reduce memory usage and improve computation speed on compatible hardware. 

Floating-point numbers are represented in a form like

$$\text{sign} \times \text{significand} \times 2^{\text{exponent} - \text{bias}}$$

The stored exponent bits typically represent an an *adjusted exponent*, and an *exponent bias* is a fixed integer subtracted from this adjusted exponent to get the actual `exponent_value`. The *significand* (often called the *mantissa* when referring to the fractional part of a normalized significand) determines the precision. For normalized numbers, the significand is of the form $1.f$, where $f$ is the fractional part represented by the mantissa bits.

Two common FP8 formats are E4M3 and E5M2 (definitely had to look these up!):
*   **E4M3 (`torch.float8_e4m3fn`)**: Has 1 sign bit, 4 exponent bits, and 3 mantissa bits. The 4 exponent bits can represent $2^4=16$ distinct exponent values. With a typical bias (e.g., 7 or 8 for this format), this defines the range of magnitudes. The 3 mantissa bits define the precision ($1.b_1b_2b_3$). For example, using NVIDIA's E4M3 definition (bias 7, max exponent 8), the range of positive normal numbers is roughly $[2^{-6}, (2-2^{-3}) \times 2^8]$
*   **E5M2 (`torch.float8_e5m2`)**: Has 1 sign bit, 5 exponent bits, and 2 mantissa bits. The 5 exponent bits allow $2^5=32$ patterns. With a typical bias (e.g., 15 or 16), this gives a wider dynamic range than E4M3. For example, NVIDIA's E5M2 (bias 15, max exponent 16) has a positive normal range of roughly $[2^{-14}, (2-2^{-2}) \times 2^{15}]$

E5M2 offers a wider range but less precision (fewer mantissa bits) compared to E4M3. The script uses E4M3 for forward pass activations/weights and E5M2 for gradients, where wider range might be more beneficial.

This script uses E4M3 for forward pass activations and weights, and E5M2 for gradients, where the wider dynamic range of E5M2 can be more suitable for accommodating potentially larger gradient values. Due to the limited range and precision, values must be scaled before conversion to FP8 to fit within the representable range and preserve information.

With these FP8 formats in mind, let's look at how the script implements the forward pass for an FP8 matrix multiplication.

**B. `mm_op`: Forward Pass**
This function, named `mm_op`, defines the custom forward operation for computing $Y = XW^T$ using FP8 arithmetic.
```python
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous() # Contiguous tensors are more efficient to process.
        # x_s, w_s are per-tensor scales for X and W
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn) # X_fp8 = X / x_s
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn) # W_fp8 = W / w_s
        
        # Computes (X_fp8 W_fp8^T) * x_s * w_s
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32), 
            scale_b=x.new_tensor(w_s, dtype=torch.float32), 
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)
```
Here is what's going on:
- Inputs `x` (activations) and `w` (weights) are scaled by `x_s^{-1}` and `w_s^{-1}$` respectively, then cast to `torch.float8_e4m3fn`. 
- Then we apply the function: `torch._scaled_mm(A, B, out_dtype, scale_a, scale_b)`. 
    - If `A` and `B` are FP8 tensors, this operation computes $(A  B) \times \text{scale_a} \times \text{scale_b}$ where the product $A B$ is internally accumulated (perhaps in higher precision) and then scaled and cast to `out_dtype`. So, the effective computation is 

    $$((X/x_s)_{FP8} (W/w_s)_{FP8}^T) \times x_s \times w_s \approx XW^T$$

    - The output `out` is in `bfloat16`, yet another floating point format, that we won't go into.
    - `use_fast_accum=True` can enable hardware accumulators that might use lower internal precision for speed. The factor `grad_s` is for the backward pass. `x_f8` and `w_f8` are saved.

**C. `mm_op.register_fake`: A "Meta" Implementation for Tracing**

After defining the custom forward operation `mm_op`, the script registers a "fake" implementation for it. This is a mechanism used by PyTorch's JIT compilation tools, particularly `TorchDynamo` (the Python frontend for `torch.compile`).
```python
@mm_op.register_fake
def _(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float): # Matched signature
    # Assertions ensure input metadata (ndim, shape, device, contiguity)
    # matches expectations for a 2D matrix multiplication.
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1] # Inner dimensions must match for X @ W.T
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    

    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
```
When `TorchDynamo` traces a model containing `mm_op`, it doesn't necessarily execute the full, potentially complex, `@torch.compile`d `impl` function of `mm_op` with actual data. Instead, it can run this registered `_` fake function with "fake tensors." These fake tensors carry metadata (like shape, dtype, device) but not actual numerical data.

The purpose of this fake implementation is to allow the tracer to:
1.  Verify that the custom operation can handle inputs with the given metadata.
2.  Determine the metadata (shape, dtype, etc.) of the outputs that the custom operation would produce.

This information allows `TorchDynamo` to construct an accurate graph of operations and their dependencies. Based on this graph, Inductor (the backend) can generate optimized code. The fake function provides a lightweight way to simulate the op's behavior at the metadata level, without the overhead of running the real computation or needing specialized hardware (like FP8 support) during the tracing phase itself. 

**D. `mm_backward_op`: Backward Pass**

When defining a custom forward operation like `mm_op` that involves specific numerical representations (FP8) and scaling, PyTorch's automatic differentiation engine needs to be explicitly provided with the corresponding backward logic. If our forward operation is $Y = XW^T$, and $L$ is the overall loss function, autograd works by propagating $\frac{\partial L}{\partial Y}$ backward and requires functions that can compute the terms needed for $\frac{\partial L}{\partial X}$ and $\frac{\partial L}{\partial W}$. These are vector-Jacobian products (VJPs). For a matrix multiplication $Y=XW^T$, the relationships are (more on Jacobians [here](https://damek.github.io/STAT-4830/section/5/notes.html#extending-to-higher-dimensions-the-jacobian)):

$$ \begin{align*}
\frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} W \\
\frac{\partial L}{\partial W} &= \left(\frac{\partial L}{\partial Y}\right)^T X 
\end{align*} $$

The `mm_backward_op` function implements these relationships, accounting for the FP8 quantization and scaling used in the forward pass `mm_op`.

```python
@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor): # grad is dL/dY
        assert grad.is_contiguous()
        # These are the original scales from the forward pass, not "inverse" in the sense of 1/scale.
        # They will be used by _scaled_mm to correctly scale the FP8 products.
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32) 
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2) # (dL/dY)_fp8 = (dL/dY) / grad_s

        # Compute dL/dX = (dL/dY) @ W
        # This is ((dL/dY / grad_s)_fp8 @ (W / w_s)_fp8) * grad_s * w_s
        grad_x = torch._scaled_mm(
            grad_f8,                                # Input1: (dL/dY)_fp8
            w_f8.T.contiguous().T,                  # Input2: (W/w_s)_fp8
            out_dtype=torch.bfloat16,               # dL/dX output precision
            scale_a=grad_inv_s, # Scale for grad_f8 input to _scaled_mm, effectively grad_s
            scale_b=w_inv_s,    # Scale for w_f8 input to _scaled_mm, effectively w_s
            use_fast_accum=False, # Potentially more precise accumulation
        )

        # Compute dL/dW = (dL/dY).T @ X
        # This is ((X / x_s)_fp8.T @ (dL/dY / grad_s)_fp8) * x_s * grad_s, then outer transpose
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),                    # Input1: (X/x_s)_fp8.T
            grad_f8.T.contiguous().T,               # Input2: (dL/dY)_fp8 
            out_dtype=torch.float32,                # dL/dW output precision
            scale_a=x_inv_s,    # Scale for x_f8.T input, effectively x_s
            scale_b=grad_inv_s, # Scale for grad_f8 input, effectively grad_s
            use_fast_accum=False,
        ).T 
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)
```

The `impl` function within `mm_backward_op` takes the incoming gradient `grad` (which is $\frac{\partial L}{\partial Y}$, the gradient of the loss $L$ with respect to the output $Y$ of the forward `mm_op`), and the FP8 tensors `x_f8` and `w_f8` saved from the forward pass. It also receives the original scaling factors `x_s`, `w_s`, and `grad_s`.

First, the incoming gradient `grad` is prepared for FP8 computation:
```python
grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
```
This scales `grad` by `grad_s^{-1}` and converts it to the E5M2 FP8 format, which we can denote as $(\frac{\partial L}{\partial Y})\_{FP8S} = \left(\frac{1}{\text{grad\_s}}\frac{\partial L}{\partial Y}\right)\_{FP8}$. The script also creates tensor versions of the original scales, `x_s`, `w_s`, `grad_s`, naming them `x_inv_s`, `w_inv_s`, `grad_inv_s`. This is slightly bad notation, since despite the `_inv_s` suffix, these hold the original scale values.

Next, `grad_x` (representing $\frac{\partial L}{\partial X}$) is computed. The target mathematical operation is $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W$. The code implements this using `torch._scaled_mm` as:
```python
grad_x = torch._scaled_mm(
    grad_f8,             # A_fp8 = (dL/dY)_fp8s
    w_f8.T.contiguous().T, # B_fp8 = (W/w_s)_fp8
    out_dtype=torch.bfloat16,
    scale_a=grad_inv_s,    # S_A = grad_s
    scale_b=w_inv_s,       # S_B = w_s
    use_fast_accum=False,
)
```
The `torch._scaled_mm` operation, with FP8 inputs $A_{FP8}$, $B_{FP8}$ and scales $S_A$, $S_B$, calculates a result approximately equal to $(A_{FP8} \cdot S_A) (B_{FP8} \cdot S_B)$. Substituting our terms:

$$ \text{grad_x} \approx \left( \left(\frac{1}{\text{grad_s}}\frac{\partial L}{\partial Y}\right)_{FP8} \cdot \text{grad_s} \right) \left( \left(\frac{W}{w_s}\right)_{FP8} \cdot w_s \right) $$

This approximately reconstructs the desired $\frac{\partial L}{\partial Y} W$. The result `grad_x` is stored in `bfloat16`.

Then, `grad_w` (representing $\frac{\partial L}{\partial W}$) is computed. The target is $\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial Y})^T X$. The code computes $X^T \frac{\partial L}{\partial Y}$ and then transposes:
```python
grad_w = torch._scaled_mm(
    x_f8.T.contiguous(),       # A_fp8 = (X/x_s)_fp8^T
    grad_f8.T.contiguous().T,  # B_fp8 = (dL/dY)_fp8s
    out_dtype=torch.float32,
    scale_a=x_inv_s,           # S_A = x_s
    scale_b=grad_inv_s,        # S_B = grad_s
    use_fast_accum=False,
).T
```
The computation within `_scaled_mm` is:

$$ \left( \left(\frac{X}{x_s}\right)_{FP8}^T \cdot x_s \right) \left( \left(\frac{1}{\text{grad_s}}\frac{\partial L}{\partial Y}\right)_{FP8} \cdot \text{grad_s} \right) \approx X^T \frac{\partial L}{\partial Y} $$

The final `.T` transposes this result to yield $\frac{\partial L}{\partial W}$. This gradient for the weights is stored in `float32`. Using a higher precision like `float32` for weight gradients is common practice since optimizers accumulate gradient statistics over time and that can cause a loss of precision. The activation gradients (`grad_x`), which flow backward to earlier layers, are kept in `bfloat16`; this attempts to balance precision with memory and computational efficiency.

**E. Autograd Integration**

Since `mm_op` (and its backward logic `mm_backward_op`) are custom operations defined outside PyTorch's standard library of differentiable functions, we need to explicitly tell PyTorch's automatic differentiation engine (autograd) how to handle them. This is achieved by defining two helper functions, conventionally a `backward` function and a `setup_context` function (or `save_for_backward` if subclassing `torch.autograd.Function`), and then registering them.

The `setup_context` function is called by PyTorch during the *forward pass* of `mm_op`. Its role is to save any tensors or data from the forward pass that will be needed later to compute gradients during the *backward pass*.
```python
def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    # mm_op inputs = (x, w, x_s, w_s, grad_s)
    # mm_op output = (out, x_f8, w_f8)
    *_, x_s, w_s, grad_s = inputs # Unpack scales from mm_op's inputs
    _, x_f8, w_f8 = output       # Unpack FP8 tensors from mm_op's outputs
    
    ctx.save_for_backward(x_f8, w_f8) # Save these tensors onto the context object
    ctx.scales = x_s, w_s, grad_s     # Scales can also be saved on ctx
    ctx.set_materialize_grads(False)  # Optimization: don't create grad tensors until needed
```    

The `ctx` object of type `torch.autograd.function.FunctionCtx` acts as a communication channel between the forward and backward passes of the custom operation.

The `backward` function is called by PyTorch during the *backward pass*. It receives the `ctx` object (containing the saved items) and the gradient of the loss with respect to the output of `mm_op`. Its job is to compute the gradients of the loss with respect to the *inputs* of `mm_op`.
```python
def backward(ctx, grad_out: Tensor, *_): # grad_out is dL/d(out) from mm_op
    x_f8, w_f8 = ctx.saved_tensors         # Retrieve saved FP8 tensors
    x_s, w_s, grad_s = ctx.scales          # Retrieve saved scales
    
    # Call the custom C++ op for backward computation
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    
    # Return gradients for each input of mm_op: (x, w, x_s, w_s, grad_s)
    # Since x_s, w_s, grad_s are floats and not Tensors requiring grads,
    # their gradients are None.
    return grad_x, grad_w, None, None, None
```

Finally, these two functions are registered with `mm_op`:
```python
mm_op.register_autograd(backward, setup_context=setup_context)
```
This line informs PyTorch that whenever `mm_op` is used in a computation graph where gradients are required, it should use the provided `setup_context` during the forward pass and the provided `backward` function during the backward pass.

### Up next

I planned to write this in one post, but ran out of time. In [part II](/random/modded-nanogpt-walkthrough-ii) of this post, I will introduce the Muon optimizer, the GPT-2 model architecture, and discuss the parallelism strategies for running the code across multiple GPUs.