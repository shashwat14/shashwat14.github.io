---
title: "Muon Optimizer For Dummies"
date: 2025-12-01
draft: false
summary: "A deep dive into Muon, the optimizer that trains models 35% faster by preserving rare learning directions that AdamW misses."
ShowToc: true
math: true
---

# Introduction

For most of us, AdamW has been good enough. But Keller Jordan questioned it, found some issues, and built Muon. It achieves a **1.35x speedup** on the NanoGPT benchmark. That means 35% fewer tokens to reach the same performance, or equivalently, reaching lower loss with the same data.

The name stands for **M**oment**U**m **O**rthogonalized by **N**ewton-Schulz. Muon orthogonalizes the momentum (accumulated gradients) using Newton-Schulz iteration. It is only applicable for **hidden layer** 2D weight matrices. Embedding layers and the output classifier head should still use AdamW.

Here's what we'll cover:
1. The problem with AdamW
2. The fix: orthogonalizing momentum
3. Making it fast with Newton-Schulz
4. The "cursed" quintic polynomial
5. The full algorithm

Let's dive in.

# The Problem with AdamW

## A Quick Recap

Let's start with a quick review of AdamW. It's simply the Adam optimizer + weight decay.

```python
for each parameter tensor θ:

    # 1. Decoupled weight decay as separate step
    if weight_decay > 0 and θ should_decay:
        θ = θ * (1 - lr * weight_decay)

    # 2. Standard Adam update on the shrunken θ
    g = grad(θ)

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g * g)

    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)

    adam_update = m_hat / (sqrt(v_hat) + eps)

    θ = θ - lr * adam_update
```

We are not going to jump into the intuition behind AdamW, but the key takeaway is that all the math here is **elementwise**. 

An interesting thing about deep learning is that there is quite a bit of matrix algebra used in both the forward pass and backward pass. For example, we say $y = xW^\top$. Similarly, in the backward pass, we compute $\frac{\partial L}{\partial W}$.

Looking at AdamW at a high level, we have a gradient that we use to update the matrix. We first perform weight decay. Now unlike SGD, we don't just apply this gradient as-is. We compute a few quantities like first and second moments and then use them to create the update. Note that at no point did we use any matrix algebra operations. Everything is elementwise.

This raises a question: why even bother? Why do we need to use matrix operations, and why are elementwise operations not enough?

## The Ill-Conditioned Momentum Problem

The theoretical justification is that we are ignoring correlations between different directions in the weight space.

The empirical argument is more interesting. When we track the momentum (accumulated gradients) updates, a strange pattern emerges: this update matrix has a **high condition number**. 

The condition number of a matrix is the ratio of its largest singular value to its smallest: $\kappa = \sigma_{max} / \sigma_{min}$. A high condition number means some directions in the update are much larger than others. The updates for all weights end up dominated by just a few directions, while other "rare directions" have small magnitude but are nevertheless important for learning.

## Why Are Gradients Ill-Conditioned?

But why is the update matrix ill-conditioned in the first place? For that, we need to look even more under the hood and understand how these gradients are computed. Consider a simple feedforward layer:

Let $T$ be the sequence length (number of tokens) and $d_{model}$ be the hidden dimension. For a linear layer projecting from $d_{model}$ to $d_{out}$:

- $x \in \mathbb{R}^{T \times d_{model}}$ is the input
- $W \in \mathbb{R}^{d_{out} \times d_{model}}$ is the weight matrix
- $y \in \mathbb{R}^{T \times d_{out}}$ is the output

The forward pass is:
$$y = xW^\top$$

During backpropagation, the gradient with respect to the weights is computed as:
$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial y}\right)^T x \in \mathbb{R}^{d_{out} \times d_{model}}$$

Here, $\frac{\partial L}{\partial y} \in \mathbb{R}^{T \times d_{out}}$ is the gradient flowing from the upper layers, and $x \in \mathbb{R}^{T \times d_{model}}$ is the input to this layer. If either of these matrices has a high condition number, their product (the weight gradient) will also tend to have a high condition number. So either the input activations $x$ or the upstream gradients $\frac{\partial L}{\partial y}$ (or both) are ill-conditioned, leading to an ill-conditioned weight gradient.

# The Fix: Orthogonalize the Momentum

We can think of matrices as transforms that rotate and scale space. A high condition number means the scaling is very uneven; some directions get stretched a lot, others barely at all. 

Muon's fix: **orthogonalize the momentum matrix**. This replaces all singular values with 1, keeping only the rotation part ($UV^\top$ from the SVD). This effectively rescues those rare but important directions.

Remember, we are talking about the **momentum matrix** as a transformation, not the weight matrix itself.

The whole Muon optimizer is about making this transform (the momentum matrix) orthogonal. And that improves token efficiency.

# Making Orthogonalization Fast

## SVD is Too Slow

The simplest approach to orthogonalization is via SVD. And in fact, this is what Keller Jordan et al. did in their initial experiments. However, SVD is quite expensive to compute on GPUs, especially for large matrices. So even though it works, it's not practical. The token efficiency gains are overshadowed by the wall-clock time increase due to SVD computation.

This motivates us to look for cheaper alternatives and raises the question: can we compute an approximate orthogonalization that is fast to compute on GPUs?

If we can find a polynomial approximation that, when applied to a matrix repeatedly, converges to its orthogonal form, we can use that instead. One such method is the Newton-Schulz iteration.

## Building Intuition with Scalars

To build intuition, let's first see how iterative convergence works for scalars before extending to matrices.

Consider the following cubic polynomial:
$$p(x) = \frac{3}{2}x - \frac{1}{2}x^3$$ 

where $x$ is a scalar and lies in the range $[0, 1]$.

If we start with any scalar value in this range and keep applying this polynomial repeatedly, we will converge to $1$.

Let's show that this is true. 

Consider $x = 1$: 

$$p(1) = \frac{3}{2} \cdot 1 - \frac{1}{2} \cdot 1^3 = 1$$

Thus, $x = 1$ is a stationary point.

Now consider $x < 1$: 

$$p(x) - x = \frac{3}{2}x - \frac{1}{2}x^3 - x = \frac{1}{2}x(1 - x^2) > 0$$

Thus $p(x) > x$ for all $x < 1$. This means that if we start with any value less than 1, applying $p(x)$ will increase its value.

Now that we have built some intuition, let's see how this extends to matrices. But first, what properties must our polynomial have?

### Properties of a Good Polynomial

1. It must be an **odd function**. This ensures the polynomial has the right symmetry properties for the Newton-Schulz iteration to work correctly.

2. The polynomial should have a **fixed point at $1$**. This ensures that once the input reaches this value, it remains stable.

## Extending to Matrices

Consider the matrix $M$. 

It can be decomposed using Singular Value Decomposition (SVD) as:
$$M = U \Sigma V^\top$$

Here, $U$ and $V$ are orthogonal matrices, and $\Sigma$ is a diagonal matrix containing the singular values of $M$.

When we apply the polynomial $p$ to the matrix $M$, we do so by applying it to its singular values:
$$p(M) = U p(\Sigma) V^\top$$
where $p(\Sigma)$ is obtained by applying the polynomial $p$ to each singular value in $\Sigma$.

By repeatedly applying $p$ to $M$, we get:
$$M_{k+1} = p(M_k) = U p(\Sigma_k) V^\top$$

As $k$ approaches infinity, the singular values in $\Sigma_k$ converge to 1 (since singular values are always non-negative). Consequently, $M_k$ converges to an orthogonal matrix $UV^\top$.

## Why Odd Polynomials Only Affect Singular Values

Let's assume that our quintic (degree 5) polynomial is an odd polynomial that can orthogonalize a matrix $G$:

$$G' = aG + b(GG^\top)G + c(GG^\top)^2G$$

Consider the SVD of $G$:
$$G = U \Sigma V^\top$$

Let's compute $GG^\top$:
$$GG^\top = (U \Sigma V^\top)(U \Sigma V^\top)^\top = U \Sigma V^\top V \Sigma^\top U^\top = U \Sigma \Sigma^\top U^\top = U \Sigma^2 U^\top$$

Since $V^\top V = I$ (orthogonal matrix) and $\Sigma^\top = \Sigma$ (diagonal matrix), we get that $GG^\top$ only depends on $U$ and the squared singular values.

Now let's compute $(GG^\top)^2$:
$$(GG^\top)^2 = (U \Sigma^2 U^\top)(U \Sigma^2 U^\top) = U \Sigma^2 U^\top U \Sigma^2 U^\top = U \Sigma^2 \Sigma^2 U^\top = U \Sigma^4 U^\top$$

Again, since $U^\top U = I$, we see that $(GG^\top)^2$ only depends on $U$ and $\Sigma^4$.

Now let's substitute everything into our quintic polynomial:

$$G' = aG + b(GG^\top)G + c(GG^\top)^2G$$

Substituting $G = U \Sigma V^\top$, $GG^\top = U \Sigma^2 U^\top$, and $(GG^\top)^2 = U \Sigma^4 U^\top$:

$$G' = a(U \Sigma V^\top) + b(U \Sigma^2 U^\top)(U \Sigma V^\top) + c(U \Sigma^4 U^\top)(U \Sigma V^\top)$$

Simplifying each term using $U^\top U = I$:

$$G' = aU \Sigma V^\top + bU \Sigma^2 \Sigma V^\top + cU \Sigma^4 \Sigma V^\top$$

$$G' = U (a\Sigma + b\Sigma^3 + c\Sigma^5) V^\top$$

$$G' = U \cdot p(\Sigma) \cdot V^\top$$

where $p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ is our quintic polynomial applied to each singular value.

Therefore, applying the Newton-Schulz iteration to a matrix $G$ is equivalent to applying the scalar polynomial $p(\sigma)$ to each of its singular values, while preserving the left and right singular vectors $U$ and $V$. However, we still need to find the coefficients $a$, $b$, and $c$ for our polynomial that will ensure convergence to 1 (since singular values are non-negative).

# The "Cursed" Quintic Polynomial

## Finding Coefficients Analytically

Consider the cubic polynomial with a and b as unknowns:
$$f(x) = ax + bx^3$$

To ensure that $f(x)$ has a fixed point at $1$, we set up the following equations:
1. For $x = 1$:
$$f(1) = a(1) + b(1)^3 = a + b = 1$$
2. Derivative should be 0 at $x = 1$ for stability:
$$f'(x) = a + 3bx^2$$
Setting $f'(1) = 0$ gives:
$$a + 3b = 0$$

Solving these equations simultaneously:

From the first equation:
$$a + b = 1 \quad (1)$$
From the second equation:
$$a + 3b = 0 \quad (2)$$
Subtracting (1) from (2):
$$2b = -1 \implies b = -\frac{1}{2}$$
Substituting $b$ back into (1):
$$a - \frac{1}{2} = 1 \implies a = \frac{3}{2}$$

Thus, the cubic polynomial is:
$$f(x) = \frac{3}{2}x - \frac{1}{2}x^3$$

## The Empirical Quintic

The Muon paper uses what they call a "cursed" quintic polynomial. Instead of deriving the coefficients analytically, they optimized for a polynomial that rapidly pushes the smallest singular values toward 1. They used an empirical approach to find a polynomial that converges in just 5 iterations. Interestingly, the polynomial is not actually stable at 1. Here is the exact function they used: 

$$f(x) = 3.4445x - 4.7750x^3 + 2.0315x^5$$

## Why Not Use The Simple Cubic?

1. **Cubic polynomial requires more iterations to converge**, leading to higher computational cost. This means more GPU time per optimization step — negating some of the speedup benefits of Muon.

2. **The "cursed" quintic polynomial pushes singular values toward 1 more aggressively**, leading to a better tradeoff between speed and convergence quality. Empirically, this is good enough for training models.

# Putting It All Together

## Which Parameters Use Muon vs AdamW?

Muon is only applied to **2D weight matrices in hidden layers** (e.g., attention projections, MLP layers). Everything else uses AdamW:

- **Embedding layers** — Use AdamW (different optimization dynamics per modular norm theory)
- **Language model head (output projection)** — Use AdamW (empirically performs better)
- **LayerNorm/RMSNorm scale and shift parameters** — Use AdamW (these are 1D vectors, not 2D matrices)
- **Biases** — Use AdamW (1D vectors)

For transformers specifically, Muon should be applied to Q, K, V projection weights separately rather than as a combined fused QKV matrix.

## Pseudocode

Here's the pseudocode for the Muon optimizer:

```python
# Muon Optimizer Pseudocode

def newton_schulz_orthogonalize(G, num_iters=5):
    # Normalize by Frobenius norm
    G = G / G.norm()
    
    # Transpose if tall matrix (more rows than columns)
    transposed = False
    if G.size(0) > G.size(1):
        G = G.T
        transposed = True
    
    # Cursed quintic polynomial coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    
    for _ in range(num_iters):
        A = G @ G.T
        G = a*G + b*A @ G + c*A @ A @ G
    
    # Transpose back if we transposed earlier
    if transposed:
        G = G.T
    
    return G

def muon_step(params, grads, momentum_buffer, lr, beta=0.95):
    for param, grad in zip(params, grads):
        if param.ndim == 2 and is_hidden_layer(param):
            # Nesterov momentum (empirically works better than standard momentum)
            momentum_buffer[param] = beta * momentum_buffer[param] + grad
            update = beta * momentum_buffer[param] + grad
            
            # Orthogonalize and apply
            param -= lr * newton_schulz_orthogonalize(update)
        else:
            # Use AdamW for embeddings, output head, 1D params, etc.
            adamw_step(param, grad)
```

# Key Takeaways

1. **AdamW updates are elementwise** — they ignore the matrix structure of weight gradients, leading to updates dominated by a few directions.

2. **Muon orthogonalizes the momentum** — by replacing the momentum matrix with its nearest orthogonal matrix ($UV^\top$), it rescues rare but important learning directions.

3. **Newton-Schulz iteration is the key trick** — it approximates orthogonalization using only matrix multiplications, which are fast on GPUs. The "cursed" quintic polynomial converges in just 5 iterations.

4. **Muon only applies to hidden layer 2D weights** — embeddings, output heads, and 1D parameters still use AdamW.

For the original writeup and implementation, see [Keller Jordan's blog post](https://kellerjordan.github.io/posts/muon/) and the [Muon GitHub repo](https://github.com/KellerJordan/Muon).
