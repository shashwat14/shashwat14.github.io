---
title: "Understanding Quantization in Neural Networks"
date: 2025-10-18
draft: false
summary: "A mathematical exploration of quantization techniques for model compression."
ShowToc: true
math: true
---

## Introduction

Quantization is a fundamental technique for reducing the memory footprint and computational cost of neural networks. By converting high-precision floating-point weights to lower-precision representations, we can achieve significant speedups with minimal loss in accuracy.

## The Mathematics of Quantization

### Basic Quantization Formula

The core idea of quantization is to map a continuous range of floating-point values to a discrete set of integers. For a given real value $r$, the quantized value $q$ is computed as:

$$
q = \text{round}\left(\frac{r}{S}\right) - Z
$$

where:
- $S$ is the **scale factor** (a positive real number)
- $Z$ is the **zero-point** (an integer)

The dequantization process recovers an approximation of the original value:

$$
\tilde{r} = S(q + Z)
$$

### Determining Scale and Zero-Point

For symmetric quantization around zero, the scale factor is computed as:

$$
S = \frac{\max(|r_{\min}|, |r_{\max}|)}{2^{b-1} - 1}
$$

where $b$ is the bit-width of the quantized representation.

For asymmetric quantization, which can better utilize the full range:

$$
S = \frac{r_{\max} - r_{\min}}{2^b - 1}
$$

$$
Z = \text{round}\left(-\frac{r_{\min}}{S}\right)
$$

## Quantization-Aware Training

### Straight-Through Estimator

During backpropagation, the quantization operation is non-differentiable. The **Straight-Through Estimator (STE)** provides a solution by approximating the gradient:

$$
\frac{\partial q}{\partial r} \approx \begin{cases}
1 & \text{if } r \in [r_{\min}, r_{\max}] \\
0 & \text{otherwise}
\end{cases}
$$

This allows gradients to flow through the quantization operation during training.

### Quantized Matrix Multiplication

Consider two quantized matrices $A$ and $B$. Their product can be computed efficiently in integer arithmetic:

$$
C = AB = S_A S_B (q_A + Z_A)(q_B + Z_B)
$$

Expanding this:

$$
C = S_A S_B \left[q_A q_B + Z_A q_B + Z_B q_A + Z_A Z_B\right]
$$

The term in brackets can be computed entirely in integer arithmetic, with the floating-point scale applied only once at the end.

## Per-Channel vs Per-Tensor Quantization

### Per-Tensor Quantization

A single scale and zero-point is used for an entire tensor:

$$
q_{i,j} = \text{round}\left(\frac{r_{i,j}}{S}\right) - Z
$$

### Per-Channel Quantization

Different scales and zero-points for each channel (typically output channels for weights):

$$
q_{i,j} = \text{round}\left(\frac{r_{i,j}}{S_j}\right) - Z_j
$$

Per-channel quantization typically provides better accuracy at the cost of slightly more complex dequantization.

## Signal-to-Quantization-Noise Ratio

The quality of quantization can be measured using SQNR:

$$
\text{SQNR} = 10 \log_{10} \left(\frac{\mathbb{E}[r^2]}{\mathbb{E}[(r - \tilde{r})^2]}\right)
$$

For uniform quantization with step size $\Delta = S$, the SQNR for a signal with variance $\sigma^2$ is approximately:

$$
\text{SQNR} \approx 10 \log_{10}\left(\frac{12\sigma^2}{\Delta^2}\right) = 6.02b + 10.79 \text{ dB}
$$

where $b$ is the number of bits.

## Mixed-Precision Quantization

Not all layers contribute equally to model accuracy. We can formulate layer-wise bit-width selection as an optimization problem:

$$
\min_{b_1, \ldots, b_L} \mathcal{L}(\mathbf{W}_q)
$$

subject to:

$$
\sum_{i=1}^{L} M_i \cdot b_i \leq B_{\text{total}}
$$

where $\mathcal{L}$ is the loss function, $M_i$ is the number of parameters in layer $i$, and $B_{\text{total}}$ is the total bit budget.

## Practical Considerations

### Dynamic Range and Clipping

The quantization range $[r_{\min}, r_{\max}]$ significantly impacts accuracy. Using percentile-based clipping can improve SQNR:

$$
r_{\min} = \text{percentile}(r, \alpha), \quad r_{\max} = \text{percentile}(r, 100-\alpha)
$$

where $\alpha$ is typically chosen between 0.01 and 1.

### KL Divergence Minimization

For post-training quantization, we can choose the threshold $T$ that minimizes the KL divergence between the original and quantized distributions:

$$
T^* = \arg\min_T D_{KL}(P \parallel Q) = \arg\min_T \sum_i P(i) \log\frac{P(i)}{Q(i)}
$$

## Conclusion

Quantization provides a powerful tool for model compression, trading off minimal accuracy for significant gains in efficiency. The mathematical framework presented here forms the foundation for various quantization schemes used in production systems today.

Key takeaways:
- 8-bit quantization can achieve near-lossless compression for many models
- Per-channel quantization offers better accuracy than per-tensor
- Quantization-aware training can recover accuracy lost in post-training quantization
- Mixed-precision approaches allow fine-grained control over the accuracy-efficiency trade-off

## References

1. Jacob et al. (2018) - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
2. Krishnamoorthi (2018) - "Quantizing deep convolutional networks for efficient inference: A whitepaper"
3. Gholami et al. (2021) - "A Survey of Quantization Methods for Efficient Neural Network Inference"
