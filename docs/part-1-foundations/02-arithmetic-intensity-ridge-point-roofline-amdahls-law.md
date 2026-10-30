# Chapter 2: Arithmetic Intensity, Ridge Point, Roofline, and Amdahl's Law

> **Prerequisite:** [Chapter 1: Latency, throughput, FLOPs, bytes, bandwidth](01-latency-throughput-flops-bytes-bandwidth.md)


## Learning objectives

After this chapter, you should be able to:

- Explain arithmetic intensity in one sentence.
- Hand-calculate the work ($F$), the minimum bytes ($B$), and the arithmetic intensity for six common operations.
- Explain the ridge point and calculate it for a GPU.
- Read a roofline chart and use it to say how fast an operation can possibly go.
- Explain why estimates and measurements do not, by themselves, identify the real bottleneck.
- Use Amdahl's law to decide whether speeding up one kernel is worth the effort.

## 2.1 The question this chapter answers

In Chapter 1 you learned the vocabulary: latency, throughput, FLOPs, bytes, and bandwidth. Now a natural question appears:

> *My kernel takes 0.2 ms. Is that good? How fast could it possibly be? And is it worth my time to make it faster?*

This chapter gives you the tools to answer it:

- **Arithmetic intensity** and **hand calculations** let you describe an operation with a few numbers.
- The **roofline model** answers: *what is the best speed this operation can reach on this GPU?*
- **Amdahl's law** answers: *if I speed up this operation, how much faster does the whole program get?*

There are **two words we need to know** as they will appear in the following sections: *memory-limited* (the slowest part is moving data) and *compute-limited* (the slowest part is doing math). Chapter 3 explains these ideas in depth, and adds a third reason an operation can be slow (the fixed cost of starting work) that the roofline cannot show.

## 2.2 Arithmetic intensity

**Arithmetic intensity** is how much math you do for each byte of data you move:

$$
\text{arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes moved}} \quad (\text{FLOP per byte})
$$

Think of cooking. If a recipe needs you to carry one ingredient from a far-away fridge and then only stir it once, you spend your time walking. If a recipe needs you to carry the same ingredient once and then chop, boil, fry, and season it, you spend your time cooking. Arithmetic intensity is "how much cooking per trip to the fridge".

To get the intensity of an operation, you need only two numbers from Chapter 1:

1. **$F$**: the work (FLOPs, or operation counts where FLOPs are not meaningful).
2. **$B$**: the minimum logical bytes read and written.
3. Divide: $\text{intensity} = F / B$.

Three things to keep in mind:

- Intensity depends on the **shape** and the **dtype**, not only on the operation. FP16 moves half the bytes of FP32 for the same FLOPs, so the intensity doubles.
- It uses the **minimum logical bytes**. If an implementation reads the same data several times, the real intensity is lower than the number you calculate.
- It is a property of the *workload*. It says nothing yet about the hardware. We add the hardware in sections 2.4 and 2.5.

## 2.3 Hand-calculating six operations

Before comparing PyTorch and Triton implementations, we calculate $F$, $B$, and intensity by hand for six operations. These numbers become the baseline for every later chapter.

**Notation.**
- For vector operations, $N$ is the vector length.
- For row-wise operations (row sum, softmax, layer norm), $M$ is the number of rows and $N$ is the length of each row.
- For matrix multiplication only, $A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$. (Here the letter $B$ is a matrix, so we write the byte count as $B_{\min}$.)
- $s$ is the number of bytes per element (FP32: 4; FP16 and BF16: 2).

**Conventions.**
- A multiply-add counts as two FLOPs.
- "Minimum logical bytes" ($B_{\min}$) assumes each required input is read once and each output is written once.
- These estimates leave out temporary tensors that a particular implementation might create, and they do not promise what the GPU memory actually moves.

**Shapes used in the examples** (all FP32): vector operations $N = 10^6$; row operations $M = 1024$, $N = 256$; matrix multiplication $M = N = K = 1024$.

### Vector addition

For $c_i = a_i + b_i$:

- **Work:** one addition per element, so $F \approx N$.
- **Bytes:** read $a$, read $b$, write $c$, so $B_{\min} = 3Ns$.
- **Intensity:** $\dfrac{N}{3Ns} = \dfrac{1}{3s}$. For FP32 that is $1/12 \approx 0.083$ FLOP per byte.

Example: $N = 10^6$ gives $10^6$ additions, 12,000,000 bytes, and intensity 0.083.

### ReLU

For $y_i = \max(0, x_i)$:

- **Work:** there is no useful ordinary-FLOP count here. Implementations use compare, max, or select instructions, not arithmetic. Under the strict convention ReLU performs **0 FLOPs**. In this repo we report it as about **$N$ compare/select operations**, and we do not compare that number directly with FLOP counts from other operations.
- **Bytes:** read $x$, write $y$, so $B_{\min} = 2Ns$.
- **Intensity:** $\dfrac{N}{2Ns} = \dfrac{1}{2s}$. For FP32 that is 0.125 operations per byte (operations, not FLOPs).

Example: $N = 10^6$ gives about $10^6$ compare/select operations, 8,000,000 bytes, and intensity 0.125.

### Row-wise sum

For an $M \times N$ matrix reduced to one value per row, each row needs $N - 1$ additions:

- **Work:** $F \approx M(N - 1)$.
- **Bytes:** read the whole matrix and write one value per row, so $B_{\min} = MNs + Ms$.
- **Intensity:** about $\dfrac{MN}{MNs} = \dfrac{1}{s}$ when rows are long. For FP32 that is about 0.25.

Example: $M = 1024$, $N = 256$ gives $1024 \times 255 = 261{,}120$ additions, $1{,}048{,}576 + 4{,}096 = 1{,}052{,}672$ bytes, and intensity $0.248$.

### Softmax

For each row:

$$
m = \max_i x_i, \qquad e_i = \exp(x_i - m), \qquad y_i = \frac{e_i}{\sum_j e_j}
$$

- **Work, per row:**

| Step | Count |
| --- | --- |
| Max | $N - 1$ comparisons |
| Subtract $m$ | $N$ subtractions |
| Exponentials | $N$ `exp` calls |
| Sum | $N - 1$ additions |
| Normalize | $N$ divisions (or one reciprocal plus $N$ multiplications) |

  Adding these gives $5N - 2$ operations per row. A single count hides the fact that an `exp` or a division costs more than an addition, so **report exponentials and comparisons separately**.
- **Bytes:** read $x$, write $y$, so $B_{\min} = 2MNs$. This is a **lower bound**. It assumes no intermediate tensor is written and read back. A naive implementation that makes separate passes for max, exp-sum, and normalize can read the input several times and move more data. A fused implementation can approach the lower bound. Chapters 14 and 17 show how.
- **Intensity:** about $\dfrac{5MN}{2MNs} = \dfrac{5}{2s}$. For FP32 that is about 0.6 (counting each `exp` as one operation).

Example: $M = 1024$, $N = 256$ gives $1024 \times 1278 = 1{,}308{,}672$ operations (of which 262,144 are exponentials), 2,097,152 bytes, and intensity $0.624$.

### Matrix multiplication

For $C = AB$ with $A: [M, K]$, $B: [K, N]$, $C: [M, N]$:

- **Work:** each of the $MN$ outputs is a dot product of length $K$, so $F \approx 2MNK$.
- **Bytes:** read $A$ and $B$, write $C$, so $B_{\min} = s\,(MK + KN + MN)$.
- **Intensity:** $\dfrac{2MNK}{s(MK + KN + MN)}$.

Example: $M = N = K = 1024$, FP32 gives 2,147,483,648 FLOPs, 12,582,912 bytes, and intensity **170.7**. Real traffic depends on tiling, cache behavior, and reuse.

**Record the precision mode.** FP32 matrix multiplication may run on tensor cores as TF32 depending on settings (for example, `torch.set_float32_matmul_precision`). That changes both the numerical results and the achievable peak, so state it next to your measurement.

#### Same operation, different intensity

Matrix multiplication shows clearly why shape matters. Take the multiplication at the heart of LLM inference: $A$ is $M \times 4096$, $B$ is $4096 \times 4096$ (the weights), in BF16 ($s = 2$). The big weight matrix has to be read no matter how few rows $A$ has:

| Rows $M$ | FLOPs | Bytes | Intensity |
| --- | --- | --- | --- |
| 1 (one token) | 33.6 million | 33.6 MB | about 1 |
| 16 | 537 million | 33.8 MB | about 16 |
| 128 | 4.3 billion | 35.7 MB | about 120 |
| 256 | 8.6 billion | 37.7 MB | about 228 |

More rows reuse the same weights, so the intensity rises roughly in proportion to $M$. This is the main reason batching helps LLM serving.

### Layer normalization

For each row, layer normalization computes a mean and variance, normalizes, and may apply a learned scale $\gamma$ and bias $\beta$:

$$
\mu = \frac{1}{N}\sum_i x_i, \qquad
\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2, \qquad
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i
$$

- **Work, per row** (reusing the centered values $x_i - \mu$):

| Step | Approx. ops per row |
| --- | --- |
| Mean (sum and divide) | $N$ |
| Center: $x_i - \mu$ | $N$ |
| Square and sum for variance | $2N$ |
| Scale by reciprocal standard deviation | $N$ |
| **Subtotal (no affine)** | **$\approx 5N$** |
| Multiply by $\gamma$, add $\beta$ | $2N$ |
| **Total (with affine)** | **$\approx 7N$** |

  So the total is about $5MN$ to $7MN$ simple operations, plus one reciprocal square root per row. This is an **order-of-magnitude estimate, not a precise FLOP count**. The exact work depends on the variance algorithm (Chapter 15 covers Welford's method), the affine parameters, and fusion.
- **Bytes:** read $x$, write $y$, and read the two shared parameter vectors $\gamma$ and $\beta$: $B_{\min} = 2MNs + 2Ns$. This assumes each parameter vector is read once in the logical model; cache reuse and implementation details affect actual traffic. In training, the forward pass typically also writes the mean and reciprocal standard deviation ($2Ms$ extra bytes) for the backward pass.
- **Intensity:** about $\dfrac{5MN \text{ to } 7MN}{2MNs} = \dfrac{2.5 \text{ to } 3.5}{s}$. For FP32 that is about 0.6 to 0.9.

Example: $M = 1024$, $N = 256$ gives about $1.31 \times 10^6$ to $1.84 \times 10^6$ operations, 2,099,200 bytes, and intensity between 0.62 and 0.87.

### Summary table

All FP32. Vector operations $N = 10^6$; row operations $M = 1024$, $N = 256$; matmul $M = N = K = 1024$.

| Operation | Work $F$ | Min. logical bytes $B_{\min}$ | Arithmetic intensity |
| --- | --- | --- | --- |
| Vector add | $10^6$ FLOPs | 12,000,000 | 0.083 |
| ReLU | $10^6$ compare/select (0 FLOPs) | 8,000,000 | 0.125 (operations, not FLOPs) |
| Row sum | 261,120 FLOPs | 1,052,672 | 0.25 |
| Softmax | about $1.3 \times 10^6$ ops (262,144 `exp`) | 2,097,152 | about 0.6 |
| Layer norm | about $1.3$ to $1.8 \times 10^6$ ops | 2,099,200 | about 0.6 to 0.9 |
| Matmul | 2,147,483,648 FLOPs | 12,582,912 | about 171 |

Look at the last column. Five of these operations do less than one operation per byte moved, while the matmul does roughly 171. That gap, two to three orders of magnitude, is the premise of most of this course: it is why some kernels are limited by memory and others by arithmetic, and why fusion and tiling matter. To turn it into a statement about a real GPU, we need the hardware's side of the story.

## 2.4 What the GPU can do: two peak numbers

A GPU has two speed limits, and both are published in its specifications:

- **Peak compute**: the most math it can do per second, in FLOP/s (for example, 200 TFLOP/s).
- **Peak memory bandwidth**: the most data it can move per second, in bytes/s (for example, 2 TB/s).

Three practical tips:

1. **Pick the compute peak for the dtype you use.** FP32 CUDA-core throughput, TF32 tensor-core throughput, and BF16/FP16 tensor-core throughput can differ substantially. Also check whether the published figure assumes sparsity.
2. **Check whether a number is "dense" or "with sparsity".** Spec sheets sometimes list a higher figure that only applies to a special sparse mode. Use the dense number.
3. **Published peaks are rarely reached exactly.** Theoretical bandwidth is a hardware ceiling, not a performance promise. You can estimate sustainable bandwidth by benchmarking a large streaming operation, such as vector addition or memory copy. The result depends on the operation, access pattern, and memory level.

In this chapter we use an *illustrative* GPU with **200 TFLOP/s** and **2 TB/s**. It is not a specific product. Swap in your own GPU's numbers when you do the exercise.

## 2.5 The ridge point

The **ridge point** is the GPU's own arithmetic intensity: how much math it can do per byte it can move.

$$
\text{ridge point} = \frac{\text{peak compute}}{\text{peak memory bandwidth}}
$$

For our illustrative GPU:

$$
\frac{200 \times 10^{12}\ \text{FLOP/s}}{2 \times 10^{12}\ \text{B/s}} = 100\ \text{FLOP per byte}
$$

(Handy trick: TFLOP/s divided by TB/s gives FLOP/byte directly.)

The ridge point is a break-even line between two situations:

- Below the ridge point: The operation performs relatively little computation for each byte transferred. Data cannot arrive fast enough to keep the math units busy, so the operation is **memory-limited**.
- Above the ridge point: The operation performs relatively more computation per byte. The math units are the slowest part, so the operation is **compute-limited**.

Using the table in 2.3: vector add (0.083) is far below a ridge point of 100, and the 1024³ matmul (171) is above it.

**The ridge point changes.** A faster math path (such as tensor cores) raises the compute peak and therefore the ridge point: the GPU now needs *more* work per byte to stay busy, and more operations become memory-limited. In general, GPU math speed has grown faster than memory bandwidth over the years, so ridge points have tended to rise.

