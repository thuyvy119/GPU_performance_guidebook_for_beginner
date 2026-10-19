# Chapter 1: Latency, Throughput, FLOPs, Bytes, and Bandwidth

## Learning objectives

After this chapter, you should be able to:
- Distinguish latency from throughput, and relate them with Little's law.
- Explain what FLOPs (a count) and FLOP/s (a rate) measure.
- Estimate the arithmetic work of a simple operation by hand.
- Estimate the minimum logical bytes read and written by an operation.
- Calculate effective throughput and effective bandwidth from measurements.
- Explain why these estimates do not, by themselves, identify the actual bottleneck.

## 1.1 Why start here? A motivating example

Suppose you serve an 8-billion-parameter language model in BF16 (2 bytes per parameter). The weights occupy about 16 GB.

To generate **one token**, a GPU must read roughly all of those weights from memory once. If the GPU's memory bandwidth is 2 TB/s (an illustrative number, not a specific product):

$$
t \gtrsim \frac{16\ \text{GB}}{2\ \text{TB/s}} = 8\ \text{ms}
\quad\Rightarrow\quad
\text{tokens/s} \lesssim \frac{1}{0.008} = 125
$$

Nothing in that estimate mentions FLOPs, kernels, or clever optimizations. It only counts **bytes moved** and divides by **bandwidth**. Yet it already explains a lot about real inference behavior, including why batching many requests together helps: the weights are read once and serve every sequence in the batch.

This chapter builds the vocabulary and the habit behind that estimate:
1. Define what you are measuring (latency or throughput).
2. Count the work (FLOPs).
3. Count the data (bytes).
4. Divide by time to get rates.
5. Be honest about what those numbers do and do not prove.

## 1.2 Latency and throughput answer different questions

**Latency** asks: *How long does one operation or request take?*
**Throughput** asks: *How much work can the system complete per unit time?*

Latency might be measured in microseconds for a single kernel. 
Throughput might be measured in operations per second, requests per second, or tokens per second.

A system can have low latency for an individual request but limited throughput when many requests arrive. Conversely, batching can improve throughput while increasing the waiting time for some requests. Always choose the metric that matches the goal:

| Goal | Metric to watch |
| --- | --- |
| Interactive chat feels snappy | Latency (time to first token, time per token) |
| Serve as many users as possible per GPU | Throughput (tokens/s across all requests) |
| Offline batch processing | Throughput |

For a stable stream of independent, one-at-a-time work, throughput is sometimes approximated as the reciprocal of latency. That approximation is **not** generally valid for concurrent, batched, pipelined, or queueing systems.

### Little's law

For a stable system, the three quantities are connected:

$$
\text{concurrency} = \text{throughput} \times \text{latency}
$$

If each request takes 50 ms and you want 1,000 requests/s, then on average $1000 \times 0.05 = 50$ requests must be in flight at once. This is why batching and concurrency raise throughput without lowering per-request latency, and it will return when we discuss GPU occupancy and LLM batching.

## 1.3 FLOPs and FLOP/s

Two terms are easy to confuse:
- **FLOPs** (lowercase s): a **count** of floating-point operations. It measures *work*.
- **FLOP/s**: a **rate**, floating-point operations per second. It measures *speed*.

For a matrix multiplication:

$$
C_{M \times N} = A_{M \times K}\, B_{K \times N}
$$

Each output element is a dot product of length $K$, which needs $K$ multiplies and about $K$ adds. Under the common convention that one multiply and one add count as two FLOPs (hardware often fuses them into one *fused multiply-add*, FMA, instruction, which is why the convention counts FMA as 2 FLOPs), the approximate arithmetic work is:

$$
\text{FLOPs} \approx 2MNK
$$

This is an approximation based on a counting convention. Some contexts count differently, so **state your convention** when reporting performance.

If an operation performs $F$ FLOPs in $t$ seconds, its achieved arithmetic throughput is:

$$
P = \frac{F}{t}
$$

For example, a workload performing $2 \times 10^{12}$ FLOPs in $0.01$ s achieves:

$$
P = \frac{2 \times 10^{12}}{0.01} = 2 \times 10^{14}\ \text{FLOP/s} = 200\ \text{TFLOP/s}
$$

This is achieved throughput *under the assumed operation count*. It does not guarantee that all the work maps to the same kind of hardware instruction (for example, tensor cores versus regular FP32 units).

**Not all operations are equal.** Additions, multiplications, divisions, and transcendental functions such as `exp` have different hardware costs. Counting each as "one FLOP" is a convenient simplification, and Section 1.6 shows where it becomes misleading.

## 1.4 Bytes moved

For a first estimate, count the bytes that must be read and written at the tensor level. For an array with $N$ elements and element size $s$ bytes:

$$
\text{Array size in bytes} = N s
$$

Common element sizes: FP32 = 4 bytes, FP16/BF16 = 2 bytes, FP8/INT8 = 1 byte.

### Example: vector addition
For $C = A + B$ with $N$ elements:
- Read $A$: $Ns$ bytes
- Read $B$: $Ns$ bytes
- Write $C$: $Ns$ bytes

$$
\text{Bytes}_{\min} = 3Ns
$$

The arithmetic work is about $N$ additions, so each element costs one addition but moves $3s$ bytes. For FP32 that is $N$ FLOPs over $12N$ bytes, or $1/12 \approx 0.083$ FLOP per byte. That ratio is called **arithmetic intensity** (FLOPs per byte moved). It is formally introduced in Chapter 3, but the pattern is already visible: very little math per byte means vector addition is commonly memory-bandwidth-bound for sufficiently large arrays.

> **Logical versus physical traffic.** This is a *logical* estimate: the minimum bytes the operation must read and write if every input is read once and every output is written once. Physical DRAM traffic can differ because of caches, write policies, alignment, and implementation details.

## 1.5 Bandwidth and effective bandwidth

**Bandwidth** is the amount of data transferred per unit time, often expressed in GB/s or TB/s. A simple estimate from a measurement is:
$$
B_{\text{effective}} = \frac{\text{estimated bytes moved}}{\text{elapsed time}}
$$

### Running example
Take a vector addition over $N = 25 \times 10^6$ FP32 elements:

$$
\text{Bytes}_{\min} = 3 \times 25 \times 10^6 \times 4 = 300\ \text{MB}
$$

If the measured time is $0.001$ s:

$$
B_{\text{effective}} = \frac{300\ \text{MB}}{0.001\ \text{s}} = 300\ \text{GB/s}
$$

The same measurement gives only $25 \times 10^6 / 0.001 = 25\ \text{GFLOP/s}$, a tiny fraction of a modern GPU's arithmetic peak. For this operation, bytes are the interesting quantity, not FLOPs.

**Unit note:** this example uses decimal units (1 GB = $10^9$ bytes). Be consistent about decimal versus binary units (GiB = $2^{30}$ bytes).

### Two cautions about measurement

1. **Effective bandwidth is an estimate.** It is computed from *logical* tensor bytes, so it is not necessarily actual DRAM bandwidth. Data may be served from L1 or L2 cache, and physical traffic can include effects the simple estimate ignores. Call the metric an *estimate* unless hardware counters establish the real traffic.
2. **GPU timing is tricky.** Kernel launches are asynchronous: the CPU returns before the GPU finishes. A naive `time.time()` around a launch measures launch cost, not execution. Use CUDA events or synchronize before reading the clock, and warm up first. See [`docs/benchmark-methodology.md`](../benchmark-methodology.md).

## 1.6 Hand-calculating operations

These estimates create a baseline before comparing PyTorch and Triton implementations.

**Notation.** In this section, for the row-wise operations (row sum, softmax, layer norm), $M$ is the number of rows and $N$ is the length of each row. For vector operations, $N$ is the vector length. For matrix multiplication only, $A$ is $M \times K$, $B$ is $K \times N$, and $C$ is $M \times N$. $s$ is bytes per element (FP32: 4, FP16/BF16: 2).

**Conventions.**
- A multiply-add counts as two FLOPs.
- "Minimum logical bytes" assumes required inputs are read once and outputs written once.
- Estimates omit implementation-specific temporaries and do not guarantee physical HBM traffic.

### Vector addition

For $c_i = a_i + b_i$:

$$
F \approx N, \qquad B_{\min} = 3Ns
$$

For $N = 10^6$ FP32 elements: about $10^6$ additions and 12,000,000 minimum logical bytes.

### ReLU

For $y_i = \max(0, x_i)$, there is no useful ordinary-FLOP count: implementations use compare, max, or select instructions, not arithmetic. Under the strict convention ReLU performs **0 FLOPs**. In this repo we report it as approximately **$N$ elementwise compare/select operations**, and we do not compare that number directly with FLOP counts from other operations.

$$
B_{\min} = 2Ns
$$

For $N = 10^6$ FP32 elements: 8,000,000 minimum logical bytes.

### Row-wise sum

For an $M \times N$ matrix reduced to one value per row, each row needs $N - 1$ additions:

$$
F \approx M(N-1), \qquad B_{\min} = MNs + Ms
$$

For $M = 1024$, $N = 256$, FP32 input and output: 261,120 additions and 1,052,672 minimum logical bytes.

### Softmax

For each row:

$$
m = \max_i x_i, \qquad e_i = \exp(x_i - m), \qquad y_i = \frac{e_i}{\sum_j e_j}
$$

Work per row:
| Step | Count |
| --- | --- |
| Max | $N - 1$ comparisons |
| Subtract $m$ | $N$ subtractions |
| Exponentials | $N$ `exp` calls |
| Sum | $N - 1$ additions |
| Normalize | $N$ divisions (or one reciprocal plus $N$ multiplications) |

A single FLOP count hides the cost differences between these operation types (an `exp` or a division costs more than an addition), so **report exponentials and comparisons separately**.

For input and output only:

$$
B_{\min} = 2MNs
$$

This is a **lower bound**. It assumes no intermediate tensor is written and read back. A naive implementation that makes separate passes for max, exp-sum, and normalize can read the input several times and move more data. A fused implementation can approach the lower bound. Chapters 14 and 17 show how.

For $M = 1024$, $N = 256$, FP32: about $1.3 \times 10^6$ operations (of which 262,144 are exponentials) and 2,097,152 minimum logical bytes.

### Matrix multiplication

For $C = AB$ with $A: [M, K]$, $B: [K, N]$, $C: [M, N]$:

$$
F \approx 2MNK, \qquad B_{\min} = s\,(MK + KN + MN)
$$

For $M = N = K = 1024$, FP32: 2,147,483,648 FLOPs and 12,582,912 minimum logical bytes. Real traffic depends on tiling, cache behavior, and reuse.

**Record the precision mode.** FP32 matrix multiplication may run on tensor cores as TF32 depending on settings (for example, `torch.set_float32_matmul_precision`). That changes both numerical results and the achievable peak, so state it next to your measurement.

### Layer normalization

For each row, layer normalization computes a mean and variance, normalizes, and may apply a learned scale $\gamma$ and bias $\beta$:

$$
\mu = \frac{1}{N}\sum_i x_i, \qquad
\sigma^2 = \frac{1}{N}\sum_i (x_i - \mu)^2, \qquad
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i
$$

A rough count per row, reusing the centered values $x_i - \mu$:
| Step | Approx. ops per row |
| --- | --- |
| Mean (sum and divide) | $N$ |
| Center: $x_i - \mu$ | $N$ |
| Square and sum for variance | $2N$ |
| Scale by reciprocal standard deviation | $N$ |
| **Subtotal (no affine)** | **$\approx 5N$** |
| Multiply by $\gamma$, add $\beta$ | $2N$ |
| **Total (with affine)** | **$\approx 7N$** |

So the total is on the order of $5MN$ to $7MN$ simple operations, plus the reductions' structure and one reciprocal square root per row. This is an **order-of-magnitude estimate, not a precise FLOP count**. The exact work depends on the variance algorithm (Chapter 15 covers Welford's method), the affine parameters, and fusion.

For input, output, and two shared parameter vectors:

$$
B_{\min} = 2MNs + 2Ns
$$

This assumes each parameter vector is read once in the logical model; cache reuse and implementation details affect actual traffic. In training, the forward pass typically also writes the mean and reciprocal standard deviation ($2Ms$ extra bytes) for use in the backward pass.

### Summary table

Shapes: vector ops $N = 10^6$; row ops $M = 1024$, $N = 256$; matmul $M = N = K = 1024$. All FP32.
| Operation | Work | Min. logical bytes | Arithmetic intensity (ops/byte) |
| --- | --- | --- | --- |
| Vector add | $10^6$ FLOPs | 12,000,000 | 0.083 |
| ReLU | $10^6$ compare/select (0 FLOPs) | 8,000,000 | 0.125 (not FLOPs) |
| Row sum | 261,120 FLOPs | 1,052,672 | 0.25 |
| Softmax | $\approx 1.3 \times 10^6$ ops (262,144 `exp`) | 2,097,152 | $\approx 0.6$ |
| Layer norm | $\approx 1.3$ to $1.8 \times 10^6$ ops | 2,099,200 | $\approx 0.6$ to $0.9$ |
| Matmul | 2,147,483,648 FLOPs | 12,582,912 | $\approx 171$ |

Look at the last column. Five of these operations do less than one operation per byte moved, while matmul does roughly 171. That gap, two to three orders of magnitude, is the premise of most of this course: it is why some kernels are limited by memory and others by arithmetic, and why fusion and tiling matter. Chapter 3 formalizes it as the roofline model.

## 1.7 Why estimates do not identify the bottleneck

The numbers above are **estimates built from stated assumptions**, and measurements are **derived from those estimates plus timing**. Neither directly shows *why* a kernel is slow. Be careful of these traps:
- **High effective bandwidth does not prove memory-bound.** If the logical bytes are served from cache, the estimate can even exceed the DRAM peak.
- **High FLOP/s does not prove compute-bound.** Different instruction types (tensor-core versus regular units) have very different peaks.
- **Low numbers do not identify a cause.** A kernel far below peak may be limited by launch overhead, poor memory access patterns, low occupancy, synchronization, or something else entirely.
- **The estimate may not describe what actually ran.** Extra temporaries, multiple kernels, or a different algorithm change the real work and traffic.

The right procedure is: compare each measurement with the *relevant hardware limits* (published peaks for your exact GPU and precision), form a hypothesis, then look for **additional evidence**, such as profiler counters, to confirm or reject it. Chapters 2 and 3 provide the vocabulary for forming hypotheses; Chapter 22 and Lab 06 provide the evidence.

## Summary

- Latency measures how long a defined unit of work takes; throughput measures how much work completes per unit time. Little's law links them: concurrency = throughput × latency.
- FLOPs count arithmetic work; FLOP/s is the rate at which that work is performed. State your counting convention.
- Logical-byte estimates describe required reads and writes under stated assumptions, not physical traffic.
- Arithmetic intensity (work per byte) already separates operations such as vector add (≈0.08) from matmul (≈171).
- Effective bandwidth and achieved FLOP/s are derived from estimates plus timing, so interpretation depends on both.
- Estimates and measurements do not by themselves identify a bottleneck; they produce hypotheses that need further evidence.
- Measurements are more useful when timing boundaries, shapes, dtypes, assumptions, and methodology are recorded.

## References

- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA Matrix Multiplication Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

**Next:** [Chapter 2: Compute-bound, memory-bound, and launch overhead](02-compute-memory-bound-launch-overhead.md)