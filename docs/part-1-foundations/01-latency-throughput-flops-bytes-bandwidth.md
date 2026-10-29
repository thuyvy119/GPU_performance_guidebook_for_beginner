# Chapter 1: Latency, Throughput, FLOPs, Bytes, and Bandwidth

## Learning objectives

After this chapter, we should be able to:
- Distinguish latency from throughput, and relate them with Little's law.
- Explain what FLOPs (a count) and FLOP/s (a rate) measure.
- Estimate the arithmetic work of a simple operation by hand.
- Estimate the minimum logical bytes read and written by an operation.
- Calculate effective throughput and effective bandwidth from measurements.
- Explain why effective bandwidth computed from logical bytes is only an estimate.

## 1.1 Why start here? A motivating example

Suppose serving an 8-billion-parameter language model in BF16 (2 bytes per parameter). The weights occupy about 16 GB.

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

Latency might be measured in microseconds for a single kernel. Throughput might be measured in operations per second, requests per second, or tokens per second.

A system can have low latency for an individual request but limited throughput when many requests arrive. Conversely, batching can improve throughput while increasing the waiting time for some requests. Always choose the metric that matches the goal:

| Goal | Metric to watch |
| --- | --- |
| Interactive chat feels snappy | Latency (time to first token, time per token) |
| Serve as many users as possible per GPU | Throughput (tokens/s across all requests) |
| Offline batch processing | Throughput |

For a stable stream of independent, one-at-a-time work, throughput is sometimes approximated as the reciprocal of latency. That approximation is **not** generally valid for concurrent, batched, pipelined, or queueing systems.

### Little's law

For a stable system, the three quantities are connected. This is **Little's law** [1], usually written $L = \lambda W$, where $L$ is the average number of items in the system, $\lambda$ is the average arrival rate (equal to throughput in steady state), and $W$ is the average time an item spends in the system (latency):

$$
\text{concurrency} = \text{throughput} \times \text{latency}
$$

The law holds for long-run averages of a stable (stationary) system, and it does not depend on the arrival pattern or scheduling order.

If each request takes 50 ms and you want 1,000 requests/s, then on average $1000 \times 0.05 = 50$ requests must be in flight at once. This is why batching and concurrency raise throughput without lowering per-request latency, and it will return when we discuss GPU occupancy and LLM batching. On a GPU the same relationship says how much memory traffic must be *in flight* to saturate bandwidth: bytes in flight $\approx$ bandwidth $\times$ memory latency (Chapter 8).

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

**Not all operations are equal.** Additions, multiplications, divisions, and transcendental functions such as `exp` have different hardware costs. Counting each as "one FLOP" is a convenient simplification, and Chapter 2 (section 2.3) shows where it becomes misleading.

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

The arithmetic work is about $N$ additions, so each element costs one addition but moves $3s$ bytes. For FP32 that is $N$ FLOPs over $12N$ bytes, or $1/12 \approx 0.083$ FLOP per byte. That ratio is called **arithmetic intensity** (FLOPs per byte moved). It is explained fully in Chapter 2, but the pattern is already visible: very little math per byte means vector addition is commonly memory-bandwidth-bound for sufficiently large arrays.

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

## Summary

- Latency measures how long a defined unit of work takes; throughput measures how much work completes per unit time. Little's law links them: concurrency = throughput × latency.
- FLOPs count arithmetic work; FLOP/s is the rate at which that work is performed. State your counting convention.
- Logical-byte estimates describe required reads and writes under stated assumptions, not physical traffic.
- Work per byte moved (arithmetic intensity, for example about 0.08 FLOP/byte for FP32 vector add) is the idea Chapter 2 builds on.
- Effective bandwidth and achieved FLOP/s are derived from estimates plus timing, so interpretation depends on both. Chapter 2 explains why they do not, by themselves, identify a bottleneck.
- Measurements are more useful when timing boundaries, shapes, dtypes, assumptions, and methodology are recorded.

## References

- Little, J. D. C. (1961). "A Proof for the Queuing Formula: L = λW." *Operations Research* 9(3), 383–387. doi:10.1287/opre.9.3.383.
- Little, J. D. C. (2011). "Little's Law as Viewed on Its 50th Anniversary." *Operations Research* 59(3), 536–549. doi:10.1287/opre.1110.0940. A retrospective that also covers uses in computer architecture.
- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [NVIDIA Matrix Multiplication Performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)