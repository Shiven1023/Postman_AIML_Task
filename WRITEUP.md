# Sparse Attention from Scratch

## 1. Attention Implementations

The goal of this project was to implement dense and sparse variants of multi-head self-attention from scratch in PyTorch and compare their correctness, runtime behavior, and effect on language-model quality.

All implementations use standard multi-head attention. Given an input

$$
X \in \mathbb{R}^{B \times N \times d},
$$

linear projections produce queries, keys, and values, which are split into multiple heads:

$$
Q,K,V \in \mathbb{R}^{B \times H \times N \times d_h}.
$$

Attention scores are then computed as

$$
S = \frac{QK^T}{\sqrt{d_h}},
$$

followed by masking, softmax, multiplication with \(V\), and a final output projection.

### Dense Causal Attention

Dense attention computes the full \(N \times N\) attention matrix for every head.

A causal mask ensures that token \(i\) can only attend to positions \(j \le i\). Therefore, every token has direct access to its entire previous context.

The main drawback is that both computation and attention-memory usage grow quadratically with sequence length.

---

### Sliding-Window Attention

Sliding-window attention restricts each token to a fixed-size local context.

For a window size \(W\), token \(i\) attends only to

$$
\max(0,i-W+1),\ldots,i.
$$

Instead of computing the complete \(N \times N\) attention matrix, the implementation only computes attention over the selected local window.

This reduces the theoretical attention complexity from approximately

$$
O(N^2)
$$

to

$$
O(NW).
$$

The cost is that distant tokens cannot communicate directly.

---

### Block-Sparse Attention

The block-sparse implementation divides the sequence into blocks of fixed size.

Each query block attends to:

* itself,
* a fixed number of previous blocks.

For example, with one local previous block, block \(B_i\) attends to \(B_{i-1}\) and \(B_i\).

Causality still has to be enforced inside the current block. Otherwise, an earlier token inside a block could attend to a later token in the same block.

This is handled by comparing the actual query and key positions and only allowing

$$
\text{key position} \le \text{query position}.
$$

Block sparsity also has the practical benefit of operating on larger chunks rather than processing one token at a time.

---

### BigBird-Style Sparse Attention

The BigBird-style implementation combines three types of connections:

1. **Local blocks** — each query block attends to nearby previous blocks.
2. **Global block** — the first block is visible to later blocks.
3. **Random blocks** — each query block also attends to randomly selected previous blocks.

The random blocks provide long-range shortcuts that pure local attention does not have.

The global block is also useful because information near the beginning of the sequence can be accessed directly by many later tokens.

Since this implementation is causal, the first block cannot attend to future blocks. It therefore acts more like a shared early-context block than a fully bidirectional global block.

---

## 2. Correctness Harness

To test the sparse implementations, dense multi-head attention was modified to accept a Boolean mask as input.

The mask convention is:

* `False` — connection is allowed,
* `True` — connection is masked.

This makes dense attention useful as a reference implementation for different sparsity patterns.

For example, to test sliding-window attention, a dense mask is constructed that only allows the same local window used by the sparse implementation. The same idea is used for block-sparse and BigBird-style attention.

The correctness procedure is:

1. Create the dense and sparse models.
2. Copy the same attention projection weights into both.
3. Give both the same input tensor.
4. Construct a dense mask representing the sparse pattern.
5. Run dense attention using that mask.
6. Run the sparse implementation.
7. Compare their final outputs.

The comparison uses `torch.testing.assert_close` with small numerical tolerances.

All implemented sparse variants matched their corresponding dense masked references.

This was especially useful because comparing against unrestricted dense attention would not be correct. Sparse attention changes which values participate in the softmax, so the reference must use exactly the same allowed connections.

---

## 3. Avoiding NaNs

A possible issue with masked attention is that a query might have every attention score masked to

$$
-\infty.
$$

Softmax over such a row produces `NaN`.

This can happen in sparse attention if a sparsity pattern introduces keys from positions further ahead in the sequence and those keys are later removed by the causal mask.

The implementations avoid this by always allowing every token to attend to itself.

Therefore, every query has at least one valid key after causal masking, which prevents an all-masked softmax row.

---

## 4. Wall-Clock Runtime Benchmark

The attention implementations were benchmarked on CPU for sequence lengths from 512 to 8192.

The measured forward-pass times were:

| Sequence Length |      Dense |    Sliding |      Block |    BigBird |
| --------------: | ---------: | ---------: | ---------: | ---------: |
|             512 | 0.001980 s | 0.012397 s | 0.000775 s | 0.001411 s |
|            1024 | 0.004873 s | 0.024489 s | 0.001403 s | 0.002588 s |
|            2048 | 0.021321 s | 0.050066 s | 0.002629 s | 0.005327 s |
|            4096 | 0.104258 s | 0.099276 s | 0.005114 s | 0.011015 s |
|            8192 | 0.396511 s | 0.199831 s | 0.011004 s | 0.024912 s |

Relative to dense attention:

| Sequence Length | Sliding | Block | BigBird |
| --------------: | ------: | ----: | ------: |
|             512 |   6.26x | 0.39x |   0.71x |
|            1024 |   5.03x | 0.29x |   0.53x |
|            2048 |   2.35x | 0.12x |   0.25x |
|            4096 |   0.95x | 0.05x |   0.11x |
|            8192 |   0.50x | 0.03x |   0.06x |

The overall growth is consistent with the expected behavior.

Dense attention becomes much more expensive as sequence length increases because it computes all pairwise token interactions.

Sliding-window attention grows much more slowly because the window size stays fixed. However, for small sequence lengths, it is actually slower than dense attention.

This is mainly an implementation effect. Dense attention uses one large, highly optimized matrix multiplication, while the sliding-window implementation contains a Python loop over sequence positions and performs many smaller matrix multiplications.

Because of that overhead, sliding attention is over 6x slower than dense at \(N=512\), even though it computes fewer theoretical attention scores.

As sequence length increases, the quadratic growth of dense attention eventually dominates. Around \(N=4096\), sliding attention becomes roughly equal in runtime, and at \(N=8192\) it is about twice as fast as dense attention.

Block-sparse attention performs particularly well because it combines sparsity with larger block-level matrix multiplications rather than using one loop iteration per token.

BigBird is slower than simple block sparsity because it additionally gathers global and random blocks, but it still scales much better than dense attention.

These results highlight an important distinction:

> Lower theoretical complexity does not automatically mean faster execution. The way the computation is implemented and how well it maps to optimized matrix operations also matters.

A peak-memory benchmark was also attempted. The CPU version uses sampled process memory, so it is only an approximate measurement and is less reliable than CUDA allocator-based peak-memory measurements. For this reason, the memory results should be treated as rough rather than exact. Final benchmark plots were also not included.

---

## 5. Quality Evaluation on TinyShakespeare

To test whether sparse attention significantly affects model quality, the attention implementations were used inside a small character-level GPT trained on TinyShakespeare.

The model contains:

* character embeddings,
* learned positional embeddings,
* two Transformer decoder blocks,
* pre-normalization,
* feed-forward layers,
* a final LayerNorm,
* a linear language-model head.

The model architecture was kept the same across experiments. Only the attention mechanism was changed.

The main configuration was:

* context length: 256,
* batch size: 32,
* embedding dimension: 128,
* attention heads: 4,
* Transformer layers: 2,
* optimizer: AdamW,
* learning rate: \(3\times10^{-4}\),
* training steps: 1000.

For sliding-window attention, the window size was 64.

For BigBird-style attention, the block size was 32 with one previous local block and one random block.

The final validation losses were:

| Attention      | Validation Loss |
| -------------- | --------------: |
| Dense          |      **2.3981** |
| Sliding Window |      **2.3372** |
| BigBird-style  |      **2.3706** |

All three variants trained successfully and achieved fairly similar quality.

Sliding-window attention achieved the lowest validation loss in this experiment, followed by BigBird and dense attention.

This does not mean sparse attention is generally better than dense attention. The experiment uses a small model, a short context length, one training seed, and only 1000 training steps.

Character-level language modeling also depends strongly on local information. Nearby characters are important for spelling, word structure, punctuation, and short-range syntax. A local window of 64 characters therefore already contains a large amount of useful context.

BigBird adds longer-range random and global connections, but at this scale those extra connections do not appear necessary enough to outperform the simpler sliding-window pattern.

---

## 6. Main Takeaways

Dense attention gives every token direct access to its entire previous context, but its cost grows quadratically with sequence length.

Sliding-window attention greatly reduces the number of interactions and works well when useful information is mostly local. Its simple implementation can still be slow for shorter sequences because of Python-loop overhead.

Block-sparse attention provides strong runtime performance because the computation remains sparse while still using reasonably large matrix multiplications.

BigBird-style attention adds global and random connections to local attention. These connections provide long-range shortcuts and can help information travel across the sequence without restoring full dense attention.

The TinyShakespeare experiment also showed that sparse attention does not necessarily cause a large drop in model quality. In this small character-level setting, both sparse variants matched or slightly outperformed dense attention.

Overall, the experiments show that attention sparsity can reduce computation substantially while preserving useful model behavior, but practical performance depends both on the chosen sparsity pattern and on how efficiently that pattern is implemented.
