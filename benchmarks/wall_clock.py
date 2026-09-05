import time
import torch

from attention.dense import MultiHeadAttention
from attention.sliding_window import SlidingWindowAttention
from attention.block import SparseBlockAttention
from attention.BigBird import BigBirdSparse

def benchmark_time(model, warmup=3, runs=5):
    # Warmup runs, not measured
    with torch.inference_mode():
        for _ in range(warmup):
            model()
    times = []
    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()
            model()
            end = time.perf_counter()
            times.append(end - start)

    return sum(times) / len(times)

#True --> masked
def make_causal_mask(N):
    return torch.triu(torch.ones(N, N, dtype=torch.bool),diagonal=1)

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 1
    d = 64
    heads = 4
    window_size = 128
    block_size = 64
    num_local_blocks = 1
    num_random_blocks = 1
    seq_lens = [512,1024,2048,4096,8192]
    print("Device: CPU")

    for N in seq_lens:
        print(f"Sequence length: {N}")
        x = torch.randn(B, N, d)
        # Models
        dense = MultiHeadAttention(d, heads)
        sliding = SlidingWindowAttention(d,heads,window_size)
        block = SparseBlockAttention(d,heads,block_size,num_local_blocks)
        bigbird = BigBirdSparse(d,heads,block_size,num_local_blocks,num_random_blocks)
        causal_mask = make_causal_mask(N)

        # Lambda function called inside benchmark_time
        dense_time = benchmark_time(lambda: dense(x, causal_mask))

        sliding_time = benchmark_time(lambda: sliding(x))
        block_time = benchmark_time(lambda: block(x))
        bigbird_time = benchmark_time(lambda: bigbird(x))

        print(f"Dense: {dense_time:.6f} s")
        print(f"Sliding: {sliding_time:.6f} s")
        print(f"Block: {block_time:.6f} s")
        print(f"BigBird: {bigbird_time:.6f} s")

        # Relative to dense
        print("Relative to dense:")
        print(f"Dense: 1.00x")
        print(f"Sliding: {sliding_time / dense_time:.2f}x")
        print(f"Block: {block_time / dense_time:.2f}x")
        print(f"BigBird: {bigbird_time / dense_time:.2f}x")
        print()