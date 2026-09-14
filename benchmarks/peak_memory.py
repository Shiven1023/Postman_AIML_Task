from memory_profiler import memory_usage
import torch

from attention.dense import MultiHeadAttention
from attention.sliding_window import SlidingWindowAttention
from attention.block import SparseBlockAttention
from attention.BigBird import BigBirdSparse

def benchmark_memory(forward_fn):
    def run():
        with torch.inference_mode():
            forward_fn()
    memory = memory_usage((run,), interval=0.001,retval=False)
    peak = max(memory)
    baseline = memory[0]

    return peak - baseline

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


        dense_memory = benchmark_memory(lambda: dense(x, causal_mask))
        sliding_memory = benchmark_memory(lambda: sliding(x))
        block_memory = benchmark_memory(lambda: block(x))
        bigbird_memory = benchmark_memory(lambda: bigbird(x))

        print(f"Dense: {dense_memory:.6f} MiB")
        print(f"Sliding: {sliding_memory:.6f} MiB")
        print(f"Block: {block_memory:.6f} MiB")
        print(f"BigBird: {bigbird_memory:.6f} MiB")

        # Relative to dense
        print("Relative to dense:")
        print(f"Dense: 1.00x")
        print(f"Sliding: {sliding_memory / dense_memory:.2f}x")
        print(f"Block: {block_memory / dense_memory:.2f}x")
        print(f"BigBird: {bigbird_memory / dense_memory:.2f}x")
        print()