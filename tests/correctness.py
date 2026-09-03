import torch
import random

from attention.dense import MultiHeadAttention
from attention.sliding_window import SlidingWindowAttention
from attention.block import SparseBlockAttention
from attention.BigBird import BigBirdSparse


# make all models use the same projection weights 
def copy_weights(source, target):
    target.in_proj.load_state_dict(source.in_proj.state_dict())
    target.out_proj.load_state_dict(source.out_proj.state_dict())


# Sliding-window mask
#False --> allowed

def make_sliding_mask(N, window_size, device):
    mask = torch.ones(N, N,dtype=torch.bool,device=device)
    for i in range(N):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = False

    return mask
#block sparse mask
def make_block_mask(N, block_size, num_blocks, device):
    mask = torch.ones(N, N,dtype=torch.bool,device=device)
    for i in range(0, N, block_size):
        end = min(i + block_size, N)
        start = max(0,i - num_blocks * block_size)
        for idx in range(i, end):
            mask[idx, start:idx + 1] = False

    return mask

#BigBird pattern dictrionary (block_num , used_blocks)
def build_bigbird_pattern(N,block_size,num_local_blocks,num_random_blocks,seed=42):
    rng = random.Random(seed)
    pattern = {}
    num_total_blocks = (N + block_size - 1) // block_size 

    for block_idx in range(num_total_blocks):
        used = []
        # Local 
        start_block = max(0,block_idx - num_local_blocks)

        for block in range(start_block,block_idx + 1):
            used.append(block)

        # Global block (Block 0)
        if 0 not in used:
            used.append(0)
        # Random past blocks
        possible = []

        for block in range(block_idx):
            if block not in used:
                possible.append(block)

        num_to_pick = min(
            num_random_blocks,
            len(possible)
        )

        random_blocks = rng.sample(
            possible,
            num_to_pick
        )

        for block in random_blocks:
            used.append(block)

        pattern[block_idx] = used

    return pattern


# Turn BigBird connectivity into an N x N  mask

def make_bigbird_mask(
    N,
    block_size,
    pattern,
    device
):
    mask = torch.ones(N, N,dtype=torch.bool,device=device)

    for block_idx, used_blocks in pattern.items():

        q_start = block_idx * block_size
        q_end = min(q_start + block_size, N)

        for q_idx in range(q_start, q_end):

            for key_block in used_blocks:
                k_start = key_block * block_size
                k_end = min(k_start + block_size,N)

                for k_idx in range(k_start, k_end):
                    # causal constraint
                    if k_idx <= q_idx:
                        mask[q_idx, k_idx] = False

    return mask


# TESTS

def test_sliding(x, d, heads, window_size):

    dense = MultiHeadAttention(d, heads)
    sparse = SlidingWindowAttention(
        d,
        heads,
        window_size
    )

    # Make projection weights identical
    copy_weights(dense, sparse)

    N = x.shape[1]

    mask = make_sliding_mask(
        N,
        window_size,
        x.device
    )

    dense_out = dense(x, mask)
    sparse_out = sparse(x)

    torch.testing.assert_close(
        dense_out,
        sparse_out,
        atol=1e-5,
        rtol=1e-4
    )

    print("[PASS] Sliding-window attention")


def test_block(
    x,
    d,
    heads,
    block_size,
    num_blocks
):

    dense = MultiHeadAttention(d, heads)

    sparse = SparseBlockAttention(
        d,
        heads,
        block_size,
        num_blocks
    )

    copy_weights(dense, sparse)

    N = x.shape[1]

    mask = make_block_mask(
        N,
        block_size,
        num_blocks,
        x.device
    )

    dense_out = dense(x, mask)
    sparse_out = sparse(x)

    torch.testing.assert_close(
        dense_out,
        sparse_out,
        atol=1e-5,
        rtol=1e-4
    )

    print("[PASS] Block-sparse attention")


def test_bigbird(x,d,heads,block_size,num_local_blocks,num_random_blocks):

    dense = MultiHeadAttention(d, heads)

    sparse = BigBirdSparse(
        d,
        heads,
        block_size,
        num_local_blocks,
        num_random_blocks
    )
    copy_weights(dense, sparse)
    N = x.shape[1]
    pattern = build_bigbird_pattern(
        N,
        block_size,
        num_local_blocks,
        num_random_blocks,
        seed=42
    )

    mask = make_bigbird_mask(N,block_size,pattern,x.device)

    dense_out = dense(x, mask)
    sparse_out = sparse(x)

    torch.testing.assert_close(
        dense_out,
        sparse_out,
        atol=1e-5,
        rtol=1e-4
    )

    print("[PASS] BigBird-style attention")


if __name__ == "__main__":
    torch.manual_seed(42)
    B = 2
    N = 16
    d = 32
    heads = 4
    x = torch.randn(B, N, d)
    test_sliding(
        x,
        d,
        heads,
        window_size=4
    )
    test_block(
        x,
        d,
        heads,
        block_size=4,
        num_blocks=1
    )

    test_bigbird(
        x,
        d,
        heads,
        block_size=4,
        num_local_blocks=1,
        num_random_blocks=1
    )

    print("\nAll correctness tests passed.")