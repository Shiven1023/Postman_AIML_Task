import torch
import torch.nn as nn

"""
q , k ,v --> (B,H ,N , d )9after permutation) (N--> seq_len)
qi --> (B ,H ,block_size ,d) , ki, vi --> (B ,H , (num_blocks+1)*block_size , d) (in the middle somewhere)
score , attention(i) --> (B, H , block_size , (num_blocks+1)*block_size)
causal mask needed to prevent score computing for the earlier values in a block with further ones in same block

"""
class SlparseBlockAttention(nn.Module):
    def __init__(self, d, n_heads, block_size,num_blocks):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.num_blocks = num_blocks #how many blocks excluding current to query with
        self.block_size = block_size

        assert d % n_heads == 0 

        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        # x: (B, N, d)
        B, N, d = x.shape

        x = self.in_proj(x)
        # Each is (B, N, d)
        q, k, v = torch.split(x, self.d, dim=-1)

        q = q.reshape(B, N, self.n_heads, self.head_dim)
        k = k.reshape(B, N, self.n_heads, self.head_dim)
        v = v.reshape(B, N, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        outputs = []
        for i in range(0,N, self.block_size):
            start = max(0, i - self.num_blocks*self.block_size)
            end = min(i + self.block_size , N)
            qi = q[: , : ,i: end,:]
            ki = k[:, :, start:end, :]
            vi = v[:, :, start:end, :]
            scores = qi @ ki.transpose(-2, -1)

            scores = scores / (self.head_dim ** 0.5)
            q_indices = torch.arange(i, end, device=x.device).unsqueeze(1) #(block,size , 1) size column vector
            k_indices = torch.arange(start, end, device=x.device).unsqueeze(0) #(1, block_size*(num_blocks+1)) size row vector 
            allowed = k_indices <= q_indices
            scores = scores.masked_fill(~allowed, float("-inf"))

            attention = torch.softmax(scores, dim=-1)
            out_i = attention @ vi
            outputs.append(out_i)

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.d)

        out = self.out_proj(out)
        return out