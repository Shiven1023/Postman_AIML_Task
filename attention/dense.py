import torch
import torch.nn as nn

"""
q , k ,v --> (B,H ,N , d )9after permutation) (N--> seq_len)
attention --> (B, H , seq_len , seq_len)
mask --> (1 , 1 , seq_len , seq_len) (broadcasted)
score --> (B , H , seq_len , d) (attention @ v)

"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
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
        
        k_t = k.transpose(-2, -1) 
        attention = q @ k_t
        attention = attention / (self.head_dim ** 0.5)
        
        mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), #causal mask (broadcasts across B ,H)
            diagonal=1
        )
        attention = attention.masked_fill(mask, float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        score = attention @ v 

        score = score.permute(0, 2, 1, 3)
        score = score.reshape(B, N, self.d)

        out = self.out_proj(score)

        # (B,N,d)
        return out