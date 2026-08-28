import torch
import torch.nn as nn

"""
q , k ,v --> (B,H ,N , d )9after permutation) (N--> seq_len)
qi --> (B ,H ,1 ,d) , ki, vi --> (B ,H , window_size , d)
score(i) = (B, H , 1 , window_size) (ith tokens scores) --> softmax this
out(i) = (B, H , 1 , d) 

"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d, n_heads, window_size):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        slef.window_size = window_size
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
        for i int range(N):
            start = max(0, i - self.window_size + 1)
            qi = q[: , : ,i: i+1,:]
            ki = k[:, :, start:i+1, :]
            vi = v[:, :, start:i+1, :]
            scores = qi @ ki.transpose(-2, -1)

            scores = scores / (self.head_dim ** 0.5)

            attention = torch.softmax(scores, dim=-1)
            out_i = attention @ vi
            outputs.append(out_i)

        out = torch.cat(outputs, dim=2)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.d)

        out = self.out_proj(out)
        return out