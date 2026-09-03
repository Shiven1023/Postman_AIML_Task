import torch
import torch.nn as nn
import random 

"""
q , k ,v --> (B,H ,N , d )9after permutation) (N--> seq_len)
qi --> (B ,H ,block_size ,d) 
ki, vi (local (same as block sparse))
for random choose num_random_blocks number of random blocks from the avaiable blocks
maintaining "used" for all blocks already used for querying 

"""
class BigBirdSparse(nn.Module):
    def __init__(self, d, n_heads, block_size,num_blocks, num_random_blocks):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.num_blocks = num_blocks #how many blocks excluding current to query with
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks

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
        rng = random.Random(42)
        for i in range(0,N, self.block_size):
            start_local = max(0, i - self.num_blocks*self.block_size)
            end = min(i + self.block_size , N)
            block_idx = i// self.block_size
            start_idx = start_local// self.block_size
            qi = q[: , : ,i: end,:]
            ki = k[:, :, start_local:end, :]
            vi = v[:, :, start_local:end, :]
            used = []
            for block in range(start_idx , block_idx +1):
                used.append(block)

            if start_idx > 0:
                ki = torch.cat([ki , k[:,:,0:self.block_size,:]] , dim = 2) #global block (first block in this case)
                vi = torch.cat([vi , v[:,:,0:self.block_size,:]] , dim = 2)
                used.append(0)
            possible = []
            for _ in range(0, block_idx +1):
                if _ not in used:
                    possible.append(_)
            num_to_pick = min(self.num_random_blocks, len(possible))
            
            random_blocks = rng.sample(possible,num_to_pick)
            for block_num in random_blocks:
                ki = torch.cat([ki,k[:,:,self.block_size*block_num: min(N,self.block_size*block_num+ self.block_size), :]], dim = 2)
                vi = torch.cat([vi,v[:,:,self.block_size*block_num: min(N,self.block_size*block_num+ self.block_size), :]], dim = 2)
                used.append(block_num)
            scores = qi @ ki.transpose(-2,-1)

            scores = scores / (self.head_dim ** 0.5)
            q_indices = torch.arange(i, end, device=x.device).unsqueeze(1) #(block,size , 1) size column vector
            k_indices = []
            for block_num in used:
                block_start = block_num* self.block_size
                block_end  =min(N ,block_start + self.block_size)
                k_indices.append(torch.arange(block_start, block_end,device = x.device))
            
            k_indices = torch.cat(k_indices).unsqueeze(0)
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