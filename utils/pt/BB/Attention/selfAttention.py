import torch
from torch import nn
from utils.pt.building_block import BB

class SelfAttention(BB):
    def start(self):
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.heads = int(self.kwargs.get('heads', 8))
        self.head_dim = int(self.embed_size // self.heads)
        assert self.head_dim * self.heads == self.embed_size

        self.ninf = float('-1e20')
        self.normalizer_fraction = self.head_dim ** .5
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def tri_bool_mask(self, x, diagonal=0):
        return torch.ones_like(x).triu(diagonal).bool()
    
    def forward(self, v, k, q, mask=False):
        N = q.shape[0]
        vlen, klen, qlen = v.shape[1], k.shape[1], q.shape[1]

        v = self.V(v.reshape(N, vlen, self.heads, self.head_dim))
        k = self.K(k.reshape(N, klen, self.heads, self.head_dim))
        q = self.Q(q.reshape(N, qlen, self.heads, self.head_dim))


        print('********', k.shape,v.shape,q.shape, k.device,v.device,q.device)
        energy = torch.einsum('nqhd,nkhd->nhqk', q, k)
        print('$$$$', energy.shape, energy.device)
        assert False

        if mask:
            energy = energy.masked_fill(self.tri_bool_mask(energy), self.ninf)

        print('0 ------->', energy.shape)
        self_attention = torch.softmax(energy / self.normalizer_fraction, dim=-1) # dim=3
        print('1 ------->', self_attention.shape)

        out = torch.einsum('nhql,nlhd->nqhd', self_attention, v).reshape(N, qlen, self.head * self.head_dim)
        return self.fc_out(out)