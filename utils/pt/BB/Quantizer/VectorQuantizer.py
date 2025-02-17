import torch
import numpy as np
from torch import nn
from einops import rearrange
from utils.pt.distance import L2S
from utils.pt.building_block import BB
from utils.pt.tricks.gradfns import onehot_with_grad, dzq_dz_eq1

class VectorQuantizer(BB):
    '''
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    '''
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def start(self):
        self.dim = int(self.kwargs.get('dim', 0))
        self.ncluster = int(self.kwargs.get('ncluster', 0))

        self.n_e = self.ncluster or int(self.kwargs['n_e'])
        self.e_dim = self.dim or int(self.kwargs['e_dim'])
        self.beta = float(self.kwargs.get('beta', 0.25))
        self.remap = self.kwargs.get('remap', None)
        self.legacy = bool(self.kwargs.get('legacy', True))
        unknown_index = self.kwargs.get('unknown_index', 'random')
        self.zwh = int(self.kwargs.get('zwh', 16))
        self.zch = int(self.kwargs.get('zch', self.e_dim))
        # currently in my codes assumes that `zch` & `e_dim` are `equals` if in your specefic case its not you must code for that
        assert self.zch == self.e_dim, f'`self.zch={self.zch}` != `self.e_dim={self.e_dim}`'
        self.zshape = [-1, self.zwh, self.zwh, self.zch]
        self.sane_index_shape = bool(self.kwargs.get('sane_index_shape', False))

        self.eshape = (self.n_e, self.e_dim)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding_weight_init()

        setattr(self, 'forward', self.fwd)

        if self.remap is not None:
            self.register_buffer('used', torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # 'random' or 'extra' or integer
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f'Remapping {self.n_e} indices to {self.re_embed} indices.' + ' ' + f'Using {self.unknown_index} for unknown indices.')
        else:
            self.re_embed = self.n_e

    def embedding_weight_init(self):
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
    
    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == 'random':
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def fwd0(self, z): # NOTE: orginal code
        z = rearrange(z.float(), 'b c h w -> b h w c').contiguous() # before: z.shape=# torch.Size([2, 256, 16, 16]) | after: z.shape=torch.Size([2, 16, 16, 256])
        z_flattened = z.view(-1, self.e_dim) # torch.Size([512, 256])
        min_encoding_indices = L2S(z_flattened, self.embedding.weight, argmin=True)
        z_q = self.embedding(min_encoding_indices).view(self.zshape) # phi
        print('---->', z_q.requires_grad)
        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients 
        print('z_q BBB', z_q.requires_grad)
        z_q = dzq_dz_eq1(z_q, z)
        # reshape back to match original input shape
        print('z_q AAA', z_q.requires_grad)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        print('z_q last', z_q.requires_grad)
        return z_q, loss
    def fwd(self, z): # NOTE: only use in my distersion
        z = rearrange(z.float(), 'b c h w -> b h w c').contiguous() # before: z.shape=# torch.Size([2, 256, 16, 16]) | after: z.shape=torch.Size([2, 16, 16, 256])
        z_flattened = z.view(-1, self.e_dim) # torch.Size([512, 256])
        min_encoding_indices = L2S(z_flattened, self.embedding.weight, argmin=True)
        z_q = self.embedding(min_encoding_indices).view(self.zshape) # phi
        # print('---->', z_q.requires_grad)
        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients 
        # print('z_q BBB', z_q.requires_grad)
        # z_q = dzq_dz_eq1(z_q, z)
        # reshape back to match original input shape
        # print('z_q AAA', z_q.requires_grad)
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # print('z_q last', z_q.requires_grad)
        return z_q, loss
    
    def fwd_getIndices(self, z, argmin=True, topk=False):
        z = rearrange(z.float(), 'b c h w -> b h w c').contiguous() # before: z.shape=# torch.Size([2, 256, 16, 16]) | after: z.shape=torch.Size([2, 16, 16, 256])
        z_flattened = z.view(-1, self.e_dim) # torch.Size([512, 256])
        min_encoding_indices = L2S(z_flattened, self.embedding.weight, argmin=argmin, topk=topk)
        if argmin:
            return min_encoding_indices.view(z.shape[:-1]) # (-1,16,16,256) -> (-1,16,16)
        else:
            return min_encoding_indices
        
    def fwd_bpi(self, idx):
        """`idx` should be `float or somthing meaningful` since `grad` can be `backprob` in it"""
        z_q = (onehot_with_grad(idx.squeeze(), self.n_e) @ self.embedding.weight).view(self.zshape)
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q

    def fwd_nbpi(self, idx):
        """get_codebook_entry function"""
        return self.embedding(idx)