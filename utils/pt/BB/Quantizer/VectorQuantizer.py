import torch
import numpy as np
from torch import nn
from einops import rearrange
from utils.pt.distance import L2S_VQ
from utils.pt.building_block import BB
from utils.pt.tricks.gradfns import onehot_with_grad

class VectorQuantizer2(BB):
    '''
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    '''
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def start(self):
        n_e = self.kwargs['n_e']
        e_dim = self.kwargs['e_dim']
        beta = self.kwargs['beta']
        remap = self.kwargs.get('remap', None)
        unknown_index = self.kwargs.get('unknown_index', 'random')
        sane_index_shape = self.kwargs.get('sane_index_shape', False)
        legacy = self.kwargs.get('legacy', True)

        self.n_e = n_e
        print('~~~~~~~~~~~~~~~~~~~', self.n_e)
        assert False
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer('used', torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # 'random' or 'extra' or integer
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f'Remapping {self.n_e} indices to {self.re_embed} indices.' + ' ' + f'Using {self.unknown_index} for unknown indices.')
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

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

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, vetoFlag=False, bypass_zshape=None, I2=None):
        if z is not None:
            assert temp is None or temp==1.0, 'Only for interface compatible with Gumbel'
            assert rescale_logits==False, 'Only for interface compatible with Gumbel'
            assert return_logits==False, 'Only for interface compatible with Gumbel'
            z = rearrange(z, 'b c h w -> b h w c').contiguous() # before: z.shape=# torch.Size([2, 256, 16, 16]) | after: z.shape=torch.Size([2, 16, 16, 256])
            z_flattened = z.view(-1, self.e_dim) # torch.Size([512, 256])

            # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            #     torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            #     torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
            # min_encoding_indices = torch.argmin(d, dim=1)
            
            min_encoding_indices = L2S_VQ(z_flattened, self.embedding.weight, argminFlag=True)
            
            if vetoFlag:
                return min_encoding_indices.view(z.shape[:-1]) # (-1,16,16,256) -> (-1,16,16)
            
            _zShape = z.shape
        else:
            _zShape = [-1,16,16,256]
            # min_encoding_indices = torch.tensor(I2.round().long())

            # print('hoooooooooooo!!', I2, I2.shape, I2.dtype, I2.requires_grad)
            # print(onehot_with_grad(I2, ))
            assert False
        
        z_q = self.embedding(min_encoding_indices).view(_zShape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if z is not None:
            if not self.legacy:
                loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                    torch.mean((z_q - z.detach()) ** 2)
            else:
                loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                    torch.mean((z_q - z.detach()) ** 2)
            # preserve gradients
            z_q = z + (z_q - z).detach()
        else:
            loss = None
        
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            assert False, 'self.remap is not None'
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            assert False, 'self.sane_index_shape'
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

