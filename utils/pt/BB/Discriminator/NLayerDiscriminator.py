import torch
import functools
from torch import nn
from utils.pt.building_block import BB
from utils.pt.BB.Norm.ActNorm import ActNorm

class NLayerDiscriminator(BB):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def start(self):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        kw = int(self.kwargs.get('kw', 4))
        input_nc=self.kwargs.get('input_nc', 3)
        ndf=self.kwargs.get('ndf', 64)
        n_layers=self.kwargs.get('n_layers', 3)
        use_actnorm=self.kwargs.get('use_actnorm', False)

        self.sig = nn.Sigmoid()

        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.tanh_actfn = nn.Tanh()
    
    def d12grad(self, grad, split: str, stag: str):
        print(split, grad.mean().item(), stag)
    def forward(self, input, split):
        """
            Standard forward.
            dloss = -Expectation(ln D)
            (D=0 / fake classified) -> dloss=inf
            (D=1 / real classified) -> dloss=0
        """
        # input = torch.cat([input[:,0:1 ,:,:], input[:,1:2 ,:,:], input[:,3:4 ,:,:]], dim=1)
        # logger.critical(input.shape)
        main_out = self.main(input)
        main_out.register_hook(lambda grad: 1e6 * grad)
        main_out.register_hook(lambda grad: self.d12grad(grad, split, f'D_main_out (base)'))
        main_out = main_out - main_out.mean().detach()
        main_out.register_hook(lambda grad: self.d12grad(grad, split, f'D_main_out | {main_out.mean().item()}'))
        tanhout_out = self.tanh_actfn(main_out) # I think 3x256x256 -> 1x30x30
        tanhout_out.register_hook(lambda grad: self.d12grad(grad, split, 'D_tanhout_out'))
        return self.sig(tanhout_out/2 + 0.5)