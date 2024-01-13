import torch
import functools
from torch import nn
from utils.pt.building_block import BB
from utils.pt.BB.Norm.ActNorm import ActNorm
import torchvision

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
        # kw = int(self.kwargs.get('kw', 4))
        # input_nc=self.kwargs.get('input_nc', 3)
        # ndf=self.kwargs.get('ndf', 64)
        # n_layers=self.kwargs.get('n_layers', 3)
        # use_actnorm=self.kwargs.get('use_actnorm', False)

        self.sig = nn.Sigmoid()
        self.tanh_actfn = nn.Tanh()

        self.vgg16_head = torchvision.models.vgg16(pretrained=True).features
        self.vgg16 = nn.Sequential(
            self.vgg16_head,
            nn.Conv2d(512, 512, 3,2,1), # 8x8 -> 4x4
            nn.Sigmoid()
        )
        self.vgg16[0][0].weight.register_hook(lambda grad: self.d12grad(grad, 'split', f'self.vgg16[0][0]'))
        print(self.vgg16)

        # if not use_actnorm:
        #     norm_layer = nn.BatchNorm2d
        # else:
        #     norm_layer = ActNorm
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        #     use_bias = norm_layer != nn.BatchNorm2d

        
        # padw = 1
        # sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # nf_mult = 1
        # nf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence += [
        #         nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        #         norm_layer(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]

        # sequence += [
        #     nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # self.main = nn.Sequential(*sequence)

    def main(self, x):
        return x
    
    
    def d12grad(self, grad, split: str, stag: str):
        print(split, grad.mean().item(), stag)
    
    def forward(self, input, split):
        """
            Standard forward.
            dloss = -Expectation(ln D)
            (D=0 / fake classified) -> dloss=inf
            (D=1 / real classified) -> dloss=0
        """
        
        main_out = self.vgg16(input)
        main_out.register_hook(lambda grad: self.d12grad(grad, split, f'D_main_out (base) 1st'))
        
        # main_out.register_hook(lambda grad: 1e6 * grad)
        # main_out.register_hook(lambda grad: self.d12grad(grad, split, f'D_main_out (base) 2nd'))
        return main_out