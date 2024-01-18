import torch
import functools
from torch import nn
from utils.pt.building_block import BB
from utils.pt.BB.Norm.ActNorm import ActNorm
import torchvision
from apps.VQGAN.models.learners import Activation

class NLayerDiscriminator(BB):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def start00(self):
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

        self.tanh = Activation()
        self.sig = Activation('sig')
        # self.sig = nn.Sigmoid()

        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        n_inputs = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), 
            # nn.ReLU(), 
            # self.tanh,
            # nn.Dropout(0.4),
            # nn.Linear(256, 128)
        )
        self.vgg16.features[24].weight.register_hook(lambda grad: self.d12grad(grad, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb A', f'self.vgg16'))
        # self.vgg16.features[28].weight.register_hook(lambda grad: self.d12grad(grad, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb B', f'self.vgg16'))
        # self.vgg16.classifier[6][0].weight.register_hook(lambda grad: self.d12grad(grad, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb C', f'self.vgg16'))
        # print(self.vgg16)

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
    
    def start(self):
        from apps.VQGAN.models.learners import FUM_Disc_Graph
        self.graph = FUM_Disc_Graph()
    
    def d12grad(self, grad, split: str, stag: str):
        print(split, grad.mean().item(), stag)
    
    def forward(self, input, flag, split):
        """
            Standard forward.
            dloss = -Expectation(ln D)
            (D=0 / fake classified) -> dloss=inf
            (D=1 / real classified) -> dloss=0
        """
        return self.graph(input, flag, split)
    

        # main_out = self.vgg16(input)
        # main_out.register_hook(lambda grad: 1e5 * grad)
        # main_out.register_hook(lambda grad: self.d12grad(grad, split, f'D_main_out (base) 1st'))
        # sigout = self.sig(main_out)
        # # sigout.register_hook(lambda grad: self.d12grad(grad, split, f'sig'))
        # return sigout