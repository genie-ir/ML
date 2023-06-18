import torch
from torch import nn
from libs.dyimport import instantiate_from_config

class LossBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.prefix = str(self.kwargs.get('prefix', ''))
        self.prefixExtended = ''

        if self.prefix:
            self.prefixExtended = self.prefix + '_'

        setattr(self, 'forward', getattr(self, 
                                         self.prefixExtended + self.kwargs['netlossfn'], 
                                         getattr(self, f'{self.prefixExtended}loss', 
                                                 getattr(self, 'lossfn', None))))
        
        self.loss_codebook = instantiate_from_config({'target': 'configs.loss.yaml'}, kwargs={'dotdictFlag': False})
        self.start()
    
    def start(self):
        assert (not(self.loss_codebook.get(self.prefix, None) is None)), '`self.prefix={}` | does not defined in `configs.loss.yaml`'.format(self.prefix)
        self.criterion = instantiate_from_config({
            'target': self.loss_codebook[self.prefix],
            'params': self.kwargs.get('criterion', dict())
        })
        print('!!!!!!!!!!!!!', self.criterion)
        assert False
    
    def lossfn(self, y, t):
        loss = self.criterion(y, t)
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log