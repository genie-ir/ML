import torch
from torch import nn
from libs.dyimport import instantiate_from_config

class LossBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.prefix = str(self.kwargs.get('criterion', ''))
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
        self.criterion = instantiate_from_config({
            'target': self.loss_codebook.get(self.prefix, self.prefix), # `criterion` in yaml file
            'params': self.kwargs.get('params', dict())
        })
    
    def lossfn_p1log(self, y, t):
        return (self.criterion(y, t) + 1).log()

    def lossfn(self, y, t):
        loss = self.criterion(y, t)
        log = {
            'loss': loss.clone().detach().mean(),
        }
        return loss, log
    
    def lossdict(self, **loss_items):
        return dict((loss_name, loss_value.clone().detach().mean()) for loss_name, loss_value in loss_items.items())