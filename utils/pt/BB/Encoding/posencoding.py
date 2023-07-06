import math
import torch
from torch import nn
from utils.plots.plot1d import Plot1D
from utils.pt.building_block import BB

class PositionalEncoding(BB):
    """
        PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. 
        The positional encodings have the same dimension as the embeddings so that the two can be summed. 
        Here, we use sine and cosine functions of different frequencies.
    """
    def start(self):
        self.fwd = str(self.kwargs.get('fwd', 'epe'))
        self.maxlen = int(self.kwargs.get('maxlen', 1e3))
        self.n_scalar = int(self.kwargs.get('n_scalar', 1e4))
        self.dropout_p = float(self.kwargs.get('dropout', 0))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        
        self.pe = nn.Embedding(self.maxlen, self.embed_size)
        self.pe.weight = nn.Parameter(self.getPositionEncoding(), requires_grad=False)
        self.dropout = nn.Dropout(self.dropout_p)

        if self.fwd == 'epe':
            self.vocabsize = int(self.kwargs.get('vocabsize', 1e3))
            self.embedding = nn.Embedding(self.vocabsize, self.embed_size)
        
        setattr(self, 'forward', getattr(self, f'forward_{self.fwd}'))

    def plot(self, path, pe=None):
        pe = self.pe(torch.arange(self.maxlen)) if pe is None else pe
        plot1d = Plot1D(xlabel='x', ylabel='y', hide_axis=True, mplstyle='neon', figsize=(10,10))
        plot1d.plot(y=pe, grid=True)
        plot1d.savefig(path)
    
    def getPositionEncoding(self):
        P = torch.zeros((self.maxlen, self.embed_size))
        for k in range(self.maxlen):
            for i in torch.arange(self.embed_size):
                denominator = self.n_scalar ** ((2*i)/self.embed_size)
                if i % 2 == 0:
                    P[k, i] = (k/denominator).sin() # original in paper
                else:
                    # P[k, i] = (k/denominator).sin()
                    P[k, i] = (k/denominator).cos() # original in paper
        return P
 
    def forward_pe(self, x):
        return self.pe(x)
    
    def forward_rpe(self, x):
        return self.dropout(x + self.pe(x))
    
    def forward_epe(self, x):
        print('epe @@@@@@@@@@@@@@@@@@', x.shape)
        return self.dropout(self.embedding(x) + self.pe(x))