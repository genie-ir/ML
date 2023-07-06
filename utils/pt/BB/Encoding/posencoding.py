import math
import torch
from torch import nn
from utils.pt.building_block import BB

class PositionalEncoding(BB):
    """
        PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence. 
        The positional encodings have the same dimension as the embeddings so that the two can be summed. 
        Here, we use sine and cosine functions of different frequencies.
    """
    def start(self):
        self.max_len = int(self.kwargs.get('max_len', 1e3))
        self.n_scalar = int(self.kwargs.get('n_scalar', 1e4))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.pe = self.getPositionEncoding()
        print('----------->', self.pe.shape, self.pe.requires_grad)
        from utils.plots.plot1d import Plot1D
        plot1d = Plot1D(xlabel='x', ylabel='y', mplstyle='@Solarize_Light2', grid_args_dict={}, figsize=(10,10))
        plot1d.plot(y=self.pe, grid=True, label='')
        plot1d.savefig('/content/a.png')
        assert False

    def getPositionEncoding(self):
        P = torch.zeros((self.max_len, self.embed_size))
        for k in range(self.max_len):
            for i in torch.arange(self.embed_size):
                denominator = self.n_scalar ** ((2*i)/self.embed_size)
                if i % 2 == 0:
                    P[k, i] = (k/denominator).sin() # original in paper
                else:
                    # P[k, i] = (k/denominator).sin()
                    P[k, i] = (k/denominator).cos() # original in paper
        return P
 
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        j= self.dropout(x + self.pe[:x.size(0)])

        print('@@@@@@@@@@@@@@@@@@@@@@@@@', j.shape)

        return j