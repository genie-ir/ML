import torch
from torch import nn
from utils.pt.building_block import BB
from utils.pt.BB.Attention.selfAttention import SelfAttention

class TransformerBlock(BB):
    def start(self):
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.heads = int(self.kwargs.get('heads', 8))
        self.fwd_expan = int(self.kwargs.get('fwd_expan', 4))
        self.dropout_p = float(self.kwargs.get('dropout', 0))

        self.self_attention = SelfAttention(embed_size=self.embed_size, heads=self.heads)
        self.dropout = nn.Dropout(self.dropout_p)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)
        self.feedfwd = nn.Sequential(
            nn.Linear(self.embed_size, self.fwd_expan * self.embed_size),
            nn.ReLU(),
            nn.Linear(self.fwd_expan * self.embed_size, self.embed_size)
        )

    def forward(self, v, k, q, mask):
        x = self.dropout(self.norm1(q + self.self_attention(v=v, k=k, q=q, mask=mask)))
        return self.dropout(self.norm2(x + self.feedfwd(x)))

class Encoder(BB):
    def start(self):
        self.heads = int(self.kwargs.get('heads', 8))
        self.maxlen = int(self.kwargs.get('maxlen', 1e3))
        self.fwd_expan = int(self.kwargs.get('fwd_expan', 4))
        self.dropout_p = float(self.kwargs.get('dropout', 0))
        self.num_layers = int(self.kwargs.get('num_layers', 1))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.src_vocab_size = int(self.kwargs.get('src_vocab_size', 1e3))

        self.pos_encoding = nn.Embedding(self.maxlen, self.embed_size) # TODO this shoude be change to positional encoding not embedding
        self.word_embedding = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.layers = nn.Sequential(*[
            TransformerBlock(
                heads=self.heads,
                dropout=self.dropout_p,
                fwd_expan=self.fwd_expan,
                embed_size=self.embed_size
            )
            for n in range(self.num_layers)
        ])

    def forward(self, x, mask):
        print('.....................', x.shape, x.device, torch.arange(seqlen).expand(N, -1).device)
        N, seqlen = x.shape
        outs = []
        x = self.dropout(self.word_embedding(x) + self.pos_encoding(torch.arange(seqlen).expand(N, -1)))
        for layer in self.layers:
            x = layer(v=x, k=x, q=x, mask=mask)
            outs.append(x)

        return outs

class DecoderBlock(BB):
    def start(self):
        self.heads = int(self.kwargs.get('heads', 8))
        self.fwd_expan = int(self.kwargs.get('fwd_expan', 4))
        self.dropout_p = float(self.kwargs.get('dropout', 0))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.norm = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.self_attention = SelfAttention(embed_size=self.embed_size, heads=self.heads)
        self.transformer_block = TransformerBlock(
            heads=self.heads,
            dropout=self.dropout_p,
            fwd_expan=self.fwd_expan,
            embed_size=self.embed_size
        )

    def forward(self, x, v, k, src_mask, trg_mask):
        return self.transformer_block(
            v=v, k=k, mask=src_mask,
            q=self.dropout(self.norm(x + self.self_attention(v=x, k=x, q=x, mask=trg_mask)))
        )

class Decoder(BB):
    def start(self):
        self.heads = int(self.kwargs.get('heads', 8))
        self.maxlen = int(self.kwargs.get('maxlen', 1e3))
        self.fwd_expan = int(self.kwargs.get('fwd_expan', 4))
        self.dropout_p = float(self.kwargs.get('dropout', 0))
        self.num_layers = int(self.kwargs.get('num_layers', 1))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.trg_vocab_size = int(self.kwargs.get('trg_vocab_size', 1e3))

        self.pos_encoding = nn.Embedding(self.maxlen, self.embed_size)
        self.word_embedding = nn.Embedding(self.trg_vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc_out = nn.Linear(self.embed_size, self.trg_vocab_size)
        self.layers = nn.Sequential(*[
            DecoderBlock(
                heads=self.heads,
                dropout=self.dropout_p,
                fwd_expan=self.fwd_expan,
                embed_size=self.embed_size
            )
            for n in range(self.num_layers)
        ])

    def forward(self, x, encoder_out, src_mask, trg_mask):
        print('!!!!!!!!!!!', len(encoder_out), encoder_out[0].shape)
        N, seqlen = x.shape
        x = self.dropout(self.word_embedding(x) + self.pos_encoding(torch.arange(seqlen).expand(N, -1)))
        
        for idx, layer in enumerate(self.layers):
            x = layer(x=x, v=encoder_out[idx], k=encoder_out[idx], src_mask=src_mask, trg_mask=trg_mask)

        return self.fc_out(x)


class Transformer(BB):
    def start(self):
        self.heads = int(self.kwargs.get('heads', 8))
        self.maxlen = int(self.kwargs.get('maxlen', 1e3))
        self.dropout_p = float(self.kwargs.get('dropout', 0))
        self.fwd_expan = int(self.kwargs.get('fwd_expan', 4))
        self.num_layers = int(self.kwargs.get('num_layers', 8))
        self.embed_size = int(self.kwargs.get('embed_size', 256))
        self.src_mask = bool(self.kwargs.get('src_mask', False))
        self.trg_mask = bool(self.kwargs.get('trg_mask', True))
        # self.src_pad_idx = int(self.kwargs.get('src_pad_idx', -1))
        # self.trg_pad_idx = int(self.kwargs.get('trg_pad_idx', -1))
        self.src_vocab_size = int(self.kwargs.get('src_vocab_size', 1e3))
        self.trg_vocab_size = int(self.kwargs.get('trg_vocab_size', 1e3))

        self.encoder = Encoder(
            heads=self.heads,
            maxlen=self.maxlen,
            dropout=self.dropout_p,
            fwd_expan=self.fwd_expan,
            num_layers=self.num_layers,
            embed_size=self.embed_size,
            src_vocab_size=self.src_vocab_size
        )
        self.decoder = Decoder(
            heads=self.heads,
            maxlen=self.maxlen,
            dropout=self.dropout_p,
            fwd_expan=self.fwd_expan,
            num_layers=self.num_layers,
            embed_size=self.embed_size,
            trg_vocab_size=self.trg_vocab_size
        )

    def forward(self, src, trg):
        return self.decoder(trg, self.encoder(src, self.src_mask), self.src_mask, self.trg_mask)

