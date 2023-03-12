import torch.nn as nn
import torch
# from torch import functorch
import math
import numpy as np

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


# class FourierFeature(nn.Module):

# def main():
#     a = 1
#     scale = 10
#     dim = 64
#     b = torch.random(size=(64, 2)) * scale

#     x = torch.tensor([1,1])
#     input_encoder(x, a, b)

class FourierPos():
    def __init__(self, a=1, scale=10, dim=64):
        self.a = a
        # self.scale = scale
        self.b = torch.random(size=(dim, 2)) * scale
        # self.map_fn = functorch.vmap(self.__input_encoder)

    def __call__(self, x):
        _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

        y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')

        yx_pairs = torch.dstack((y.flatten(-2), x.flatten(-2)))

        return torch.stack([self.a * torch.sin((2.*np.pi*yx) @ self.b.T), 
                            self.a * torch.cos((2.*np.pi*yx) @ self.b.T)], axis=-1)
    

class InputEncoder():
    def __init__(self, a=1, scale=10, dim=64):
        self.a = a
        # self.scale = scale
        self.b = torch.random(size=(dim, 2)) * scale
        # self.map_fn = functorch.vmap(self.__input_encoder)

    def __call__(self, x):
        """
            Args:
                x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        return torch.stack([self.a * torch.sin((2.*np.pi*x) @ self.b.T), 
                            self.a * torch.cos((2.*np.pi*x) @ self.b.T)], axis=-1)
        
        return self.map_fn(x)

    # def __input_encoder(self, x):
    #     return torch.stack([self.a * np.sin((2.*np.pi*x) @ self.b.T), 
    #                         self.a * np.cos((2.*np.pi*x) @ self.b.T)], axis=-1)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = FourierPos(dim=d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        #                                   dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        
        self.encoder = nn.Embedding(input_dim, d_model)
        self.decoder = nn.Linear(d_model, output_dim)
        self.d_model = d_model
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src) + src
        output = self.transformer_encoder(src, self.src_mask)
        # output = self.transformer(src, src, src_mask=self.src_mask)
        output = self.decoder(output)
        return output