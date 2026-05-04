import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerTS(nn.Module):
    """
    Older Transformer model (represents Autoformer/FEDformer era time-series transformers)
    """
    def __init__(self, seq_len, pred_len, channels, d_model=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.enc_embedding = nn.Linear(channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2, batch_first=True),
            num_layers=1
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * d_model, pred_len * channels)
        )

    def forward(self, x, prompt=None):
        enc_in = self.enc_embedding(x)
        enc_in = self.pos_encoder(enc_in)
        enc_out = self.transformer(enc_in)
        dec_out = self.decoder(enc_out)
        return dec_out.view(-1, self.pred_len, self.channels)
