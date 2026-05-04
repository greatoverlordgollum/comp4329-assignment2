import torch
import torch.nn as nn

class PatchTST(nn.Module):
    """
    Strong modern baseline: PatchTST.
    Important modern time-series Transformer baseline.
    """
    def __init__(self, seq_len, pred_len, channels, patch_len=16, stride=8, d_model=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        
        self.projection = nn.Linear(patch_len, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.head = nn.Linear(self.patch_num * d_model, pred_len)
        
    def forward(self, x, prompt=None):
        B, L, C = x.shape
        x = x.permute(0, 2, 1) 
        
        patches = x.unfold(2, self.patch_len, self.stride)
        patches = patches.reshape(B * C, self.patch_num, self.patch_len)
        
        enc_in = self.projection(patches)
        enc_out = self.transformer(enc_in)
        
        enc_out = enc_out.reshape(B * C, -1)
        dec_out = self.head(enc_out)
        
        dec_out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return dec_out
