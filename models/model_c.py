import torch
import torch.nn as nn

class MultimodalTimeVLM(nn.Module):
    """ 
    Model C and Model C+ abstraction.
    Represents Time-LLM / GPT4TS / Time-VLM-lite
    We implement the C vs C+ variance not at an architectural level, but via the prompt passed to it.
    """
    def __init__(self, seq_len, pred_len, channels, num_prompt_features=12, d_model=32):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        
        # In a true VLM, this might be a frozen language model embedding.
        # Here we simulate with an LSTM feature extractor and projector.
        self.ts_encoder = nn.LSTM(channels, d_model, batch_first=True)
        
        self.prompt_projection = nn.Sequential(
            nn.Linear(num_prompt_features, d_model),
            nn.ReLU()
        )
        
        self.forecaster = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * channels)
        )

    def forward(self, x, prompt_features):
        # 1. Time-series extraction
        _, (hn, _) = self.ts_encoder(x)
        ts_emb = hn[-1] # [Batch, d_model]
        
        # 2. textual / semantic prompt embedding mapping
        p_emb = self.prompt_projection(prompt_features) # [Batch, d_model]
        
        # 3. Fuse 
        combined = torch.cat([ts_emb, p_emb], dim=1)
        
        # 4. Forecast
        out = self.forecaster(combined)
        return out.view(-1, self.pred_len, self.channels)
