import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def extract_cplus_prompt_stats(df, variant="full"):
    """
    Extracts the semantic features outlined in Section 7 (C+ prompt refinement).
    Variants filter out specific ablation features to simulate experiments.
    """
    values = df.iloc[:, -1].dropna().values
    
    if len(values) == 0:
        return torch.zeros(12, dtype=torch.float32)

    # 1. Basic Stats (5)
    mean, median, std = np.mean(values), np.median(values), np.std(values)
    min_val, max_val = np.min(values), np.max(values)
    
    # 2. Trend (2)
    slope = (values[-1] - values[0]) / (len(values) + 1e-5)
    trend_str = np.abs(slope)
    
    # 3. Volatility (2)
    rolling_var = pd.Series(values).rolling(window=10).var().mean()
    if np.isnan(rolling_var): 
        rolling_var = np.var(values)
    global_var = np.var(values)
    
    # 4. Periodicity (1) (Approx with correlation lag 1)
    if len(values) > 1:
        corr_1 = np.corrcoef(values[:-1], values[1:])[0, 1]
    else:
        corr_1 = 0
    if np.isnan(corr_1): 
        corr_1 = 0
        
    # 5. Stability (2)
    half = len(values) // 2
    early_mean = np.mean(values[:half]) if half > 0 else 0
    late_mean = np.mean(values[half:]) if half > 0 else 0
    mean_diff = abs(late_mean - early_mean)
    var_diff = abs(np.var(values[half:]) - np.var(values[:half]) if half > 0 else 0)

    vec = [mean, median, std, min_val, max_val, slope, trend_str, 
           rolling_var, global_var, corr_1, mean_diff, var_diff]
    vec_tensor = torch.tensor(vec, dtype=torch.float32)
    
    # Handle NaNs just in case
    vec_tensor = torch.nan_to_num(vec_tensor, nan=0.0)

    # Apply ablation masks
    if variant == "basic_only":
        vec_tensor[5:] = 0.0
    elif variant == "no_periodicity":
        vec_tensor[9] = 0.0
    elif variant == "no_trend_stability":
        vec_tensor[5:7] = 0.0 # Trend
        vec_tensor[10:12] = 0.0 # Stability
    elif variant == "original":
        # Model C original: basic stats only (or simpler)
        vec_tensor[5:] = 0.0 

    return vec_tensor

def get_dataloader(dataset_name, fraction, seq_len=96, pred_len=96, batch_size=32):
    try:
        df = pd.read_csv(f"./datasets/{dataset_name}.csv").drop(columns=['date'], errors='ignore').ffill().bfill()
    except FileNotFoundError:
        print(f"Dataset ./datasets/{dataset_name}.csv not found.")
        return None, None, None
        
    df = df.iloc[:int(len(df) * fraction)]
    
    # Scale Data (Standard practice for these TS datasets to prevent exploding MSE)
    scaler = StandardScaler()
    data = scaler.fit_transform(df.values)
    
    x, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        x.append(data[i : i+seq_len])
        y.append(data[i+seq_len : i+seq_len+pred_len])
        
    if len(x) == 0:
        return None, None, None
        
    x_t = torch.tensor(np.array(x), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.float32)
    
    ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), df, data.shape[1]
