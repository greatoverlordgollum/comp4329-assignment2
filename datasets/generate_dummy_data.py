import pandas as pd
import numpy as np
import os

def generate_datasets():
    """Generates dummy datasets to simulate the required project datasets"""
    datasets = ['ETTh1', 'ETTh2', 'Weather', 'ECL']
    
    # ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    for name in datasets:
        # Create 1000 rows, 4 numerical columns + date roughly 
        dates = pd.date_range("2020-01-01", periods=1000, freq="H")
        data = np.random.randn(1000, 4)
        
        # Add some mock trends and periodicities for the C+ prompt to detect
        data[:, 0] += np.linspace(0, 10, 1000) # Trend
        data[:, 1] += np.sin(np.linspace(0, 50, 1000)) * 5 # Seasonality/Periodicity
        
        df = pd.DataFrame(data, columns=['OT', 'M1', 'M2', 'M3'])
        df.insert(0, 'date', dates)
        
        target_path = os.path.join(os.path.dirname(__file__), f"{name}.csv")
        df.to_csv(target_path, index=False)
        print(f"Generated {target_path}")

if __name__ == "__main__":
    generate_datasets()
