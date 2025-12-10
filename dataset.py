import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AoharuDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file (e.g., 'training_data.csv').
        """
        # Load the raw data
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {csv_file}. Make sure you've generated the CSV first.")

        # --- CONFIGURATION ---
        self.categorical_col = 'training_type' # 0=Spd, 1=Sta, 2=Pow, 3=Gut, 4=Wis, 5=Rest
        
        self.numerical_cols = [
            'mood',                # 0-4
            'facility_level',      # 1-5
            'count_support',       # 0-5
            'count_rainbow',       # 0-5
            'count_unity_growing', # 0-X 
            'count_unity_maxed',   # 0-X 
            'count_bursts_ready'   # 0-5
        ]
        
        self.target_cols = [
            'gain_spd', 'gain_sta', 'gain_pow', 
            'gain_gut', 'gain_wis', 'gain_skill', 'gain_energy'
        ]

        # --- PRE-NORMALIZATION ---
        # Scale inputs to 0-1 range for better Neural Network performance
        self.processed_data = self.data.copy()
        self.processed_data['mood'] = self.processed_data['mood'] / 4.0
        self.processed_data['facility_level'] = self.processed_data['facility_level'] / 5.0

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.processed_data.iloc[idx]
        
        # 1. Get Action Type (0-5)
        t_type = int(row[self.categorical_col])
        
        # 2. Get Numerical Features
        vals = row[self.numerical_cols].values.astype(float)
        
        # SPECIAL CASE: "Rest" (Type 5)
        # If action is Rest, zero out facility/card stats. 
        # Index 0 is 'mood', which we keep. Index 1+ are facility stats.
        if t_type == 5:
            vals[1:] = 0.0 

        x_numerical = torch.tensor(vals, dtype=torch.float32)

        # 3. One-Hot Encoding
        # [Speed, Sta, Pow, Gut, Wis, Rest]
        x_categorical = torch.zeros(6)
        x_categorical[t_type] = 1.0
        
        # 4. Combine Inputs (7 numerical + 6 categorical = 13 features)
        x_input = torch.cat((x_numerical, x_categorical))

        # 5. Targets
        y_target = torch.tensor(row[self.target_cols].values, dtype=torch.float32)

        return x_input, y_target