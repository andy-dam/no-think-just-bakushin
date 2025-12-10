import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AoharuDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {csv_file}. Run data_logger.py first.")

        # --- CONFIGURATION ---
        self.categorical_col = 'training_type' # 0-5
        
        # 1. Base State Features (7)
        self.base_numerical_cols = [
            'mood',                # 0-4
            'facility_level',      # 1-5
            'count_support',       # 0-5
            'count_rainbow',       # 0-5
            'count_unity_growing', # 0-X 
            'count_unity_maxed',   # 0-X 
            'count_bursts_ready'   # 0-5
        ]
        
        # 2. Growth Rate Features (5) - NEW
        # e.g., 1.0, 1.20, 1.10...
        self.bonus_cols = [
            'bonus_spd', 'bonus_sta', 'bonus_pow', 'bonus_gut', 'bonus_wis'
        ]
        
        # 3. Targets (7)
        self.target_cols = [
            'gain_spd', 'gain_sta', 'gain_pow', 
            'gain_gut', 'gain_wis', 'gain_skill', 'gain_energy'
        ]

        # --- PRE-NORMALIZATION ---
        self.processed_data = self.data.copy()
        self.processed_data['mood'] = self.processed_data['mood'] / 4.0
        self.processed_data['facility_level'] = self.processed_data['facility_level'] / 5.0
        # Bonuses are usually 1.0 to 1.3, so they don't need heavy normalization.

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.processed_data.iloc[idx]
        t_type = int(row[self.categorical_col])
        
        # A. Base Features
        base_vals = row[self.base_numerical_cols].values.astype(float)
        
        # B. Bonus Features
        bonus_vals = row[self.bonus_cols].values.astype(float)

        # SPECIAL CASE: Rest (Type 5)
        # Zero out facility stats. Keep Mood (index 0).
        # Bonuses also don't apply to Rest (Energy gain is fixed), so we can leave them or zero them.
        # Leaving them allows the model to learn they are irrelevant.
        if t_type == 5:
            base_vals[1:] = 0.0 

        # Combine Numerical: [7 base + 5 bonuses] = 12 floats
        x_numerical = torch.tensor(np.concatenate((base_vals, bonus_vals)), dtype=torch.float32)

        # C. One-Hot Encoding (6)
        x_categorical = torch.zeros(6)
        x_categorical[t_type] = 1.0
        
        # D. Total Input: 12 + 6 = 18 Features
        x_input = torch.cat((x_numerical, x_categorical))

        # E. Targets
        y_target = torch.tensor(row[self.target_cols].values, dtype=torch.float32)

        return x_input, y_target