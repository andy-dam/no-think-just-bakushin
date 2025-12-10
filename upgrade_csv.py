import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = 'training_data.csv'
OUTPUT_FILE = 'training_data_v2.csv' # We save to a new file to be safe

# The specific bonuses for the run you already logged
# 10% Speed = 1.10
# 20% Guts  = 1.20
BONUSES = {
    'bonus_spd': 1.10,
    'bonus_sta': 1.00,
    'bonus_pow': 1.00,
    'bonus_gut': 1.20,
    'bonus_wis': 1.00
}

def upgrade_csv():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # 1. Add the new columns with the fixed values
    print("Adding bonus columns...")
    for col, value in BONUSES.items():
        df[col] = value

    # 2. Define the exact column order expected by dataset.py
    target_order = [
        'mood', 
        'facility_level', 
        'training_type', 
        'count_support', 
        'count_rainbow', 
        'count_unity_growing', 
        'count_unity_maxed', 
        'count_bursts_ready', 
        # The 5 New Columns
        'bonus_spd', 
        'bonus_sta', 
        'bonus_pow', 
        'bonus_gut', 
        'bonus_wis', 
        # The Targets
        'gain_spd', 
        'gain_sta', 
        'gain_pow', 
        'gain_gut', 
        'gain_wis', 
        'gain_skill', 
        'gain_energy'
    ]

    # 3. Reorder the dataframe
    try:
        df_v2 = df[target_order]
    except KeyError as e:
        print(f"Error: Your CSV is missing a required column. Details: {e}")
        return

    # 4. Save
    df_v2.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Upgraded file saved as: {OUTPUT_FILE}")
    print("You can now rename this to 'training_data.csv' and use it with the V2 logger.")

if __name__ == "__main__":
    upgrade_csv()