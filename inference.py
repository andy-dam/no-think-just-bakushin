import torch
import numpy as np
from model import StatPredictor

# --- CONFIGURATION ---
MODEL_PATH = "aoharu_model.pth"
FAILURE_PENALTY = 50.0  # Points subtracted from Expected Value if a failure occurs

# Strategy Weights
WEIGHTS = {
    'spd': 1.0, 'sta': 1.0, 'pow': 1.0, 'gut': 0.8, 'wis': 1.0, 
    'skill': 0.5, 'energy': 2.0 
}

model = StatPredictor()
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
except FileNotFoundError:
    print("WARNING: Model file not found. Please run train.py first.")

def decide_best_move(global_mood, facility_states, current_energy, failure_rates):
    """
    Args:
        global_mood (int): 0-4
        facility_states (list of lists): A list of 5 feature vectors.
            Each vector: [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
            UnityG: Unity Growing
            UnityM: Unity Maxed
            Burst: Burst Ready
            Index 0=Speed, 1=Stamina, 2=Power, 3=Guts, 4=Wisdom.
        current_energy (float): 0-100
        failure_rates (list of floats): The 5 failure rates shown on screen (0.0 to 1.0).
            e.g., [0.0, 0.15, 0.0, 0.80, 0.0] for [Spd, Sta, Pow, Gut, Wis]
    """
    options = ["Speed", "Stamina", "Power", "Guts", "Wisdom", "Rest"]
    best_option = None
    best_score = -9999
    
    print(f"\n--- ANALYSIS (Energy: {current_energy}) ---")

    # Loop through Speed(0) to Rest(5)
    for action_idx in range(6):
        
        # --- 1. CONSTRUCT INPUT VECTOR ---
        if action_idx == 5: 
            # REST CASE: Dummy vector
            feats = [global_mood / 4.0, 0, 0, 0, 0, 0, 0]
            fail_prob = 0.0 # Rest is always safe
        else:
            # TRAINING CASE: Grab specific features
            raw_feats = facility_states[action_idx] 
            
            # Normalize Mood and Facility
            norm_mood = global_mood / 4.0
            norm_fac = raw_feats[0] / 5.0 
            
            feats = [norm_mood, norm_fac, *raw_feats[1:]]
            
            # Get the exact failure rate from user input
            fail_prob = failure_rates[action_idx]

        # Convert to tensor
        numerical_feats = np.array(feats, dtype=float)
        categorical_feats = np.zeros(6)
        categorical_feats[action_idx] = 1.0
        
        full_input = np.concatenate((numerical_feats, categorical_feats))
        tensor_input = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)
        
        # --- 2. PREDICT ---
        with torch.no_grad():
            pred = model(tensor_input)[0]
        
        g_spd, g_sta, g_pow, g_gut, g_wis, g_skill, g_eng = pred
        
        # --- 3. SCORING (Dynamic Energy Logic) ---
        projected_total = current_energy + g_eng
        effective_energy_change = (100 - current_energy) if projected_total > 100 else g_eng
            
        if current_energy > 70: energy_weight = 0.2 
        elif current_energy > 40: energy_weight = 1.0
        else: energy_weight = 3.0

        success_score = (g_spd * WEIGHTS['spd']) + \
                        (g_sta * WEIGHTS['sta']) + \
                        (g_pow * WEIGHTS['pow']) + \
                        (g_gut * WEIGHTS['gut']) + \
                        (g_wis * WEIGHTS['wis']) + \
                        (g_skill * WEIGHTS['skill']) + \
                        (effective_energy_change * energy_weight)

        # --- 4. EXPECTED VALUE CALCULATION ---
        win_prob = 1.0 - fail_prob
        
        # EV = (Win_Score * Win%) - (Penalty * Fail%)
        final_score = (success_score * win_prob) - (FAILURE_PENALTY * fail_prob)

        # Formatting output
        risk_str = f"{int(fail_prob*100)}%"
        print(f"{options[action_idx]:<7} | Score: {final_score:>5.1f} | Risk:{risk_str:<4} | Pred: Spd+{int(g_spd)} Eng{int(g_eng)}")
        
        if final_score > best_score:
            best_score = final_score
            best_option = options[action_idx]
            
    print(f"----------------\n>>> RECOMMENDATION: {best_option}")
    return best_option

# --- TEST RUN ---
if __name__ == "__main__":
    # Example Scenario:
    # Energy is low (20), causing high failure rates on Speed/Power.
    # Wisdom is safe (0%).
    
    # 5 Training States: [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
    # Speed is tempting (Burst ready) but risky!
    speed_feats =   [3, 1, 0, 0, 0, 1] 
    stamina_feats = [2, 0, 0, 0, 0, 0]
    power_feats =   [4, 2, 1, 0, 0, 0]
    guts_feats =    [1, 0, 0, 0, 0, 0]
    wisdom_feats =  [3, 2, 1, 0, 0, 0] # Good wisdom training
    
    all_states = [speed_feats, stamina_feats, power_feats, guts_feats, wisdom_feats]
    
    # Failure Rates from Screen: [Spd, Sta, Pow, Gut, Wis]
    # Speed has 35% fail rate due to low energy! Wisdom has 0%.
    rates = [0.35, 0.30, 0.30, 0.25, 0.0]
    
    global_mood = 2 # Normal
    current_energy = 20
    
    decide_best_move(global_mood, all_states, current_energy, rates)