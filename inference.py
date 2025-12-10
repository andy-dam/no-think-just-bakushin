import torch
import numpy as np
from model import StatPredictor

# --- CONFIGURATION ---
MODEL_PATH = "aoharu_model.pth"
FAILURE_PENALTY = 50.0 

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
    print("WARNING: Model file not found.")

def calculate_failure_chance(action_idx, current_energy, predicted_cost):
    # (Same failure logic as before...)
    if action_idx == 5: return 0.0
    if action_idx == 4: return 0.05 if current_energy < 5 else 0.0

    projected_energy = current_energy + predicted_cost
    if projected_energy < 0:
        return 0.60 + (abs(projected_energy) * 0.02)
    elif current_energy < 30:
        return 0.15 + ((30 - current_energy) * 0.01)
    else:
        return 0.0

def decide_best_move(global_mood, facility_states, current_energy):
    """
    Args:
        global_mood (int): 0-4
        facility_states (list of lists): A list of 5 feature vectors.
            Each vector must contain: [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
            Index 0 = Speed info, Index 1 = Stamina info, etc.
        current_energy (float): 0-100
    """
    options = ["Speed", "Stamina", "Power", "Guts", "Wisdom", "Rest"]
    best_option = None
    best_score = -9999
    
    print(f"\n--- ANALYSIS (Energy: {current_energy}) ---")

    # Loop through Speed(0) to Rest(5)
    for action_idx in range(6):
        
        # --- 1. CONSTRUCT INPUT VECTOR ---
        # We must grab the specific features for THIS button
        
        if action_idx == 5: 
            # REST CASE: It has no facility features.
            # We create a dummy vector of zeros.
            # Mood is normalized (mood / 4.0)
            feats = [global_mood / 4.0, 0, 0, 0, 0, 0, 0]
        else:
            # TRAINING CASE: Grab the specific features for this button
            # facility_states[0] is Speed features, [1] is Stamina, etc.
            raw_feats = facility_states[action_idx] 
            
            # Normalize Mood (Global) and Facility Level (Index 0 of raw_feats)
            norm_mood = global_mood / 4.0
            norm_fac = raw_feats[0] / 5.0 # Facility Level is first item
            
            # Reconstruct the 7-item numerical vector:
            # [Mood, Fac, Supp, Rain, UnityG, UnityM, Burst]
            feats = [norm_mood, norm_fac, *raw_feats[1:]]

        # Convert to numpy/tensor
        numerical_feats = np.array(feats, dtype=float)
        
        # One-Hot Encoding
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

        fail_prob = calculate_failure_chance(action_idx, current_energy, g_eng)
        win_prob = 1.0 - fail_prob
        final_score = (success_score * win_prob) - (FAILURE_PENALTY * fail_prob)

        risk_str = f"{int(fail_prob*100)}%" if fail_prob > 0 else "Safe"
        print(f"{options[action_idx]:<7} | Score: {final_score:>5.1f} | Risk:{risk_str:<4} | Pred: Spd+{int(g_spd)} Pow+{int(g_pow)}")
        
        if final_score > best_score:
            best_score = final_score
            best_option = options[action_idx]
            
    print(f"----------------\n>>> RECOMMENDATION: {best_option}")
    return best_option

# --- TEST RUN ---
if __name__ == "__main__":
    # Example: 
    # Speed (0) is BAD: Lv1, No supports
    # Power (2) is GODLY: Lv5, 3 Supports, 2 Rainbows, 1 Burst
    
    # Format per button: [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
    speed_feats = [1, 0, 0, 0, 0, 0]
    stamina_feats = [2, 1, 0, 1, 0, 0]
    power_feats = [5, 3, 2, 0, 0, 1]  # The winner
    guts_feats = [1, 0, 0, 0, 0, 0]
    wisdom_feats = [3, 1, 0, 0, 0, 0]
    
    all_states = [speed_feats, stamina_feats, power_feats, guts_feats, wisdom_feats]
    
    global_mood = 4 # Very Good
    current_energy = 80
    
    decide_best_move(global_mood, all_states, current_energy)