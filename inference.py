import torch
import numpy as np
from model import StatPredictor

# --- CONFIGURATION ---
MODEL_PATH = "aoharu_model.pth"
FAILURE_PENALTY = 50.0  # Points subtracted from Expected Value if a failure occurs

# Strategy Weights
# Adjust these to change your bot's personality!
WEIGHTS = {
    'spd': 1.0, 
    'sta': 1.0, 
    'pow': 1.0, 
    'gut': 0.8, 
    'wis': 1.0, 
    'skill': 0.5, 
    'energy': 2.0 
}

# Initialize Model (Input Size = 18 for V2 with Growth Rates)
model = StatPredictor(input_size=18)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
except FileNotFoundError:
    print("WARNING: Model file not found. Predictions will be random noise.")

def decide_best_move(global_mood, facility_states, current_energy, failure_rates, growth_rates):
    """
    Args:
        global_mood (int): 0-4
        facility_states (list of lists): 5 vectors [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
        current_energy (float): 0-100
        failure_rates (list): 5 floats (0.0 - 1.0)
        growth_rates (list): 5 floats representing multipliers (e.g. [1.1, 1.0, 1.0, 1.2, 1.0])
                             Order: [Spd, Sta, Pow, Gut, Wis]
    """
    options = ["Speed", "Stamina", "Power", "Guts", "Wisdom", "Rest"]
    best_option = None
    best_score = -9999
    
    print(f"\n--- ANALYSIS (Energy: {current_energy}) ---")
    # Header for the table
    print(f"{'Option':<7} | {'Score':<5} | {'Risk':<4} | {'Predictions (Raw)':<50}")
    print("-" * 85)

    for action_idx in range(6):
        
        # --- 1. PREPARE FEATURES ---
        if action_idx == 5: 
            # REST CASE: Zero out facility features. Keep Mood (Index 0).
            # [Mood, Fac, Supp, Rain, UnityG, UnityM, Burst]
            base_feats = [global_mood / 4.0, 0, 0, 0, 0, 0, 0]
            fail_prob = 0.0
        else:
            # TRAINING CASE: Grab specific features
            raw = facility_states[action_idx] 
            norm_mood = global_mood / 4.0
            norm_fac = raw[0] / 5.0 
            
            # Reconstruct Base Vector (7 items)
            base_feats = [norm_mood, norm_fac, *raw[1:]]
            fail_prob = failure_rates[action_idx]

        # --- 2. APPEND GROWTH RATES ---
        # Numerical Vector: [7 Base] + [5 Bonuses] = 12 items
        # Note: We include growth rates even for Rest, so the model inputs align.
        full_numerical = base_feats + growth_rates
        
        # --- 3. CREATE TENSOR ---
        num_tensor = np.array(full_numerical, dtype=float)
        cat_tensor = np.zeros(6) # One-Hot for Action Type
        cat_tensor[action_idx] = 1.0
        
        full_input = np.concatenate((num_tensor, cat_tensor)) # Total 18 inputs
        tensor_input = torch.tensor(full_input, dtype=torch.float32).unsqueeze(0)
        
        # --- 4. PREDICT ---
        with torch.no_grad():
            pred = model(tensor_input)[0]
        
        g_spd, g_sta, g_pow, g_gut, g_wis, g_skill, g_eng = pred
        
        # --- 5. SCORING ---
        # Dynamic Energy Logic
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

        # Expected Value (Risk Assessment)
        win_prob = 1.0 - fail_prob
        final_score = (success_score * win_prob) - (FAILURE_PENALTY * fail_prob)

        # --- 6. PRINT RESULTS ---
        risk_str = f"{int(fail_prob*100)}%"
        
        # Format stats aligned
        stats_str = (
            f"Spd:{int(g_spd):>3} "
            f"Sta:{int(g_sta):>3} "
            f"Pow:{int(g_pow):>3} "
            f"Gut:{int(g_gut):>3} "
            f"Wis:{int(g_wis):>3} "
            f"Skl:{int(g_skill):>2} "
            f"Eng:{int(g_eng):>3}"
        )

        print(f"{options[action_idx]:<7} | {final_score:>5.1f} | {risk_str:<4} | {stats_str}")
        
        if final_score > best_score:
            best_score = final_score
            best_option = options[action_idx]
            
    print("-" * 85)
    print(f">>> RECOMMENDATION: {best_option}")
    return best_option

# --- MAIN DRIVER: SCENARIO TESTING ---
if __name__ == "__main__":
    print("=== STARTING SCENARIO TESTS ===")
    
    # --- CONSTANTS FOR STATES ---
    # [FacLvl, Supp, Rain, UnityG, UnityM, Burst]
    GODLY_STATE   = [5, 4, 2, 0, 0, 1] # Massive Rainbow + Burst
    GOOD_STATE    = [3, 2, 1, 0, 0, 0] # Solid Training
    AVERAGE_STATE = [2, 1, 0, 0, 0, 0] # Meh
    GARBAGE_STATE = [1, 0, 0, 0, 0, 0] # Empty

    # Standard "Flat" Growth Rates (No bias)
    FLAT_GROWTH = [1.0, 1.0, 1.0, 1.0, 1.0]

    # ------------------------------------------------------------------
    # TEST 1: THE OBVIOUS WINNER
    # Situation: Energy is full. Speed is Godly.
    # Expectation: SPEED should win with a huge score.
    # ------------------------------------------------------------------
    print("\n>>> TEST 1: THE SPEED DEMON (High Energy, Godly Speed)")
    states_1 = [
        GODLY_STATE,   # Speed (WINNER)
        GARBAGE_STATE, # Stamina
        GARBAGE_STATE, # Power
        GARBAGE_STATE, # Guts
        GARBAGE_STATE  # Wisdom
    ]
    rates_1 = [0.0, 0.0, 0.0, 0.0, 0.0] # 0% fail at max energy
    
    decide_best_move(4, states_1, 100, rates_1, FLAT_GROWTH)


    # ------------------------------------------------------------------
    # TEST 2: THE SAFETY PLAY (WISDOM)
    # Situation: Energy is low (25). Speed is Good but risky (30% Fail).
    # Expectation: WISDOM should win because it's safe (0% Risk) and gives stats.
    # ------------------------------------------------------------------
    print("\n>>> TEST 2: THE SAFETY PLAY (Low Energy)")
    states_2 = [
        GOOD_STATE,    # Speed is tempting...
        AVERAGE_STATE, 
        GARBAGE_STATE,
        GARBAGE_STATE,
        GOOD_STATE     # Wisdom (WINNER due to safety)
    ]
    # Physical trainings have high failure rates due to low energy
    rates_2 = [0.30, 0.30, 0.30, 0.30, 0.0] 
    
    decide_best_move(2, states_2, 25, rates_2, FLAT_GROWTH)


    # ------------------------------------------------------------------
    # TEST 3: THE EMERGENCY REST
    # Situation: Energy is critical (10). Fail rates are deadly (60%+).
    # Expectation: REST should win. Training is mathematically suicide.
    # ------------------------------------------------------------------
    print("\n>>> TEST 3: EMERGENCY REST (Critical Energy)")
    states_3 = [
        GODLY_STATE,   # Even a Godly training isn't worth a 60% fail chance
        GARBAGE_STATE,
        GARBAGE_STATE,
        GARBAGE_STATE,
        GARBAGE_STATE  # Even Wisdom is weak here
    ]
    rates_3 = [0.60, 0.60, 0.60, 0.60, 0.05] 
    
    decide_best_move(1, states_3, 10, rates_3, FLAT_GROWTH)


    # ------------------------------------------------------------------
    # TEST 4: THE GROWTH RATE TIE-BREAKER
    # Situation: Speed and Power facilities are IDENTICAL (same level/support).
    # But: Uma has +20% Power Growth (1.20) vs +0% Speed Growth (1.0).
    # Expectation: POWER should win because the predicted gain is higher.
    # ------------------------------------------------------------------
    print("\n>>> TEST 4: GROWTH RATE BIAS (+20% Power)")
    
    # Identical "Good" states
    states_4 = [
        GOOD_STATE,    # Speed
        GARBAGE_STATE,
        GOOD_STATE,    # Power (WINNER due to bonus)
        GARBAGE_STATE,
        GARBAGE_STATE
    ]
    rates_4 = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Bias: [Spd=1.0, Sta=1.0, Pow=1.2, Gut=1.0, Wis=1.0]
    power_bias_growth = [1.0, 1.0, 1.2, 1.0, 1.0]
    
    decide_best_move(4, states_4, 90, rates_4, power_bias_growth)


    # ------------------------------------------------------------------
    # TEST 5: THE "OVERHEAL" TRAP
    # Situation: Energy is 100.
    # Choice: A weak Speed training vs. Rest.
    # Expectation: SPEED must win. Rest gives +0 effective energy (capped at 100),
    # so even a weak training is better than wasting a turn resting.
    # ------------------------------------------------------------------
    print("\n>>> TEST 5: THE OVERHEAL TRAP (Don't Rest at 100 Energy)")
    states_5 = [
        AVERAGE_STATE, # Speed (Weak, but > 0) -> WINNER
        GARBAGE_STATE,
        GARBAGE_STATE,
        GARBAGE_STATE,
        GARBAGE_STATE
    ]
    # Rest will predict high raw energy, but logic should cap it at 0 value
    
    decide_best_move(4, states_5, 100, rates_4, FLAT_GROWTH)