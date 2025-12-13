import csv
import os

CSV_FILE = 'training_data.csv'

# Updated Headers
HEADERS = [
    'mood', 'facility_level', 'training_type', 
    'count_support', 'count_rainbow', 'count_unity_growing', 'count_unity_maxed', 'count_bursts_ready',
    # New Columns
    'bonus_spd', 'bonus_sta', 'bonus_pow', 'bonus_gut', 'bonus_wis',
    # Targets
    'gain_spd', 'gain_sta', 'gain_pow', 'gain_gut', 'gain_wis', 'gain_skill', 'gain_energy'
]

def get_input(prompt, default=None):
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ")
        return user_input if user_input.strip() else default
    else:
        return input(f"{prompt}: ")

def main():
    file_exists = os.path.isfile(CSV_FILE)
    
    # --- SETUP PHASE ---
    print("\n--- AOHARU DATA LOGGER V2 ---")
    if not file_exists:
        print("Creating NEW dataset.")
    else:
        print("Appending to EXISTING dataset.")
        
    # Ask for Growth Rates (Once per session/run)
    print("\n[SETUP] Enter Growth Rates for this Uma (e.g., '20 0 10 0 0' for 20% Spd, 10% Pow)")
    raw_rates = input("Rates %: ").strip().split()
    
    if len(raw_rates) != 5:
        print("Error: Must enter 5 numbers. Exiting.")
        return
        
    # Convert "20" -> 1.20
    bonuses = [1.0 + (float(x) / 100.0) for x in raw_rates]
    print(f"Stored Multipliers: Spd={bonuses[0]:.2f}, Sta={bonuses[1]:.2f}, Pow={bonuses[2]:.2f}, Gut={bonuses[3]:.2f}, Wis={bonuses[4]:.2f}")

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)

        last_mood = 0
        last_fac_lvl = 1

        while True:
            try:
                print("\n--- NEW TURN ---")
                
                # 1. Global
                print("  Mood: 0=Awful, 1=Bad, 2=Normal, 3=Good, 4=Great")
                i_mood = get_input("Mood (0-4)", last_mood)
                if i_mood == 'q': break
                last_mood = i_mood

                # 2. Action
                print("  Type: 0=Speed, 1=Stamina, 2=Power, 3=Guts, 4=Wit, 5=Rest")
                i_type = get_input("Type (0-5)", "0")
                if i_type == 'q': break
                
                # 3. Features
                if int(i_type) == 5: # Rest
                    print(">> Rest selected (auto-zero facility).")
                    feats = [0, 0, 0, 0, 0] # Fac, Supp, Rain, Grow, Max, Burst (Fac handled separately)
                    # For simplicty in CSV, we save 0s
                    i_fac, i_supp, i_rain, i_grow, i_maxed, i_burst = 0, 0, 0, 0, 0, 0
                else:
                    i_fac = get_input("FacLvl (1-5)", last_fac_lvl)
                    if i_fac == 'q': break
                    last_fac_lvl = i_fac

                    line = get_input("Supp Rain Grow Max Burst (e.g. '3 1 1 0 1')")
                    if line == 'q': break
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    i_supp, i_rain, i_grow, i_maxed, i_burst = parts

                # 4. Gains
                g_line = get_input("Gains: Spd Sta Pow Gut Wis Skill Eng")
                if g_line == 'q': break
                gains = g_line.strip().split()
                if len(gains) != 7: continue

                # 5. Write
                row = [
                    last_mood,
                    i_fac,
                    i_type,
                    i_supp, i_rain, i_grow, i_maxed, i_burst,
                    *bonuses, # Add the static bonuses
                    *gains
                ]
                
                writer.writerow(row)
                f.flush()
                print(">> Saved.")

            except ValueError:
                print("Invalid number.")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()