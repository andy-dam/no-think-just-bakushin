import csv
import os

CSV_FILE = 'training_data.csv'

# Define the exact column order matching dataset.py
HEADERS = [
    'mood', 
    'facility_level', 
    'training_type', 
    'count_support', 
    'count_rainbow', 
    'count_unity_growing', 
    'count_unity_maxed', 
    'count_bursts_ready', 
    'gain_spd', 'gain_sta', 'gain_pow', 'gain_gut', 'gain_wis', 'gain_skill', 'gain_energy'
]

def get_input(prompt, default=None):
    """Helper to get input with a default value."""
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ")
        return user_input if user_input.strip() else default
    else:
        return input(f"{prompt}: ")

def main():
    # 1. Initialize CSV if it doesn't exist
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADERS)
            print(f"Created new file: {CSV_FILE}")
        else:
            print(f"Appending to existing: {CSV_FILE}")

        print("\n--- AOHARU DATA LOGGER ---")
        print("Tip: Press ENTER to reuse the value in [brackets].")
        print("Type 'q' at any prompt to quit.")

        # Default values (persist across loops)
        last_mood = 0
        last_fac_lvl = 1

        while True:
            try:
                print("\n--- NEW ENTRY ---")
                
                # --- 1. Global Context ---
                # Mood is sticky (doesn't change often)
                i_mood = get_input("Mood (0=Bad ... 4=VeryGood)", last_mood)
                if i_mood == 'q': break
                last_mood = i_mood

                # --- 2. The Action ---
                print("Options: 0=Spd, 1=Sta, 2=Pow, 3=Gut, 4=Wis, 5=Rest")
                i_type = get_input("Training Type", "0")
                if i_type == 'q': break
                
                # --- 3. Context Specifics ---
                # If REST (5), we skip the facility questions and auto-zero them
                if int(i_type) == 5:
                    print(">> Selected REST. Skipping facility inputs (auto-zero).")
                    i_fac = 0
                    i_supp = 0
                    i_rain = 0
                    i_grow = 0
                    i_maxed = 0
                    i_burst = 0
                else:
                    # Facility Level is sticky
                    i_fac = get_input("Facility Level (1-5)", last_fac_lvl)
                    if i_fac == 'q': break
                    last_fac_lvl = i_fac

                    # These change every turn, so no defaults usually
                    # Short inputs for speed
                    line = get_input("Enter: [Supports] [Rainbows] [UnityGrow] [UnityMax] [Bursts]\n(e.g., '3 1 2 0 1')")
                    if line == 'q': break
                    
                    # Parse the space-separated line
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print("Error: Please enter exactly 5 numbers separated by spaces.")
                        continue
                        
                    i_supp, i_rain, i_grow, i_maxed, i_burst = parts

                # --- 4. The Targets (Gains) ---
                # This matches what you see in the preview bubble
                print("\nEnter Gains shown in bubble:")
                g_line = get_input("Spd Sta Pow Gut Wis Skill Eng\n(e.g., '20 0 15 0 0 4 -20')")
                if g_line == 'q': break
                
                gains = g_line.strip().split()
                if len(gains) != 7:
                    print("Error: You must enter 7 numbers for the gains.")
                    continue

                # --- 5. Write to CSV ---
                row = [
                    last_mood,
                    i_fac,
                    i_type,
                    i_supp,
                    i_rain,
                    i_grow,
                    i_maxed,
                    i_burst,
                    *gains # Unpack the gains list
                ]
                
                writer.writerow(row)
                f.flush() # Ensure data is saved immediately
                print(">> Saved row!")

            except ValueError:
                print("Invalid input. Please enter numbers.")
            except KeyboardInterrupt:
                print("\nQuitting...")
                break

if __name__ == "__main__":
    main()