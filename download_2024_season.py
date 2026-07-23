import fastf1
import pandas as pd
from pathlib import Path
import time

# Create cache folder if it doesn't exist
cache_dir = 'C:/F1_data_cache'
Path(cache_dir).mkdir(parents=True, exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache(cache_dir)

# Get the full 2024 race schedule
schedule = fastf1.get_event_schedule(2024)
print(f"Total events in 2024: {len(schedule)}")
print(schedule[['RoundNumber', 'EventName', 'Country']])

# Empty list to collect all races' lap data
all_laps = []

# Loop through each race and download lap data
for idx, event in schedule.iterrows():
    round_num = event['RoundNumber']
    event_name = event['EventName']
    
    # Skip testing events (round 0)
    if round_num == 0:
        continue
    
    try:
        print(f"\nDownloading Round {round_num}: {event_name}...")
        session = fastf1.get_session(2024, round_num, 'R')  # 'R' = Race
        session.load()
        
        laps = session.laps.copy()
        laps['Year'] = 2024
        laps['RoundNumber'] = round_num
        laps['EventName'] = event_name
        
        all_laps.append(laps)
        print(f"  -> Got {len(laps)} laps")
        
        time.sleep(1)  # Be polite to the server, avoid overwhelming it
        
    except Exception as e:
        print(f"  -> FAILED: {e}")
        continue

# Combine all races into one big dataframe
combined_laps = pd.concat(all_laps, ignore_index=True)
print(f"\n\nTOTAL LAPS COLLECTED: {len(combined_laps)}")
print(f"Races successfully downloaded: {combined_laps['EventName'].nunique()}")

# Save to CSV for now (just to inspect)
combined_laps.to_csv('f1_2024_all_races.csv', index=False)
print("Saved to f1_2024_all_races.csv")