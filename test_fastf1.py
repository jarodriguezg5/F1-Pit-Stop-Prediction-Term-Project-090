import fastf1
import pandas as pd
from pathlib import Path

# Create cache folder if it doesn't exist
cache_dir = 'C:/F1_data_cache'
Path(cache_dir).mkdir(parents=True, exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache(cache_dir)

# Grab a single 2024 race to test
session = fastf1.get_session(2024, 'Bahrain', 'R')  # 2024 Bahrain race
session.load()

# Get lap data
laps = session.laps

# Save to CSV so you can open in Excel
laps.to_csv('bahrain_2024_laps.csv')
print(f"Saved {len(laps)} laps to bahrain_2024_laps.csv")

# Also show pit stop rows in terminal
print("\nLaps with pit stop activity:")
pit_laps = laps[laps['PitInTime'].notna() | laps['PitOutTime'].notna()]
print(pit_laps[['Driver', 'LapNumber', 'PitInTime', 'PitOutTime', 'Compound', 'TyreLife']])