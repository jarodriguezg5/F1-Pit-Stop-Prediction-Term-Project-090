import pandas as pd

# Load the full 2024 season we downloaded earlier
df = pd.read_csv('f1_2024_all_races.csv')

#flag every lap where the driver actually pitted
# When a driver dives into the pit lane, FastF1 fills in "PitInTime" on that lap.
# So "is PitInTime filled?" = "did they pit on this lap?" (1 = yes, 0 = no)
df['PittedThisLap'] = df['PitInTime'].notna().astype(int)

#put the laps in proper order
# We sort by year -> race -> driver -> lap so that each driver's race
# reads top-to-bottom in the right sequence. This matters a LOT for the next step.
df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'LapNumber']).reset_index(drop=True)

# build the target -> "will this driver pit on the NEXT lap?"
# shift(-1) means "grab the value from the row directly below." Within each
# driver's race, that next row is their next lap. So if the next lap has
# PittedThisLap = 1, then THIS lap gets PitNextLap = 1.
# We group by race + driver so we never accidentally peek at a different
# driver's lap or bleed from one race into another.
df['PitNextLap'] = df.groupby(['Year', 'RoundNumber', 'Driver'])['PittedThisLap'].shift(-1)

# handle the final lap of each race
# The last lap a driver runs has no "next lap," so shift(-1) leaves it blank (NaN).
# We can't label those, so we drop them. It's a tiny slice (~476 rows).
df = df.dropna(subset=['PitNextLap'])
df['PitNextLap'] = df['PitNextLap'].astype(int)


print(f"Total labeled laps: {len(df)}")
print("\nTarget balance (0 = no pit next lap, 1 = pits next lap):")
print(df['PitNextLap'].value_counts())
print(f"\nPositive rate: {100 * df['PitNextLap'].mean():.2f}%")

# Verify against a known example: VER pitted on laps 17 and 37 in Bahrain
print("\n--- VER Bahrain check (should see PitNextLap=1 on laps 16 and 36) ---")
ver = df[(df['Driver'] == 'VER') & (df['EventName'] == 'Bahrain Grand Prix')]
print(ver[['LapNumber', 'Compound', 'TyreLife', 'PittedThisLap', 'PitNextLap']].head(38).to_string(index=False))

# Save so the next step (feature engineering) can pick up right where we left off
df.to_csv('f1_2024_with_target.csv', index=False)
print("\nSaved to f1_2024_with_target.csv")