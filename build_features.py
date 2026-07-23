import pandas as pd
import numpy as np

# Pick up where the target-building left off
df = pd.read_csv('f1_2024_with_target.csv')

# ==========================================================
# FEATURE 1: Tire degradation
# ==========================================================

# LapTime comes as text like "0 days 00:01:37.284000". Can't do math on text,
# so turn it into plain seconds (97.284).
df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

# In-lap  = the lap the driver dives INTO the pits
# Out-lap = the lap they come back OUT
# Both are ~20s slower than normal because the car crawls through pit lane. That's
# got nothing to do with tire wear, so we can't let them pollute our pace numbers.
df['IsInLap']  = df['PitInTime'].notna()
df['IsOutLap'] = df['PitOutTime'].notna()

# A "clean" lap time with those two blanked out. Only used to judge true tire pace.
df['CleanLapTime'] = df['LapTimeSeconds']
df.loc[df['IsInLap'] | df['IsOutLap'], 'CleanLapTime'] = np.nan

# Group by Stint too, because tires RESET at every pit stop. Degradation on a
# fresh set should start from scratch, not inherit the worn-out set's numbers.
df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'Stint', 'LapNumber']).reset_index(drop=True)
stint_groups = df.groupby(['Year', 'RoundNumber', 'Driver', 'Stint'])

# expanding().min() tracks the fastest clean lap SO FAR in the stint. It only ever
# looks backward, never at future laps, so no leakage. Skips the NaN dirty laps too.
df['StintBestSoFar'] = stint_groups['CleanLapTime'].transform(lambda s: s.expanding().min())

# The feature: how many seconds off my own best pace am I right now?
# Fresh tires -> near 0. Worn tires -> grows. That climb is the tire falling off.
df['TireDegDelta'] = (df['LapTimeSeconds'] - df['StintBestSoFar']).fillna(0)

# ==========================================================
# FEATURE 2: Safety car / caution flags
# ==========================================================

# Re-sort without Stint — cautions are a track-wide thing, they don't care about
# which set of tires you're on.
df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'LapNumber']).reset_index(drop=True)

# FastF1 crams every status code that happened during a lap into one string, so a
# lap can read "124" = went clear -> yellow -> safety car. We just check which
# digits show up. Codes: 1=clear, 2=yellow, 4=safety car, 5=red flag, 6=VSC.
df['TrackStatus'] = df['TrackStatus'].astype(str)

df['SafetyCarThisLap'] = df['TrackStatus'].str.contains('4').astype(int)
df['VSCThisLap']       = df['TrackStatus'].str.contains('6').astype(int)
df['YellowThisLap']    = df['TrackStatus'].str.contains('2').astype(int)
df['RedFlagThisLap']   = df['TrackStatus'].str.contains('5').astype(int)

# Safety car and VSC both mean "everyone slows down, pitting is cheap right now"
df['AnyCautionThisLap'] = ((df['SafetyCarThisLap'] == 1) | (df['VSCThisLap'] == 1)).astype(int)

# Was there a caution on the PREVIOUS lap? shift(1) looks BACKWARD into the past,
# which is always safe — that's information we genuinely had at the time.
driver_race_groups = df.groupby(['Year', 'RoundNumber', 'Driver'])
df['CautionPrevLap'] = driver_race_groups['AnyCautionThisLap'].shift(1).fillna(0).astype(int)

# The good one: a caution that JUST appeared (on now, off last lap). Everyone
# dives in the moment it comes out, then the rush dies down. Splitting "brand new"
# from "already running" captures that, which one lumped-together flag would miss.
df['CautionJustStarted'] = ((df['AnyCautionThisLap'] == 1) & (df['CautionPrevLap'] == 0)).astype(int)

# ==========================================================
# FEATURE 3: Wet conditions (free weather proxy)
# ==========================================================

# If a driver is on INTERMEDIATE or WET tires, it's raining. We don't need the
# full weather telemetry to know that — the tire choice already tells us.
df['WetConditions'] = df['Compound'].isin(['INTERMEDIATE', 'WET']).astype(int)

# ==========================================================
# Sanity checks — do these features actually separate the classes?
# ==========================================================

print("Tire degradation:")
print("  normal laps:        {:.3f}s".format(df[df.PitNextLap == 0]['TireDegDelta'].mean()))
print("  lap before a pit:   {:.3f}s".format(df[df.PitNextLap == 1]['TireDegDelta'].mean()))

print("\nPit rate on the NEXT lap, by situation:")
situations = [
    ('Green flag',                df[df.AnyCautionThisLap == 0]),
    ('Any caution active',        df[df.AnyCautionThisLap == 1]),
    ('Caution JUST started',      df[df.CautionJustStarted == 1]),
    ('Caution already ongoing',   df[(df.AnyCautionThisLap == 1) & (df.CautionJustStarted == 0)]),
    ('Wet tires',                 df[df.WetConditions == 1]),
    ('Dry tires',                 df[df.WetConditions == 0]),
]
for name, subset in situations:
    print("  {:26s} n={:6d}   {:5.2f}%".format(name, len(subset), 100 * subset['PitNextLap'].mean()))

df.to_csv('f1_2024_features.csv', index=False)
print("\nSaved to f1_2024_features.csv")