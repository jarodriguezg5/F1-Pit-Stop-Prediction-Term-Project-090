import pandas as pd
import numpy as np

# Pick up where the target-building left off
df = pd.read_csv('f1_all_with_target.csv', low_memory=False)

# LapTime comes as text like "0 days 00:01:37.284000". Can't do math on text,
# so turn it into plain seconds (97.284).
df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

# FEATURE 1: Safety car / caution flags

# These come first now, because tire degradation depends on knowing which laps
# were run under caution.
#
# FastF1 crams every status code that happened during a lap into one string, so a
# lap can read "124" = went clear -> yellow -> safety car. We just check which
# digits show up. Codes: 1=clear, 2=yellow, 4=safety car, 5=red flag, 6=VSC.
df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'LapNumber']).reset_index(drop=True)
df['TrackStatus'] = df['TrackStatus'].astype(str)

df['SafetyCarThisLap'] = df['TrackStatus'].str.contains('4').astype(int)
df['VSCThisLap']       = df['TrackStatus'].str.contains('6').astype(int)
df['YellowThisLap']    = df['TrackStatus'].str.contains('2').astype(int)
df['RedFlagThisLap']   = df['TrackStatus'].str.contains('5').astype(int)

# Safety car and VSC both mean "everyone slows down, pitting is cheap right now"
df['AnyCautionThisLap'] = ((df['SafetyCarThisLap'] == 1) | (df['VSCThisLap'] == 1)).astype(int)

# Was there a caution on the PREVIOUS lap? shift(1) looks BACKWARD into the past,
# which is always safe - that's information we genuinely had at the time. Grouping
# by Year too, same reason as build_target.py: round numbers repeat across seasons.
driver_race_groups = df.groupby(['Year', 'RoundNumber', 'Driver'])
df['CautionPrevLap'] = driver_race_groups['AnyCautionThisLap'].shift(1).fillna(0).astype(int)

# The good one: a caution that JUST appeared (on now, off last lap). Everyone
# dives in the moment it comes out, then the rush dies down. Splitting "brand new"
# from "already running" captures that, which one lumped-together flag would miss.
df['CautionJustStarted'] = ((df['AnyCautionThisLap'] == 1) & (df['CautionPrevLap'] == 0)).astype(int)

# FEATURE 2: Tire degradation

# A lap time only tells us about tire wear if nothing ELSE was slowing the car
# down. Three things break that:
#   in-lap  -> driver crawls into the pit lane
#   out-lap -> driver crawls back out
#   caution -> safety car / VSC / yellow forces the whole field to slow
# Under a safety car laps run far slower than normal. Letting those through makes
# the feature read a huge "degradation" that has nothing to do with rubber.
df['IsInLap']  = df['PitInTime'].notna()
df['IsOutLap'] = df['PitOutTime'].notna()

df['DistortedLap'] = (df['IsInLap'] | df['IsOutLap'] |
                      (df['AnyCautionThisLap'] == 1) |
                      (df['RedFlagThisLap'] == 1) |
                      (df['YellowThisLap'] == 1))

# CleanLapTime = lap times we trust. Distorted ones get blanked out entirely.
df['CleanLapTime'] = df['LapTimeSeconds']
df.loc[df['DistortedLap'], 'CleanLapTime'] = np.nan

# Group by Year, Stint too, because tires RESET at every pit stop, and round
# numbers repeat across seasons.
df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'Stint', 'LapNumber']).reset_index(drop=True)
stint_groups = df.groupby(['Year', 'RoundNumber', 'Driver', 'Stint'])

# expanding().min() tracks the fastest trustworthy lap SO FAR in the stint. It
# only ever looks backward, never at future laps, so no leakage.
df['StintBestSoFar'] = stint_groups['CleanLapTime'].transform(lambda s: s.expanding().min())

# The feature: how many seconds off my own best pace am I on these tires?
df['TireDegDelta'] = df['CleanLapTime'] - df['StintBestSoFar']

# On a distorted lap the delta is undefined, so carry forward the last honest
# reading instead. ffill only reaches backward in time, so it stays leak-free.
df['TireDegDelta'] = df.groupby(
    ['Year', 'RoundNumber', 'Driver', 'Stint'])['TireDegDelta'].ffill()

# Start of a stint has no clean lap behind it yet. Tires are fresh -> 0.
df['TireDegDelta'] = df['TireDegDelta'].fillna(0)

# FEATURE 3: Wet conditions (free weather proxy)

# If a driver is on INTERMEDIATE or WET tires, it's raining. These two compound
# names are the only ones that survived unchanged across every season - the dry
# compound names (HYPERSOFT, ULTRASOFT, SUPERSOFT in 2018-19; SOFT/MEDIUM/HARD
# from 2019 on) changed, but wet-weather tires never did.
df['WetConditions'] = df['Compound'].isin(['INTERMEDIATE', 'WET']).astype(int)

# FEATURE 4: Race progress and laps remaining

# How long is this race? Take the highest lap number anyone reached, which is the
# winner's count. The +1 is a correction: build_target.py dropped every driver's
# final lap, so what we can still see is one lap short of the real distance.
#
# Computed per YEAR + RACE, never per driver, and now that matters even more:
# with 7 seasons of retirements mixed in, using a driver's own max lap would
# encode their retirement as "100% through the race" for a huge chunk of the data.
df['RaceTotalLaps'] = df.groupby(['Year', 'RoundNumber'])['LapNumber'].transform('max') + 1

# Progress is a 0-to-1 fraction, comparable across circuits AND across years -
# useful now that race distances vary not just by track but by season's calendar.
df['RaceProgress']  = df['LapNumber'] / df['RaceTotalLaps']
df['LapsRemaining'] = df['RaceTotalLaps'] - df['LapNumber']

# FEATURE 5: Stops made so far

df = df.sort_values(['Year', 'RoundNumber', 'Driver', 'LapNumber']).reset_index(drop=True)

# cumsum() adds up the pit flags as we walk down each driver's race. Grouping by
# Year keeps this from carrying a stop count over between different seasons.
df['StopsSoFar'] = df.groupby(['Year', 'RoundNumber', 'Driver'])['PittedThisLap'].cumsum()

# Sanity checks

print("Tire degradation (green and caution should be similar magnitude):")
print("  green flag laps:  {:.3f}s".format(df[df.AnyCautionThisLap == 0]['TireDegDelta'].mean()))
print("  caution laps:     {:.3f}s".format(df[df.AnyCautionThisLap == 1]['TireDegDelta'].mean()))

print("\nDegradation, no pit next vs pit next:")
for label, subset in [('all laps',      df),
                      ('green flag',    df[df.AnyCautionThisLap == 0]),
                      ('under caution', df[df.AnyCautionThisLap == 1])]:
    print("  {:14s} {:.3f}s  vs  {:.3f}s".format(
        label,
        subset[subset.PitNextLap == 0]['TireDegDelta'].mean(),
        subset[subset.PitNextLap == 1]['TireDegDelta'].mean()))

print("\nPit rate on the NEXT lap, by track situation:")
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

print("\nPit rate on the NEXT lap, by race phase:")
bins = [0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.01]
labels = ['0-15%', '15-30%', '30-45%', '45-60%', '60-75%', '75-100%']
phase = pd.cut(df['RaceProgress'], bins=bins, labels=labels, right=False)
for p in labels:
    subset = df[phase == p]
    print("  {:8s} n={:6d}   {:5.2f}%".format(p, len(subset), 100 * subset['PitNextLap'].mean()))

print("\nPit rate on the NEXT lap, by stops already made:")
for k in sorted(df['StopsSoFar'].unique()):
    subset = df[df.StopsSoFar == k]
    print("  {} stops   n={:6d}   {:5.2f}%".format(int(k), len(subset), 100 * subset['PitNextLap'].mean()))

df.to_csv('f1_all_features.csv', index=False)
print("\nSaved to f1_all_features.csv")