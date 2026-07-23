import fastf1
import pandas as pd
from pathlib import Path
import time
import os

cache_dir = 'C:/F1_data_cache'
Path(cache_dir).mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# FastF1 allows 500 API calls per hour. A fresh race costs roughly a dozen, so
# expect ~40 races per run before the limit trips. It resets on the hour, so the
# workflow is: run this, wait an hour, run it again, repeat until it says done.
#
# Everything is saved per year, and races already saved are skipped without any
# API call at all. Nothing is ever re-downloaded, so each run picks up exactly
# where the last one stopped.

def is_rate_limit(error):
    text = str(error).lower()
    return 'rate' in text or '500 calls' in text

# One-time migration: if the old single-file download is sitting there, split it
# into per-year files so those races count as already done and don't get re-fetched.
if os.path.exists('f1_all_races.csv'):
    already_split = all(os.path.exists('f1_laps_{}.csv'.format(y)) for y in [2018, 2019, 2024])
    if not already_split:
        print("Found f1_all_races.csv from the earlier run - splitting it by year")
        old = pd.read_csv('f1_all_races.csv', low_memory=False)
        for year, chunk in old.groupby('Year'):
            path = 'f1_laps_{}.csv'.format(int(year))
            if not os.path.exists(path):
                chunk.to_csv(path, index=False)
                print("  {} -> {} laps, {} races".format(path, len(chunk), chunk['RoundNumber'].nunique()))

stopped_by_limit = False

for year in YEARS:
    if stopped_by_limit:
        break

    year_path = 'f1_laps_{}.csv'.format(year)

    # What do we already have for this year?
    if os.path.exists(year_path):
        existing = pd.read_csv(year_path, low_memory=False)
        done_rounds = set(existing['RoundNumber'].unique())
    else:
        existing = None
        done_rounds = set()

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        if is_rate_limit(e):
            print("\n{}: rate limit reached on the schedule call.".format(year))
            stopped_by_limit = True
            break
        print("\n{}: could not load schedule - {}".format(year, e))
        continue

    # Round 0 is pre-season testing, not a race
    wanted = [r for r in schedule['RoundNumber'] if r != 0]
    todo = [r for r in wanted if r not in done_rounds]

    print("\n{}: {} races on calendar | {} already saved | {} to fetch".format(
        year, len(wanted), len(done_rounds), len(todo)))

    if not todo:
        continue

    new_laps = []
    for _, event in schedule.iterrows():
        round_num = event['RoundNumber']
        event_name = event['EventName']

        if round_num == 0 or round_num in done_rounds:
            continue

        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load()

            laps = session.laps.copy()
            laps['Year'] = year
            laps['RoundNumber'] = round_num
            laps['EventName'] = event_name

            new_laps.append(laps)
            print("  R{:02d} {:34s} {:5d} laps".format(round_num, event_name[:34], len(laps)))
            time.sleep(1)

        except Exception as e:
            if is_rate_limit(e):
                print("  R{:02d} {:34s} RATE LIMIT - stopping here".format(round_num, event_name[:34]))
                stopped_by_limit = True
                break
            # A genuine data gap. Log it and carry on to the next race.
            print("  R{:02d} {:34s} FAILED: {}".format(round_num, event_name[:34], str(e)[:55]))
            continue

    # Save whatever this run managed to get, even if the limit cut it short.
    # Losing 30 downloaded races to an unsaved crash would be painful.
    if new_laps:
        parts = ([existing] if existing is not None else []) + new_laps
        combined_year = pd.concat(parts, ignore_index=True)
        combined_year.to_csv(year_path, index=False)
        print("  saved {} ({} races total)".format(year_path, combined_year['RoundNumber'].nunique()))

# Stitch every year file we have into one dataset

frames = []
for year in YEARS:
    path = 'f1_laps_{}.csv'.format(year)
    if os.path.exists(path):
        frames.append(pd.read_csv(path, low_memory=False))

combined = pd.concat(frames, ignore_index=True)
combined.to_csv('f1_all_races.csv', index=False)

print("\nCURRENT TOTAL: {} laps across {} races".format(
    len(combined), combined.groupby(['Year', 'RoundNumber']).ngroups))
print("\nLaps and races per season:")
print(combined.groupby('Year').agg(laps=('LapNumber', 'size'),
                                   races=('RoundNumber', 'nunique')).to_string())

if stopped_by_limit:
    print("\nStopped early - hourly API limit reached.")
    print("Wait about an hour, then run this again. It will resume where it left off.")
else:
    print("\nAll requested seasons are downloaded.")

print("\nSaved to f1_all_races.csv")