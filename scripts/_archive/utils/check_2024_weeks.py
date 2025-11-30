#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

print('CHECKING EXISTING 2024 DATA')
print()

have_weeks = set()

# Check weeks 1-4
p1 = Path('data/historical/player_props_2024_weeks_1_4.csv')
if p1.exists():
    df1 = pd.read_csv(p1)
    df1['date'] = pd.to_datetime(df1['commence_time']).dt.date
    start = pd.Timestamp('2024-09-05').date()
    df1['week'] = df1['date'].apply(lambda d: max(1, min(18, ((d - start).days // 7) + 1)))

    weeks1 = sorted(df1['week'].unique())
    print(f'Weeks 1-4 file: {len(df1):,} props, weeks {weeks1}')
    have_weeks.update(weeks1)
else:
    print('Weeks 1-4: Not found')

# Check weeks 5-18
p2 = Path('data/historical/player_props_2024_weeks_5_18.csv')
if p2.exists():
    df2 = pd.read_csv(p2)
    df2['date'] = pd.to_datetime(df2['commence_time']).dt.date
    start = pd.Timestamp('2024-09-05').date()
    df2['week'] = df2['date'].apply(lambda d: max(1, min(18, ((d - start).days // 7) + 1)))

    weeks2 = sorted(df2['week'].unique())
    print(f'Weeks 5-18 file: {len(df2):,} props, weeks {weeks2}')
    have_weeks.update(weeks2)
else:
    print('Weeks 5-18: Not found')

print()
missing = sorted(set(range(1, 19)) - have_weeks)
print(f'HAVE: {sorted(have_weeks)}')
print(f'MISSING: {missing if missing else "NONE"}')
