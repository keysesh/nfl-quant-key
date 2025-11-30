import json
import glob
import pandas as pd

# Find all simulation result files
files = sorted(glob.glob('reports/sim_2025_08_*.json'))

results = []
for file in files:
    with open(file) as f:
        data = json.load(f)
        results.append(data)

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv('reports/week8_bets.csv', index=False)
print(f'✅ Exported {len(results)} games to reports/week8_bets.csv')
print(f'\nColumns: {", ".join(df.columns.tolist())}')
print(f'\nPreview:')
print(df.head())
