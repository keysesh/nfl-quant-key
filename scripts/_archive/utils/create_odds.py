import pandas as pd

games = ['MIA_ATL','CHI_BAL','BUF_CAR','NYJ_CIN','SF_HOU','CLE_NE','NYG_PHI','TB_NO','DAL_DEN','TEN_IND']
rows = []

for g in games:
    for side in ['home_spread','away_spread','over','under']:
        rows.append({
            'game_id': f'2025_08_{g}',
            'side': side,
            'american_odds': -110
        })

df = pd.DataFrame(rows)
df.to_csv('data/odds_week8.csv', index=False)
print('✅ Created data/odds_week8.csv with 40 rows')
