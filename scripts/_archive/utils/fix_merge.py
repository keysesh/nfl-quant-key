import pandas as pd

game_line = pd.read_csv('reports/unified_betting_recommendations.csv')
game_line = game_line[game_line['bet_type'] == 'Game Total'].copy()
game_line['player'] = ''

player_props = pd.read_csv('reports/full_framework_tonight.csv')

new_rows = []
for _, row in player_props.iterrows():
    new_rows.append({
        'bet_type': 'Player Prop',
        'game': 'WAS @ KC',
        'player': str(row['player']),
        'pick': str(row['pick']),
        'market': row['market'],
        'market_odds': row['odds'],
        'model_value': row['line'],
        'edge': row['edge_pct'] / 100,
        'our_prob': row['model_prob'],
        'market_prob': row['market_prob'],
        'ev': 0,
        'roi': row['roi_pct'] / 100,
        'bet_size': 1.0,
        'potential_profit': 1.0,
    })

unified = pd.concat([game_line, pd.DataFrame(new_rows)], ignore_index=True)
unified = unified.sort_values('our_prob', ascending=False).reset_index(drop=True)
unified['rank'] = range(1, len(unified) + 1)

unified.to_csv('reports/unified_betting_recommendations.csv', index=False)
print(f'✅ {len(unified)} picks saved')
