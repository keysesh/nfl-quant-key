import pandas as pd
import pytest
from pathlib import Path


REQUIRED_COLUMNS = [
    'rank',
    'bet_type',
    'game',
    'pick',
    'our_prob',
    'market_prob',
    'edge',
    'market_odds',
    'bet_size',
    'potential_profit',
]


def test_unified_recommendations_columns_exist():
    csv = Path('reports/unified_betting_recommendations.csv')
    assert csv.exists(), f"Missing file: {csv}"
    df = pd.read_csv(csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing required columns in unified_betting_recommendations.csv: {missing}"


def test_dashboard_data_generated_columns_exist():
    csv = Path('reports/dashboard_data.csv')
    # If the generator hasn't created this file yet, skip (not fatal)
    if not csv.exists():
        pytest.skip("reports/dashboard_data.csv not present; run generate_dashboard_with_predictions.py first")
    df = pd.read_csv(csv)
    # dashboard_data may have slightly different column names; check at least these
    for col in ['player', 'market_display', 'line', 'predicted_value', 'edge_pct']:
        assert col in df.columns, f"Missing column in reports/dashboard_data.csv: {col}"
