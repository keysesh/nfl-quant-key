#!/usr/bin/env python3
"""
Generate game lines predictions using the active model.

Applies trained models to current week's spread and totals odds.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.features import get_feature_engine
from nfl_quant.utils.odds import american_to_decimal, calculate_ev
from configs.model_config import get_active_model_path

# Active model path - use canonical source from configs.model_config
ACTIVE_MODEL_PATH = get_active_model_path()


def get_current_season():
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


def load_game_odds(season: int = None):
    """Load current game lines odds from DraftKings file."""
    if season is None:
        season = get_current_season()

    # Find most recent odds file
    data_dir = project_root / 'data'
    odds_files = list(data_dir.glob(f'odds_draftkings_*_{season}.csv'))

    if not odds_files:
        raise FileNotFoundError(f"No DraftKings odds files found for {season}")

    # Use most recent file
    odds_file = sorted(odds_files)[-1]
    print(f"Loading odds from: {odds_file.name}")

    odds = pd.read_csv(odds_file)

    # Parse into game-level data
    games = {}
    for _, row in odds.iterrows():
        game_id = row['game_id']
        if game_id not in games:
            games[game_id] = {
                'game_id': game_id,
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'commence_time': row['commence_time'],
            }

        market = row['market']
        side = row['side']
        point = row['point']
        price = row['price']

        if market == 'spread':
            if side == 'home':
                games[game_id]['spread_line'] = point
                games[game_id]['spread_home_price'] = price
            else:
                games[game_id]['spread_away_price'] = price
        elif market == 'total':
            games[game_id]['total_line'] = point
            if side == 'over':
                games[game_id]['over_price'] = price
            else:
                games[game_id]['under_price'] = price
        elif market == 'moneyline':
            if side == 'home':
                games[game_id]['home_moneyline'] = price
            else:
                games[game_id]['away_moneyline'] = price

    df = pd.DataFrame(games.values())
    print(f"Loaded {len(df)} games with odds")
    return df


def build_game_features(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Build features for prediction matching training features."""
    engine = get_feature_engine()
    features_list = []

    for _, row in games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        try:
            # Get defense EPA features
            home_pass_epa = engine.get_pass_defense_epa(home_team, season, week) if hasattr(engine, 'get_pass_defense_epa') else 0
            away_pass_def = engine.get_pass_defense_epa(away_team, season, week) if hasattr(engine, 'get_pass_defense_epa') else 0
            home_rush_epa = engine.get_rush_defense_epa(home_team, season, week) if hasattr(engine, 'get_rush_defense_epa') else 0
            away_rush_def = engine.get_rush_defense_epa(away_team, season, week) if hasattr(engine, 'get_rush_defense_epa') else 0

            spread_line = row.get('spread_line', 0) or 0
            total_line = row.get('total_line', 0) or 0

            features = {
                'game_id': row['game_id'],
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': row.get('commence_time'),

                # Betting lines
                'total_line': total_line,
                'spread_line': spread_line,
                'home_moneyline': row.get('home_moneyline'),
                'away_moneyline': row.get('away_moneyline'),
                'over_price': row.get('over_price'),
                'under_price': row.get('under_price'),
                'spread_home_price': row.get('spread_home_price'),
                'spread_away_price': row.get('spread_away_price'),

                # Line-derived features (match training)
                'implied_total': total_line,
                'implied_home_score': (total_line - spread_line) / 2 if total_line else 0,
                'implied_away_score': (total_line + spread_line) / 2 if total_line else 0,

                # Spread features
                'spread_magnitude': abs(spread_line),
                'is_pick_em': 1 if abs(spread_line) <= 1.5 else 0,
                'is_big_favorite': 1 if abs(spread_line) >= 7 else 0,

                # Total line buckets
                'is_high_total': 1 if total_line >= 48 else 0,
                'is_low_total': 1 if total_line <= 40 else 0,

                # Defense EPA
                'home_pass_def_epa': home_pass_epa,
                'away_pass_def_epa': away_pass_def,
                'home_rush_def_epa': home_rush_epa,
                'away_rush_def_epa': away_rush_def,
                'total_def_epa': home_pass_epa + away_pass_def + home_rush_epa + away_rush_def,
            }

            features_list.append(features)

        except Exception as e:
            print(f"Error building features for {home_team} vs {away_team}: {e}")
            continue

    return pd.DataFrame(features_list)


def predict_game_lines(games_df: pd.DataFrame) -> pd.DataFrame:
    """Apply trained models to generate predictions."""
    if not ACTIVE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Active model not found: {ACTIVE_MODEL_PATH}")

    bundle = joblib.load(ACTIVE_MODEL_PATH)
    models = bundle.get('models', {})
    thresholds = bundle.get('thresholds', {})

    results = []

    for _, row in games_df.iterrows():
        game_result = {
            'game_id': row['game_id'],
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'commence_time': row.get('commence_time'),
        }

        # Totals prediction
        if 'game_totals' in models:
            totals_model = models['game_totals']
            totals_threshold = thresholds.get('game_totals', {}).get('threshold', 0.55)

            totals_features = [
                'total_line', 'is_high_total', 'is_low_total',
                'spread_magnitude', 'total_def_epa',
                'home_pass_def_epa', 'away_pass_def_epa',
                'home_rush_def_epa', 'away_rush_def_epa',
            ]

            X = pd.DataFrame([row[totals_features].values], columns=totals_features)
            X = X.fillna(0)

            # Apply preprocessing
            X = totals_model['imputer'].transform(X)
            X = totals_model['scaler'].transform(X)

            probs = totals_model['model'].predict_proba(X)[0]
            p_over = probs[1]
            p_under = probs[0]

            # Get odds
            over_price = row.get('over_price', -110)
            under_price = row.get('under_price', -110)
            over_decimal = american_to_decimal(over_price)
            under_decimal = american_to_decimal(under_price)

            # Calculate EV
            over_ev = calculate_ev(p_over, over_decimal)
            under_ev = calculate_ev(p_under, under_decimal)

            # Determine best bet
            if p_over >= totals_threshold and over_ev > 0:
                game_result['totals_pick'] = 'OVER'
                game_result['totals_line'] = row['total_line']
                game_result['totals_prob'] = p_over
                game_result['totals_ev'] = over_ev
                game_result['totals_price'] = over_price
                game_result['totals_validated'] = True
            elif p_under >= totals_threshold and under_ev > 0:
                game_result['totals_pick'] = 'UNDER'
                game_result['totals_line'] = row['total_line']
                game_result['totals_prob'] = p_under
                game_result['totals_ev'] = under_ev
                game_result['totals_price'] = under_price
                game_result['totals_validated'] = True
            else:
                game_result['totals_pick'] = None
                game_result['totals_validated'] = False

        # Spreads prediction
        if 'game_spreads' in models:
            spreads_model = models['game_spreads']
            spreads_threshold = thresholds.get('game_spreads', {}).get('threshold', 0.55)

            spreads_features = [
                'spread_line', 'spread_magnitude', 'is_pick_em', 'is_big_favorite',
                'total_line', 'total_def_epa',
                'home_pass_def_epa', 'away_pass_def_epa',
            ]

            X = pd.DataFrame([row[spreads_features].values], columns=spreads_features)
            X = X.fillna(0)

            # Apply preprocessing
            X = spreads_model['imputer'].transform(X)
            X = spreads_model['scaler'].transform(X)

            probs = spreads_model['model'].predict_proba(X)[0]
            p_home_cover = probs[1]
            p_away_cover = probs[0]

            # Get odds
            home_price = row.get('spread_home_price', -110)
            away_price = row.get('spread_away_price', -110)
            home_decimal = american_to_decimal(home_price)
            away_decimal = american_to_decimal(away_price)

            # Calculate EV
            home_ev = calculate_ev(p_home_cover, home_decimal)
            away_ev = calculate_ev(p_away_cover, away_decimal)

            # Determine best bet
            if p_home_cover >= spreads_threshold and home_ev > 0:
                game_result['spread_pick'] = f"{row['home_team']} {row['spread_line']}"
                game_result['spread_prob'] = p_home_cover
                game_result['spread_ev'] = home_ev
                game_result['spread_price'] = home_price
                game_result['spread_validated'] = True
            elif p_away_cover >= spreads_threshold and away_ev > 0:
                game_result['spread_pick'] = f"{row['away_team']} +{abs(row['spread_line'])}"
                game_result['spread_prob'] = p_away_cover
                game_result['spread_ev'] = away_ev
                game_result['spread_price'] = away_price
                game_result['spread_validated'] = True
            else:
                game_result['spread_pick'] = None
                game_result['spread_validated'] = False

        results.append(game_result)

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict game lines')
    parser.add_argument('--week', type=int, required=True, help='NFL week')
    parser.add_argument('--season', type=int, default=None, help='NFL season')
    args = parser.parse_args()

    season = args.season or get_current_season()
    week = args.week

    print("=" * 60)
    print(f"GAME LINES PREDICTIONS - Week {week}, {season}")
    print("=" * 60)

    # Load odds
    games = load_game_odds(season)

    # Build features
    games_df = build_game_features(games, season, week)

    # Generate predictions
    predictions = predict_game_lines(games_df)

    # Display results
    print("\n" + "=" * 60)
    print("VALIDATED GAME LINE BETS")
    print("=" * 60)

    totals_bets = predictions[predictions['totals_validated'] == True]
    spread_bets = predictions[predictions['spread_validated'] == True]

    if len(totals_bets) > 0:
        print(f"\n--- TOTALS ({len(totals_bets)} bets) ---")
        for _, bet in totals_bets.iterrows():
            print(f"  {bet['matchup']}: {bet['totals_pick']} {bet['totals_line']}")
            print(f"    Prob: {bet['totals_prob']:.1%} | EV: {bet['totals_ev']*100:+.1f}% | Price: {bet['totals_price']}")

    if len(spread_bets) > 0:
        print(f"\n--- SPREADS ({len(spread_bets)} bets) ---")
        for _, bet in spread_bets.iterrows():
            print(f"  {bet['matchup']}: {bet['spread_pick']}")
            print(f"    Prob: {bet['spread_prob']:.1%} | EV: {bet['spread_ev']*100:+.1f}% | Price: {bet['spread_price']}")

    if len(totals_bets) == 0 and len(spread_bets) == 0:
        print("\nNo validated game line bets for this week.")

    # Save predictions
    output_file = project_root / 'reports' / f'game_lines_predictions_week{week}_{season}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"\nSaved predictions to: {output_file}")

    return predictions


if __name__ == '__main__':
    main()
