#!/usr/bin/env python3
"""
Generate Player Prop Recommendations for Tonight's Game (ARI @ DAL)
===================================================================

Matches model predictions to available DraftKings prop lines and
calculates edge using proper probability conversion.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def american_to_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Probability (0-1)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def calculate_over_probability(mean: float, std: float, line: float) -> float:
    """
    Calculate probability of going OVER a line using normal distribution.

    Args:
        mean: Model's mean projection
        std: Standard deviation
        line: Betting line

    Returns:
        Probability of OVER (0-1)
    """
    if std == 0 or pd.isna(std):
        # If no variance, use simple comparison
        return 1.0 if mean > line else 0.0

    # Calculate z-score
    z = (line - mean) / std

    # P(X > line) = 1 - CDF(line)
    prob_over = 1 - stats.norm.cdf(z)

    # Bound between 5% and 95% for safety
    return np.clip(prob_over, 0.05, 0.95)


def main():
    """Generate player prop recommendations for tonight."""

    print("=" * 80)
    print("PLAYER PROP RECOMMENDATIONS - ARI @ DAL (Monday Night Football)")
    print("=" * 80)

    # Load model predictions
    predictions_file = Path('data/model_predictions_week9.csv')
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return

    preds = pd.read_csv(predictions_file)
    print(f"\n📊 Loaded {len(preds)} player predictions")

    # Filter to tonight's game
    tonight = preds[preds['opponent'].isin(['ARI', 'DAL'])]
    print(f"   Tonight's game players: {len(tonight)}")

    # Load prop odds
    props_file = Path('data/nfl_player_props_draftkings.csv')
    if not props_file.exists():
        print(f"❌ Props file not found: {props_file}")
        return

    props = pd.read_csv(props_file)
    print(f"📈 Loaded {len(props)} prop lines")

    # Filter to ARI @ DAL game
    tonight_props = props[
        (props['home_team'] == 'Dallas Cowboys') &
        (props['away_team'] == 'Arizona Cardinals')
    ]
    print(f"   Tonight's game props: {len(tonight_props)}")

    # Map markets to prediction columns
    market_mapping = {
        'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
        'player_pass_tds': ('passing_tds_mean', 'passing_tds_std'),
        'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
        'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
        'player_rush_tds': ('rushing_tds_mean', 'rushing_tds_std'),
        'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
        'player_receptions': ('receptions_mean', 'receptions_std'),
        'player_rec_tds': ('receiving_tds_mean', 'receiving_tds_std'),
    }

    recommendations = []

    # Process each market
    for market, (mean_col, std_col) in market_mapping.items():
        market_props = tonight_props[tonight_props['market'] == market]

        if len(market_props) == 0:
            continue

        print(f"\n🎯 Processing {market}: {len(market_props)} props")

        for _, prop in market_props.iterrows():
            player_name = prop['player_name']
            line = prop['line']

            if pd.isna(line):
                continue  # Skip props without lines (e.g., anytime TD)

            # Find player prediction
            player_pred = tonight[tonight['player_name'].str.contains(player_name, case=False, na=False)]

            if len(player_pred) == 0:
                continue  # No prediction for this player

            player_pred = player_pred.iloc[0]

            # Get model projection
            if mean_col not in player_pred or std_col not in player_pred:
                continue

            mean = player_pred[mean_col]
            std = player_pred[std_col]

            if pd.isna(mean) or mean == 0:
                continue

            # Calculate probabilities
            model_prob_over = calculate_over_probability(mean, std, line)
            model_prob_under = 1 - model_prob_over

            # Get market odds
            market_odds = prop['odds']
            outcome_type = prop['outcome_type']

            if pd.isna(market_odds):
                continue

            market_prob = american_to_prob(market_odds)

            # Determine if this is Over or Under
            if outcome_type == 'Over':
                our_prob = model_prob_over
                pick = f"Over {line}"
            else:  # Under
                our_prob = model_prob_under
                pick = f"Under {line}"

            # Calculate edge
            edge = our_prob - market_prob

            # Calculate EV
            if market_odds < 0:
                payout = 100 / abs(market_odds)
            else:
                payout = market_odds / 100

            ev = (our_prob * payout) - ((1 - our_prob) * 1)
            roi = ev * 100

            # Only include if edge > 3% (balanced mode)
            if edge >= 0.03:
                recommendations.append({
                    'player': player_name,
                    'team': player_pred['team'],
                    'position': player_pred['position'],
                    'market': market.replace('player_', ''),
                    'pick': pick,
                    'line': line,
                    'odds': market_odds,
                    'model_mean': round(mean, 1),
                    'model_std': round(std, 1),
                    'our_prob': round(our_prob, 3),
                    'market_prob': round(market_prob, 3),
                    'edge': round(edge, 3),
                    'ev': round(ev, 3),
                    'roi_pct': round(roi, 1),
                })

    # Create DataFrame and sort
    if len(recommendations) == 0:
        print(f"\n⚠️  No props meet balanced criteria (3%+ edge)")
        return

    df = pd.DataFrame(recommendations)
    df = df.sort_values('edge', ascending=False)

    # Save to file
    output_file = Path('reports/TONIGHT_PLAYER_PROPS.csv')
    df.to_csv(output_file, index=False)

    print(f"\n💾 Saved {len(df)} recommendations to {output_file}")

    # Display top picks
    print(f"\n🔥 TOP 10 PLAYER PROP PICKS (by edge):")
    print("=" * 80)

    top10 = df.head(10)
    for _, row in top10.iterrows():
        print(f"\n{row['player']} ({row['team']} {row['position']}) - {row['market'].upper()}")
        print(f"  Pick: {row['pick']} @ {row['odds']}")
        print(f"  Model: {row['model_mean']} ± {row['model_std']}")
        print(f"  Edge: {row['edge']*100:.1f}% | ROI: {row['roi_pct']:.1f}%")
        print(f"  Our Prob: {row['our_prob']*100:.1f}% vs Market: {row['market_prob']*100:.1f}%")

    print(f"\n" + "=" * 80)
    print(f"✅ {len(df)} total player props meet balanced criteria (3%+ edge)")
    print("=" * 80)

    return df


if __name__ == '__main__':
    main()
