#!/usr/bin/env python3
"""
Generate Hybrid Recommendations

Combines main model predictions with edge signals for higher conviction plays.
When both the main model and an edge agree on direction, confidence is boosted.

Usage:
    python scripts/predict/generate_hybrid_recommendations.py --week 15
    python scripts/predict/generate_hybrid_recommendations.py --week 15 --save
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import DATA_DIR, MODELS_DIR
from nfl_quant.edges.ensemble import EdgeEnsemble
from nfl_quant.monitoring.edge_logger import EdgePredictionLogger
from configs.edge_config import EDGE_MARKETS


def load_main_predictions(week: int) -> pd.DataFrame:
    """
    Load main model predictions for a week.

    Args:
        week: NFL week

    Returns:
        DataFrame with main model predictions
    """
    # Look for predictions file
    preds_path = DATA_DIR / 'predictions' / f'predictions_week{week}.csv'

    if not preds_path.exists():
        # Try alternate location
        preds_path = DATA_DIR / f'player_prop_recommendations_week{week}.csv'

    if not preds_path.exists():
        print(f"No main model predictions found for week {week}")
        return pd.DataFrame()

    df = pd.read_csv(preds_path)
    print(f"Loaded {len(df)} main model predictions from {preds_path}")

    return df


def load_edge_data(week: int, season: int) -> pd.DataFrame:
    """
    Load data needed for edge evaluation.

    Args:
        week: NFL week
        season: NFL season

    Returns:
        DataFrame with features for edge evaluation
    """
    # Load enriched training data (for trailing stats)
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if enriched_path.exists():
        df = pd.read_csv(enriched_path, low_memory=False)
        # Get latest trailing stats per player
        latest = df.sort_values(['player', 'season', 'week']).groupby('player').last()
        return latest.reset_index()

    return pd.DataFrame()


def generate_edge_signals(
    predictions: pd.DataFrame,
    ensemble: EdgeEnsemble,
) -> pd.DataFrame:
    """
    Generate edge signals for each prediction.

    Args:
        predictions: DataFrame with main model predictions
        ensemble: Trained EdgeEnsemble

    Returns:
        DataFrame with edge signals added
    """
    results = []

    for _, row in predictions.iterrows():
        market = row.get('market', '')

        # Skip if market not supported by edges
        if market not in EDGE_MARKETS:
            row_dict = row.to_dict()
            row_dict['edge_direction'] = None
            row_dict['edge_source'] = 'NOT_SUPPORTED'
            row_dict['edge_confidence'] = 0.0
            results.append(row_dict)
            continue

        # Evaluate with edge ensemble
        try:
            decision = ensemble.evaluate_bet(row, market)

            row_dict = row.to_dict()
            row_dict['edge_direction'] = decision.direction if decision.should_bet else None
            row_dict['edge_source'] = decision.source.value
            row_dict['edge_confidence'] = decision.confidence if decision.should_bet else 0.0
            results.append(row_dict)

        except Exception as e:
            row_dict = row.to_dict()
            row_dict['edge_direction'] = None
            row_dict['edge_source'] = 'ERROR'
            row_dict['edge_confidence'] = 0.0
            results.append(row_dict)

    return pd.DataFrame(results)


def calculate_hybrid_score(row: pd.Series) -> float:
    """
    Calculate hybrid confidence score.

    Boosts confidence when main model and edge agree.

    Args:
        row: Series with main and edge predictions

    Returns:
        Hybrid confidence score
    """
    main_conf = row.get('confidence', row.get('prob_under', 0.5))
    main_direction = row.get('direction', row.get('recommendation', ''))

    edge_conf = row.get('edge_confidence', 0.0)
    edge_direction = row.get('edge_direction', None)

    # If edge didn't trigger, use main model only
    if edge_direction is None or edge_conf == 0:
        return main_conf

    # Normalize main direction
    if 'UNDER' in str(main_direction).upper():
        main_direction = 'UNDER'
    elif 'OVER' in str(main_direction).upper():
        main_direction = 'OVER'
    else:
        return main_conf

    # Check if directions agree
    if main_direction == edge_direction:
        # Boost confidence: weighted average with 10% bonus
        hybrid = (main_conf + edge_conf) / 2
        hybrid = min(0.95, hybrid * 1.10)  # Cap at 95%, 10% boost
        return hybrid
    else:
        # Directions conflict - reduce confidence
        return main_conf * 0.9


def generate_hybrid_recommendations(
    week: int,
    season: int = None,
    min_hybrid_conf: float = 0.55,
) -> pd.DataFrame:
    """
    Generate hybrid recommendations combining main model and edges.

    Args:
        week: NFL week
        season: NFL season
        min_hybrid_conf: Minimum hybrid confidence to recommend

    Returns:
        DataFrame with hybrid recommendations
    """
    if season is None:
        season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1

    print(f"Generating hybrid recommendations for Week {week}, {season}")

    # Load main predictions
    main_preds = load_main_predictions(week)

    if main_preds.empty:
        print("No main model predictions available")
        return pd.DataFrame()

    # Load edge ensemble
    try:
        ensemble = EdgeEnsemble.load()
        print(f"Loaded edge ensemble (v{ensemble.lvt_edge.version})")
    except Exception as e:
        print(f"Could not load edge ensemble: {e}")
        return main_preds

    # Generate edge signals
    print("Evaluating edge signals...")
    hybrid = generate_edge_signals(main_preds, ensemble)

    # Calculate hybrid scores
    hybrid['hybrid_confidence'] = hybrid.apply(calculate_hybrid_score, axis=1)

    # Determine hybrid direction
    def get_hybrid_direction(row):
        main_dir = row.get('direction', row.get('recommendation', ''))
        edge_dir = row.get('edge_direction')

        if 'UNDER' in str(main_dir).upper():
            main_dir = 'UNDER'
        elif 'OVER' in str(main_dir).upper():
            main_dir = 'OVER'
        else:
            main_dir = None

        # If edge agrees, use that direction
        if edge_dir and main_dir == edge_dir:
            return main_dir
        # If edge disagrees, still use main but flag it
        elif edge_dir:
            return f"{main_dir} (conflict)"
        # No edge signal
        else:
            return main_dir

    hybrid['hybrid_direction'] = hybrid.apply(get_hybrid_direction, axis=1)

    # Add hybrid source
    def get_hybrid_source(row):
        edge_source = row.get('edge_source', 'NONE')
        main_dir = row.get('direction', row.get('recommendation', ''))
        edge_dir = row.get('edge_direction')

        if 'UNDER' in str(main_dir).upper():
            main_dir = 'UNDER'
        elif 'OVER' in str(main_dir).upper():
            main_dir = 'OVER'

        if edge_source in ['LVT_ONLY', 'PLAYER_BIAS_ONLY', 'BOTH']:
            if main_dir == edge_dir:
                return f"MAIN+{edge_source}"
            else:
                return f"MAIN_ONLY (edge conflict)"
        else:
            return "MAIN_ONLY"

    hybrid['hybrid_source'] = hybrid.apply(get_hybrid_source, axis=1)

    # Filter by minimum confidence
    recommendations = hybrid[hybrid['hybrid_confidence'] >= min_hybrid_conf].copy()

    # Sort by hybrid confidence
    recommendations = recommendations.sort_values('hybrid_confidence', ascending=False)

    print(f"\nGenerated {len(recommendations)} hybrid recommendations "
          f"(>={min_hybrid_conf:.0%} confidence)")

    return recommendations


def print_recommendations(df: pd.DataFrame, top_n: int = 20) -> None:
    """Print top recommendations to console."""
    if df.empty:
        print("No recommendations available")
        return

    print("\n" + "=" * 80)
    print("TOP HYBRID RECOMMENDATIONS")
    print("=" * 80)

    cols_to_show = ['player', 'market', 'line', 'hybrid_direction', 'hybrid_confidence', 'hybrid_source']
    available_cols = [c for c in cols_to_show if c in df.columns]

    # Add fallback columns
    if 'player' not in df.columns and 'player_name' in df.columns:
        df['player'] = df['player_name']
        available_cols = [c if c != 'player' else 'player' for c in available_cols]

    top = df.head(top_n)

    print(f"\n{'Player':<25} {'Market':<20} {'Line':<8} {'Dir':<15} {'Conf':<8} {'Source':<20}")
    print("-" * 100)

    for _, row in top.iterrows():
        player = str(row.get('player', row.get('player_name', 'Unknown')))[:24]
        market = str(row.get('market', ''))[:19]
        line = row.get('line', 0)
        direction = str(row.get('hybrid_direction', ''))[:14]
        conf = row.get('hybrid_confidence', 0)
        source = str(row.get('hybrid_source', ''))[:19]

        print(f"{player:<25} {market:<20} {line:<8.1f} {direction:<15} {conf:.1%}    {source:<20}")

    # Summary by source
    print("\n" + "-" * 60)
    print("BY SOURCE:")
    for source in df['hybrid_source'].unique():
        source_df = df[df['hybrid_source'] == source]
        avg_conf = source_df['hybrid_confidence'].mean()
        print(f"  {source}: {len(source_df)} bets, avg conf {avg_conf:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Generate Hybrid Recommendations")
    parser.add_argument('--week', type=int, required=True, help='NFL week')
    parser.add_argument('--season', type=int, help='NFL season')
    parser.add_argument('--min-conf', type=float, default=0.55,
                        help='Minimum hybrid confidence')
    parser.add_argument('--save', action='store_true', help='Save recommendations')
    parser.add_argument('--log', action='store_true', help='Log to prediction logger')
    args = parser.parse_args()

    # Generate recommendations
    recommendations = generate_hybrid_recommendations(
        week=args.week,
        season=args.season,
        min_hybrid_conf=args.min_conf,
    )

    if recommendations.empty:
        return

    # Print to console
    print_recommendations(recommendations)

    # Save to file
    if args.save:
        output_path = DATA_DIR / 'predictions' / f'hybrid_recommendations_week{args.week}.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

    # Log to prediction logger
    if args.log:
        logger = EdgePredictionLogger()

        for _, row in recommendations.iterrows():
            logger.log_prediction({
                'player': row.get('player', row.get('player_name')),
                'market': row.get('market'),
                'line': row.get('line'),
                'direction': row.get('hybrid_direction', '').replace(' (conflict)', ''),
                'confidence': row.get('hybrid_confidence'),
                'source': row.get('hybrid_source'),
                'week': args.week,
                'season': args.season or datetime.now().year,
            })

        filepath = logger.save()
        print(f"\nLogged predictions to: {filepath}")


if __name__ == '__main__':
    main()
