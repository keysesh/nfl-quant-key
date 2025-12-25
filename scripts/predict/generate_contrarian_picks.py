#!/usr/bin/env python3
"""
Generate Contrarian Betting Picks

This script generates betting picks using the contrarian strategy:
- FADE the model when it diverges significantly from the line
- Apply line-level rules that historically beat the vig
- EXCLUDE passing yards entirely (catastrophic -20% ROI)

Expected performance (from backtest):
- 58.5% Win Rate
- +11.7% ROI
- ~40-50 picks per week

Usage:
    python scripts/predict/generate_contrarian_picks.py --week 14
    python scripts/predict/generate_contrarian_picks.py --week 14 --min-roi 5
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.strategies.contrarian import (
    generate_contrarian_picks,
    summarize_picks,
    BetDirection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_props_with_predictions(week: int, season: int = 2025) -> pd.DataFrame:
    """
    Load props data with model predictions.

    Tries multiple data sources in order of preference.
    """
    project_root = Path(__file__).parent.parent.parent

    # Source 1: Unified recommendations (has everything we need)
    recs_path = project_root / 'reports' / 'CURRENT_WEEK_RECOMMENDATIONS.csv'
    if recs_path.exists():
        df = pd.read_csv(recs_path)
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} props from unified recommendations")
            return df

    # Source 2: Model predictions file
    preds_path = project_root / 'data' / f'model_predictions_week{week}.csv'
    props_path = project_root / 'data' / 'props' / f'odds_player_props_week{week}.csv'

    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        logger.info(f"Loaded {len(preds)} predictions from {preds_path}")

        # Try to merge with props for line info
        if props_path.exists():
            props = pd.read_csv(props_path)
            # Attempt merge (column names may vary)
            try:
                merged = preds.merge(
                    props[['player', 'market', 'line']],
                    left_on=['player_name'],
                    right_on=['player'],
                    how='left'
                )
                if 'line' in merged.columns and merged['line'].notna().sum() > 0:
                    return merged
            except Exception as e:
                logger.warning(f"Could not merge props: {e}")

        return preds

    # Source 3: Backtest results (for testing)
    backtest_path = project_root / 'data' / 'backtest' / 'walk_forward_with_lines_results.csv'
    if backtest_path.exists():
        df = pd.read_csv(backtest_path)
        df = df[df['week'] == week] if 'week' in df.columns else df
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} props from backtest results")
            return df

    raise FileNotFoundError(
        f"No data found for week {week}. "
        f"Tried: {recs_path}, {preds_path}, {backtest_path}"
    )


def display_picks(picks, show_all: bool = False):
    """Display picks in a formatted way."""
    if not picks:
        print("\nNo picks found matching criteria.")
        return

    # Group by confidence
    high_conf = [p for p in picks if p.confidence == "high"]
    med_conf = [p for p in picks if p.confidence == "medium"]
    speculative = [p for p in picks if p.confidence == "speculative"]

    if high_conf:
        print("\n" + "=" * 70)
        print("HIGH CONFIDENCE PICKS (Validated, ROI >= 8%)")
        print("=" * 70)
        for pick in high_conf:
            validated_tag = "✓ VALIDATED" if pick.validated else ""
            print(f"\n  {pick.direction.value} {pick.line} {pick.market} {validated_tag}")
            print(f"    Player: {pick.player}")
            print(f"    Rule: {pick.rule_name} (n={pick.sample_size:,})")
            print(f"    Model predicted: {pick.model_pred:.1f} (divergence: {pick.divergence_pct:+.1f}%)")
            print(f"    Expected: {pick.expected_wr:.1%} WR, {pick.expected_roi:+.1f}% ROI")

    if med_conf:
        print("\n" + "=" * 70)
        print("MEDIUM CONFIDENCE PICKS (Validated, ROI 3-8%)")
        print("=" * 70)
        for pick in med_conf:
            validated_tag = "✓ VALIDATED" if pick.validated else ""
            print(f"\n  {pick.direction.value} {pick.line} {pick.market} {validated_tag}")
            print(f"    Player: {pick.player}")
            print(f"    Rule: {pick.rule_name} (n={pick.sample_size:,})")
            print(f"    Model predicted: {pick.model_pred:.1f} (divergence: {pick.divergence_pct:+.1f}%)")
            print(f"    Expected: {pick.expected_wr:.1%} WR, {pick.expected_roi:+.1f}% ROI")

    if show_all and speculative:
        print("\n" + "-" * 70)
        print("SPECULATIVE PICKS (Not consistent across all seasons)")
        print("-" * 70)
        for pick in speculative:
            print(f"\n  {pick.direction.value} {pick.line} {pick.market} ⚠️ SPECULATIVE")
            print(f"    Player: {pick.player}")
            print(f"    Rule: {pick.rule_name} (n={pick.sample_size:,})")
            print(f"    Expected: {pick.expected_wr:.1%} WR, {pick.expected_roi:+.1f}% ROI")


def main():
    parser = argparse.ArgumentParser(
        description='Generate contrarian betting picks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict/generate_contrarian_picks.py --week 14
  python scripts/predict/generate_contrarian_picks.py --week 14 --validated-only
  python scripts/predict/generate_contrarian_picks.py --week 14 --high-only

VALIDATED Rules (consistent across 2023-2025):
  1. UNDER receptions when line >= 5.5: 57.1% WR, +9.0% ROI (n=804)
  2. UNDER receptions when line >= 4.5: 55.2% WR, +5.3% ROI (n=1,776)

SPECULATIVE Rules (positive overall, but lost in 2024):
  3. UNDER rushing yards when line >= 70.5: 59.5% WR, +13.6% ROI (2024: -9.8%)
  4. UNDER rushing yards when line >= 58.5: 56.0% WR, +6.9% ROI (2024: -9.4%)

REMOVED (failed large-sample validation):
  - OVER receiving yards when line <= 22.5: 50.1% WR, -4.3% ROI

EXCLUDED: Passing yards (41.8% WR, -20% ROI - catastrophic)
        """
    )
    parser.add_argument('--week', type=int, required=True, help='NFL week number')
    parser.add_argument('--min-roi', type=float, default=0.0, help='Minimum expected ROI (default: 0)')
    parser.add_argument('--high-only', action='store_true', help='Only show high confidence picks')
    parser.add_argument('--show-all', action='store_true', help='Show all picks including speculative')
    parser.add_argument('--validated-only', action='store_true', help='Only show validated picks (consistent all seasons)')
    parser.add_argument('--no-speculative', action='store_true', help='Exclude speculative (inconsistent) rules')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    args = parser.parse_args()

    print("=" * 70)
    print(f"CONTRARIAN PICKS - WEEK {args.week}")
    print("=" * 70)
    print("\nStrategy: FADE high receptions + Line-level rules")
    print("Validated on 20,816 bets (2023-2025)")
    print()

    try:
        # Load data
        df = load_props_with_predictions(args.week)
        print(f"Loaded {len(df)} props")

        # Convert to list of dicts and standardize column names
        props = df.to_dict('records')

        # Standardize column names for contrarian module
        for prop in props:
            # Map model_projection -> model_pred
            if 'model_projection' in prop and prop.get('model_projection'):
                prop['model_pred'] = prop['model_projection']
            elif 'pred_mean' in prop:
                prop['model_pred'] = prop['pred_mean']
            else:
                prop['model_pred'] = 0

            # Map stat_type -> market if needed
            if 'stat_type' in prop and 'market' not in prop:
                prop['market'] = prop['stat_type']

        # Generate picks
        min_confidence = "high" if args.high_only else "speculative"
        picks = generate_contrarian_picks(
            props,
            min_roi=args.min_roi,
            min_confidence=min_confidence,
            validated_only=args.validated_only,
            include_speculative=not args.no_speculative
        )

        # Display picks
        if not args.quiet:
            display_picks(picks, show_all=args.show_all)

        # Summary
        summary = summarize_picks(picks)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\nTotal picks: {summary['total']}")
        print(f"  High confidence: {summary['high_confidence']} (validated, ROI >= 8%)")
        print(f"  Medium confidence: {summary['medium_confidence']} (validated, ROI 3-8%)")
        print(f"  Speculative: {summary['speculative']} (not consistent across seasons)")
        print(f"  Validated (all seasons): {summary['validated_count']}")

        if summary['total'] > 0:
            print(f"\nExpected performance:")
            print(f"  Avg Win Rate: {summary['avg_wr']:.1%}")
            print(f"  Avg ROI: {summary['avg_roi']:+.1f}%")

            print(f"\nBy market:")
            for market, stats in summary['by_market'].items():
                print(f"  {market}: {stats['count']} picks ({stats['over']} OVER, {stats['under']} UNDER)")

            print(f"\nBy rule:")
            for rule, count in summary['by_rule'].items():
                print(f"  {rule}: {count} picks")

        # Save to file
        output_path = args.output or f'data/picks/contrarian_picks_week{args.week}.csv'
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if picks:
            picks_df = pd.DataFrame([p.to_dict() for p in picks])
            picks_df.to_csv(output_path, index=False)
            print(f"\nSaved {len(picks)} picks to: {output_path}")
        else:
            print("\nNo picks to save.")

        print("\n" + "=" * 70)
        print("Remember: These are CONTRARIAN picks - betting AGAINST the model!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run the prediction pipeline first:")
        print(f"  python scripts/predict/generate_unified_recommendations_v3.py --week {args.week}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error generating picks: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
