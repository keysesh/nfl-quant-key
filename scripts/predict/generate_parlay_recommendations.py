#!/usr/bin/env python3
"""
Generate Parlay Recommendations
================================

Reads recommendations (v3 unified or edge) and generates optimal cross-game parlays
using correlation-adjusted probabilities and Kelly criterion sizing.

Supports:
- V3 unified recommendations (CURRENT_WEEK_RECOMMENDATIONS.csv) - DEFAULT
- Edge recommendations (edge_recommendations_weekX_YYYY.csv) - FALLBACK

Usage:
    python scripts/predict/generate_parlay_recommendations.py --week 15
    python scripts/predict/generate_parlay_recommendations.py --week 15 --num-parlays 20
    python scripts/predict/generate_parlay_recommendations.py --week 15 --source edge  # Use edge only
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.parlay.recommendation import ParlayRecommender, SingleBet, ParlayRecommendation


def load_v3_recommendations(week: int) -> pd.DataFrame:
    """Load V3 unified recommendations from CURRENT_WEEK_RECOMMENDATIONS.csv.

    Args:
        week: NFL week number (for validation)

    Returns:
        DataFrame of V3 recommendations filtered to the specified week
    """
    path = project_root / "reports" / "CURRENT_WEEK_RECOMMENDATIONS.csv"

    if not path.exists():
        print(f"  V3 recommendations not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Filter to specified week if week column exists
    if 'week' in df.columns:
        df = df[df['week'] == week]

    if len(df) == 0:
        print(f"  No V3 recommendations found for week {week}")
        return pd.DataFrame()

    print(f"  Loaded {len(df)} V3 recommendations for week {week}")
    return df


def load_edge_recommendations(week: int, season: int = 2025) -> pd.DataFrame:
    """Load edge recommendations from CSV.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        DataFrame of edge recommendations
    """
    path = project_root / "reports" / f"edge_recommendations_week{week}_{season}.csv"

    if not path.exists():
        print(f"  Edge recommendations not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} edge recommendations from {path}")
    return df


def load_recommendations(week: int, season: int = 2025, source: str = 'auto') -> Tuple[pd.DataFrame, str]:
    """Load recommendations from best available source.

    Args:
        week: NFL week number
        season: NFL season year
        source: 'auto' (try v3 first), 'v3' (v3 only), 'edge' (edge only)

    Returns:
        Tuple of (DataFrame, source_name)
    """
    if source == 'v3':
        df = load_v3_recommendations(week)
        return (df, 'v3') if not df.empty else (pd.DataFrame(), 'none')

    if source == 'edge':
        df = load_edge_recommendations(week, season)
        return (df, 'edge') if not df.empty else (pd.DataFrame(), 'none')

    # Auto mode: try v3 first, then edge
    df = load_v3_recommendations(week)
    if not df.empty:
        return (df, 'v3')

    df = load_edge_recommendations(week, season)
    if not df.empty:
        return (df, 'edge')

    return (pd.DataFrame(), 'none')


def convert_v3_to_single_bets(df: pd.DataFrame) -> List[SingleBet]:
    """Convert V3 unified recommendations DataFrame to SingleBet objects.

    Args:
        df: V3 recommendations DataFrame (CURRENT_WEEK_RECOMMENDATIONS.csv format)

    Returns:
        List of SingleBet objects
    """
    single_bets = []

    for _, row in df.iterrows():
        # Extract fields from V3 format
        player = row.get('player', '')
        team = row.get('team', '')
        market = row.get('market', '')
        line = row.get('line', 0)
        pick = row.get('pick', 'UNDER')  # V3 uses 'pick' column
        direction = pick.upper() if isinstance(pick, str) else 'UNDER'

        # V3 confidence fields
        confidence = row.get('calibrated_prob', row.get('model_prob', 0.55))
        if pd.isna(confidence):
            confidence = 0.55

        # V3 uses kelly_units
        units = row.get('kelly_units', 1.0)
        if pd.isna(units):
            units = 1.0

        # V3 edge
        edge_pct = row.get('edge_pct', 0.05)
        if pd.isna(edge_pct):
            edge_pct = 0.05

        # Get game info
        game = row.get('game', '')
        if not game or pd.isna(game):
            opponent = row.get('opponent', '')
            if opponent and team:
                game = f"{team}v{opponent}"
            elif team:
                game = f"GAME_{team}"

        # Create bet name
        name = f"{player} {direction} {line}"

        # Get odds from V3
        odds = row.get('odds', -110)
        if pd.isna(odds):
            odds = -110

        single_bet = SingleBet(
            name=name,
            bet_type=f"Player Prop {direction}",
            game=game,
            team=team,
            player=player,
            market=market,
            odds=int(odds),
            our_prob=float(confidence),
            market_prob=row.get('market_prob', None),
            edge=float(edge_pct) if edge_pct else 0.05,
            bet_size=float(units) if units else 1.0,
            potential_profit=None
        )
        single_bets.append(single_bet)

    return single_bets


def convert_edge_to_single_bets(df: pd.DataFrame) -> List[SingleBet]:
    """Convert edge recommendations DataFrame to SingleBet objects.

    Args:
        df: Edge recommendations DataFrame

    Returns:
        List of SingleBet objects
    """
    single_bets = []

    for _, row in df.iterrows():
        # Extract fields with fallbacks
        player = row.get('player', '')
        team = row.get('team', '')
        market = row.get('market', '')
        line = row.get('line', 0)
        direction = row.get('direction', 'UNDER')
        units = row.get('units', 1.0)
        confidence = row.get('combined_confidence', 0.55)
        source = row.get('source', 'EDGE')

        # Get game info - try multiple sources
        game = row.get('game', '')
        if not game or pd.isna(game):
            opponent = row.get('opponent', '')
            home_team = row.get('home_team', '')
            away_team = row.get('away_team', '')
            if home_team and away_team:
                game = f"{away_team}@{home_team}"
            elif opponent and team:
                # Use team vs opponent as game identifier
                game = f"{team}v{opponent}"
            elif team:
                # Last resort: use team as game identifier
                # This allows cross-team parlays (different teams = different games)
                game = f"GAME_{team}"

        # Create bet name
        name = f"{player} {direction} {line}"

        # Default odds if not provided
        odds = row.get('odds', -110)
        if pd.isna(odds):
            odds = -110

        single_bet = SingleBet(
            name=name,
            bet_type=f"Player Prop {direction}",
            game=game,
            team=team,
            player=player,
            market=market,
            odds=int(odds),
            our_prob=float(confidence) if not pd.isna(confidence) else 0.55,
            market_prob=None,
            edge=float(units) * 0.05 if units else 0.05,  # Approximate edge from units
            bet_size=float(units) if units else 1.0,
            potential_profit=None
        )
        single_bets.append(single_bet)

    return single_bets


def convert_to_single_bets(df: pd.DataFrame, source: str) -> List[SingleBet]:
    """Convert recommendations DataFrame to SingleBet objects.

    Args:
        df: Recommendations DataFrame
        source: 'v3' or 'edge' to determine format

    Returns:
        List of SingleBet objects
    """
    if source == 'v3':
        return convert_v3_to_single_bets(df)
    else:
        return convert_edge_to_single_bets(df)


def save_parlay_recommendations(
    parlays: List[ParlayRecommendation],
    recommender: ParlayRecommender,
    week: int,
    season: int = 2025,
    num_featured: int = 3
) -> Path:
    """Save parlay recommendations to CSV.

    Args:
        parlays: List of parlay recommendations
        recommender: Recommender instance (for dict conversion)
        week: NFL week
        season: NFL season
        num_featured: Number of featured parlays

    Returns:
        Path to saved file
    """
    output_path = project_root / "reports" / f"parlay_recommendations_week{week}_{season}.csv"

    if not parlays:
        print("No parlay recommendations to save.")
        return output_path

    # Convert to dict list
    dict_list = recommender.to_dict_list(parlays)

    # Create DataFrame
    df = pd.DataFrame(dict_list)

    # Add featured column
    df['featured'] = df['rank'].apply(lambda x: 'YES' if x <= num_featured else 'NO')

    # Reorder columns to put featured first
    cols = ['rank', 'featured'] + [c for c in df.columns if c not in ['rank', 'featured']]
    df = df[cols]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(parlays)} parlay recommendations ({num_featured} featured) to {output_path}")

    return output_path


def print_summary(parlays: List[ParlayRecommendation], recommender: ParlayRecommender, num_featured: int = 3):
    """Print summary of parlay recommendations.

    Args:
        parlays: List of parlay recommendations
        recommender: Recommender instance
        num_featured: Number of featured parlays to highlight
    """
    if not parlays:
        print("\nNo parlay recommendations generated.")
        return

    print("\n" + "=" * 70)
    print("PARLAY RECOMMENDATIONS SUMMARY")
    print("=" * 70)

    # Count by leg count
    by_legs = {}
    for p in parlays:
        n = len(p.legs)
        by_legs[n] = by_legs.get(n, 0) + 1

    print(f"\nTotal parlays: {len(parlays)}")
    print(f"Featured: {min(num_featured, len(parlays))}")
    for n_legs in sorted(by_legs.keys()):
        print(f"  {n_legs}-leg: {by_legs[n_legs]}")

    # Show featured parlays
    featured = parlays[:num_featured]
    print("\n" + "=" * 70)
    print(f"★ FEATURED PARLAYS (Top {len(featured)}) ★")
    print("=" * 70)

    for i, parlay in enumerate(featured, 1):
        print(f"\n★ #{i} ({len(parlay.legs)}-leg, {parlay.true_odds:+d})")
        for j, leg in enumerate(parlay.legs, 1):
            print(f"   Leg {j}: {leg.name}")
            print(f"          {leg.game}")
        print(f"   Edge: {parlay.edge:.1%} | Win Prob: {parlay.adjusted_prob:.1%}")
        print(f"   Recommended: {parlay.recommended_units:.2f}u")

    # Show remaining parlays (condensed)
    remaining = parlays[num_featured:]
    if remaining:
        print("\n" + "-" * 70)
        print(f"ADDITIONAL PARLAYS ({len(remaining)})")
        print("-" * 70)

        for i, parlay in enumerate(remaining, num_featured + 1):
            legs_short = " + ".join([f"{leg.player} {leg.name.split()[-2]} {leg.name.split()[-1]}"
                                      for leg in parlay.legs])
            print(f"#{i}: {legs_short}")
            print(f"    {parlay.true_odds:+d} | Edge: {parlay.edge:.1%} | Prob: {parlay.adjusted_prob:.1%}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate parlay recommendations from edge recommendations"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        required=True,
        help="NFL week number"
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        default=2025,
        help="NFL season (default: 2025)"
    )
    parser.add_argument(
        "--num-parlays", "-n",
        type=int,
        default=10,
        help="Number of parlays to generate (default: 10)"
    )
    parser.add_argument(
        "--num-featured",
        type=int,
        default=3,
        help="Number of featured (top) parlays to highlight (default: 3)"
    )
    parser.add_argument(
        "--max-legs", "-l",
        type=int,
        default=4,
        help="Maximum legs per parlay (default: 4)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum confidence per leg (default: 0.55)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Bankroll for Kelly sizing (default: 1000)"
    )
    parser.add_argument(
        "--max-leg-reuse",
        type=int,
        default=3,
        help="Max times a leg can appear across parlays (default: 3)"
    )
    parser.add_argument(
        "--no-diversify",
        action="store_true",
        help="Disable diversity constraints (show all top parlays by EV)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "v3", "edge"],
        help="Recommendation source: auto (v3 first, then edge), v3, or edge (default: auto)"
    )

    args = parser.parse_args()

    diversify = not args.no_diversify

    print(f"\n{'=' * 70}")
    print(f"PARLAY RECOMMENDATION GENERATOR")
    print(f"Week {args.week} | Season {args.season}")
    print(f"{'=' * 70}")
    print(f"\nSettings:")
    print(f"  Parlays: {args.num_parlays} total, {args.num_featured} featured")
    print(f"  Max legs: {args.max_legs}")
    print(f"  Min confidence: {args.min_confidence:.0%}")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print(f"  Mode: Cross-game only (no SGP)")
    print(f"  Diversification: {'ON (max ' + str(args.max_leg_reuse) + 'x leg reuse)' if diversify else 'OFF'}")
    print(f"  Source: {args.source}")

    # Load recommendations (v3 or edge)
    print(f"\n[1/4] Loading recommendations (source: {args.source})...")
    recs_df, source_used = load_recommendations(args.week, args.season, args.source)

    if recs_df.empty or source_used == 'none':
        print("No recommendations found. Exiting.")
        print("  Run either:")
        print("    - python scripts/predict/generate_unified_recommendations_v3.py --week <WEEK>")
        print("    - python scripts/predict/generate_edge_recommendations.py --week <WEEK>")
        return 1

    print(f"  Using {source_used.upper()} recommendations")

    # Convert to SingleBet objects
    print("[2/4] Converting to bet objects...")
    single_bets = convert_to_single_bets(recs_df, source_used)
    print(f"  Converted {len(single_bets)} bets")

    # Filter by minimum confidence
    qualified_bets = [b for b in single_bets if b.our_prob and b.our_prob >= args.min_confidence]
    print(f"  {len(qualified_bets)} bets meet minimum confidence threshold")

    if len(qualified_bets) < 2:
        print("Not enough qualified bets for parlays. Need at least 2.")
        return 1

    # Initialize recommender
    print("[3/4] Generating parlay combinations...")
    recommender = ParlayRecommender(
        max_legs=args.max_legs,
        min_confidence=args.min_confidence,
        use_empirical_correlations=True,
        cross_game_only=True,
        bankroll=args.bankroll
    )

    # Generate parlays
    parlays = recommender.generate_parlays(
        single_bets=qualified_bets,
        num_parlays=args.num_parlays,
        max_leg_reuse=args.max_leg_reuse,
        diversify=diversify
    )

    print(f"  Generated {len(parlays)} valid parlays")

    # Filter to positive edge only
    positive_edge_parlays = [p for p in parlays if p.edge > 0]
    print(f"  {len(positive_edge_parlays)} parlays with positive edge")

    # Save recommendations
    print("[4/4] Saving recommendations...")
    save_parlay_recommendations(positive_edge_parlays, recommender, args.week, args.season, args.num_featured)

    # Print summary
    print_summary(positive_edge_parlays, recommender, args.num_featured)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
