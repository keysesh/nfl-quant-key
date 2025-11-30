"""
Regime Detection Integration Script

Integrates regime detection with existing feature engineering and prediction pipeline.

This script demonstrates how to:
1. Detect regimes for all teams
2. Calculate regime-adjusted trailing stats for players
3. Feed regime-adjusted features into existing models
4. Generate regime-aware predictions and recommendations

Usage:
    python scripts/regime/integrate_regime_detection.py --week 9 --season 2025
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.regime import (
    RegimeDetector,
    RegimeAwareProjector,
    RegimeReportGenerator,
)
from nfl_quant.data.fetcher import DataFetcher
from nfl_quant.features.player_features import derive_player_features


def load_player_stats(season: int, max_week: int) -> pd.DataFrame:
    """Load player stats up to specified week."""
    stats_dfs = []

    for week in range(1, max_week):
        stats_file = Path(f"data/sleeper_stats/stats_week{week}_{season}.csv")
        if stats_file.exists():
            df = pd.read_csv(stats_file)
            df["week"] = week
            df["season"] = season
            stats_dfs.append(df)
        else:
            print(f"Warning: Missing stats file for week {week}")

    if stats_dfs:
        return pd.concat(stats_dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"No player stats found for season {season}")


def integrate_regime_detection(week: int, season: int, output_dir: str = "data/regime"):
    """
    Main integration function.

    Args:
        week: Current week to project
        season: Season year
        output_dir: Directory for regime outputs
    """
    print(f"\n{'='*60}")
    print(f"REGIME DETECTION INTEGRATION - Week {week}, {season}")
    print(f"{'='*60}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("Initializing regime detection system...")
    detector = RegimeDetector()
    projector = RegimeAwareProjector(detector=detector)
    reporter = RegimeReportGenerator()
    fetcher = DataFetcher()

    # Load data
    print("\nLoading data...")
    print("  - Play-by-play data...")
    try:
        pbp_df = fetcher.load_pbp_parquet(season)
        print(f"    ✓ Loaded {len(pbp_df)} plays")
    except Exception as e:
        print(f"    ✗ Error loading PBP: {e}")
        return

    print("  - Player stats...")
    try:
        player_stats_df = load_player_stats(season, week)
        print(f"    ✓ Loaded {len(player_stats_df)} player-week records")
    except Exception as e:
        print(f"    ✗ Error loading stats: {e}")
        return

    # Get unique teams
    teams = sorted(pbp_df["posteam"].dropna().unique())
    print(f"\nDetecting regimes for {len(teams)} teams...")

    # Detect regimes for all teams
    all_detections = {}
    regime_summary = []

    for team in teams:
        try:
            result = detector.detect_all_regimes(
                team=team,
                current_week=week,
                season=season,
                pbp_df=pbp_df,
                player_stats_df=player_stats_df,
            )

            all_detections[team] = result

            if result.has_active_regime:
                regime = result.active_regime
                regime_summary.append({
                    "team": team,
                    "regime_type": regime.trigger.type.value,
                    "description": regime.trigger.description,
                    "start_week": regime.start_week,
                    "games_in_regime": regime.games_in_regime,
                    "confidence": regime.trigger.confidence,
                    "affected_players_count": len(regime.affected_players),
                })

                print(f"  {team}: {regime.trigger.description} (Week {regime.start_week}, {regime.games_in_regime} games)")

        except Exception as e:
            print(f"  {team}: Error - {e}")

    print(f"\n✓ Detected {len(regime_summary)} active regimes")

    # Export regime summary
    regime_summary_df = pd.DataFrame(regime_summary)
    regime_summary_path = output_path / f"regime_summary_week{week}.csv"
    regime_summary_df.to_csv(regime_summary_path, index=False)
    print(f"✓ Regime summary saved to {regime_summary_path}")

    # Export full detection results (JSON)
    detections_json_path = output_path / f"regime_detections_week{week}.json"
    reporter.export_to_json(all_detections, str(detections_json_path))
    print(f"✓ Full detections saved to {detections_json_path}")

    # Get all players with stats
    players_with_stats = player_stats_df[
        player_stats_df["week"] == week - 1
    ][["player_name", "player_id", "position", "team"]].drop_duplicates()

    print(f"\nCalculating regime-adjusted stats for {len(players_with_stats)} players...")

    # Batch process players to get regime-adjusted trailing stats
    regime_adjusted_stats = projector.batch_process_players(
        players=players_with_stats.to_dict("records"),
        current_week=week,
        season=season,
        pbp_df=pbp_df,
        player_stats_df=player_stats_df,
    )

    print(f"✓ Processed {len(regime_adjusted_stats)} players")

    # Export regime-adjusted stats
    regime_stats_path = output_path / f"regime_adjusted_stats_week{week}.csv"
    regime_adjusted_stats.to_csv(regime_stats_path, index=False)
    print(f"✓ Regime-adjusted stats saved to {regime_stats_path}")

    # Compare with standard trailing stats
    print("\n" + "="*60)
    print("REGIME IMPACT ANALYSIS")
    print("="*60)

    # Calculate standard 4-week trailing stats for comparison
    standard_stats = []

    for _, player in players_with_stats.iterrows():
        player_name = player["player_name"]
        player_data = player_stats_df[
            (player_stats_df["player_name"] == player_name)
            & (player_stats_df["week"] >= week - 4)
            & (player_stats_df["week"] < week)
        ]

        if len(player_data) == 0:
            continue

        games_played = len(player_data)

        standard_stats.append({
            "player_name": player_name,
            "standard_targets_per_game": player_data["targets"].sum() / games_played if "targets" in player_data.columns else 0,
            "standard_carries_per_game": player_data["rush_attempts"].sum() / games_played if "rush_attempts" in player_data.columns else 0,
        })

    standard_stats_df = pd.DataFrame(standard_stats)

    # Merge and compare
    comparison = regime_adjusted_stats.merge(
        standard_stats_df,
        on="player_name",
        how="inner",
    )

    # Calculate deltas
    comparison["targets_delta"] = (
        comparison["targets_per_game"] - comparison["standard_targets_per_game"]
    )
    comparison["carries_delta"] = (
        comparison["carries_per_game"] - comparison["standard_carries_per_game"]
    )

    # Find players with significant regime impact
    significant_changes = comparison[
        (abs(comparison["targets_delta"]) > 1.5) |
        (abs(comparison["carries_delta"]) > 3.0)
    ].sort_values("targets_delta", ascending=False)

    print(f"\nPlayers with significant regime impact ({len(significant_changes)}):")
    print("=" * 80)

    for _, row in significant_changes.head(20).iterrows():
        player_name = row["player_name"]
        position = row["position"]
        team = row["team"]

        if abs(row["targets_delta"]) > 1.5:
            direction = "↑" if row["targets_delta"] > 0 else "↓"
            print(f"{player_name} ({team} {position})")
            print(f"  Targets: {row['standard_targets_per_game']:.1f} → {row['targets_per_game']:.1f} ({direction} {abs(row['targets_delta']):.1f})")

            if row.get("regime_detected"):
                print(f"  Regime: {row.get('regime_type', 'unknown')} (Week {row.get('regime_start_week')})")

            print()

    # Export comparison
    comparison_path = output_path / f"regime_vs_standard_week{week}.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"✓ Comparison saved to {comparison_path}")

    # Integration guide
    print("\n" + "="*60)
    print("INTEGRATION GUIDE")
    print("="*60)
    print("""
To integrate regime-adjusted stats into your prediction pipeline:

1. REPLACE standard trailing stats calculation:

   OLD (in generate_model_predictions.py):
   ```
   trailing_stats = calculate_trailing_4_week_avg(player_stats_df)
   ```

   NEW:
   ```
   from nfl_quant.regime import RegimeAwareProjector

   projector = RegimeAwareProjector()
   regime_stats = projector.batch_process_players(
       players=player_list,
       current_week=week,
       season=season,
       pbp_df=pbp_df,
       player_stats_df=player_stats_df,
   )
   ```

2. USE regime-adjusted stats as model inputs:

   ```
   model_input = {
       'snaps_per_game': regime_stats['snaps_per_game'],
       'targets_per_game': regime_stats['targets_per_game'],
       'carries_per_game': regime_stats['carries_per_game'],
       ...
   }

   prediction = usage_model.predict(model_input)
   ```

3. ADD regime flags to recommendations:

   ```
   if regime_stats['regime_detected']:
       recommendation['regime_flag'] = regime_stats['regime_type']
       recommendation['regime_confidence'] = regime_stats['regime_confidence']
       recommendation['regime_weeks'] = regime_stats['weeks_in_regime']
   ```

4. FILTER low-confidence regime bets (optional):

   ```
   if regime_stats.get('insufficient_sample_warning'):
       # Reduce bet size or skip
       bet_size *= 0.5
   ```

Output files generated:
- {regime_summary_path}
- {detections_json_path}
- {regime_stats_path}
- {comparison_path}

Use these to:
- Identify high-impact regime changes
- Adjust player projections dynamically
- Generate regime-aware betting recommendations
""")

    print("\n" + "="*60)
    print("REGIME DETECTION COMPLETE")
    print("="*60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Integrate regime detection with prediction pipeline"
    )
    parser.add_argument(
        "--week", "-w",
        type=int,
        required=True,
        help="Current week to project"
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        required=True,
        help="Season year"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/regime",
        help="Output directory for regime data"
    )

    args = parser.parse_args()

    try:
        integrate_regime_detection(
            week=args.week,
            season=args.season,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
