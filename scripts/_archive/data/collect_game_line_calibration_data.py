#!/usr/bin/env python3
"""
Collect Historical Game Line Predictions + Outcomes for Calibration Analysis
================================================================================

This script:
1. Loads historical game simulation results
2. Matches to actual game outcomes (from nflverse)
3. Collects predictions vs outcomes for spreads, totals, moneylines
4. Prepares data for calibration analysis
"""

import sys
from pathlib import Path
import pandas as pd
import json
import glob
from typing import Dict, List
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.team_names import normalize_team_name
from nfl_quant.utils.season_utils import get_current_season

def load_simulation_results(week: int, season: int = None) -> List[Dict]:
    """Load all simulation results for a given week."""
    if season is None:
        season = get_current_season()
    sim_files = glob.glob(f'reports/sim_{season}_{week:02d}_*.json')
    results = []

    for sim_file in sim_files:
        try:
            with open(sim_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {sim_file}: {e}")

    return results

def get_actual_game_outcomes(season: int = None) -> pd.DataFrame:
    """Load actual game outcomes from nflverse."""
    if season is None:
        season = get_current_season()

    try:
        from nfl_quant.utils.nflverse_loader import load_schedules as nfl_load_schedules
        NFLVERSE_AVAILABLE = True
    except ImportError:
        NFLVERSE_AVAILABLE = False

    if not NFLVERSE_AVAILABLE:
        return pd.DataFrame()

    try:
        schedules = nfl_load_schedules(seasons=season)

        # Filter to completed games
        if 'game_status' in schedules.columns:
            schedules = schedules[schedules['game_status'] == 'COMP']
        elif 'status' in schedules.columns:
            schedules = schedules[schedules['status'] == 'COMP']

        return schedules
    except Exception as e:
        print(f"⚠️  Error loading schedules: {e}")
        return pd.DataFrame()

def match_simulation_to_outcome(sim_data: Dict, outcomes_df: pd.DataFrame) -> Dict:
    """Match simulation prediction to actual game outcome."""
    game_id = sim_data.get('game_id', '')
    if not game_id:
        return None

    # Parse game_id: 2024_01_AWAY_HOME
    parts = game_id.split('_')
    if len(parts) < 4:
        return None

    season = int(parts[0])
    week = int(parts[1])
    away_team = normalize_team_name(parts[2])
    home_team = normalize_team_name(parts[3])

    # Extract predictions
    result = {
        'game_id': game_id,
        'season': season,
        'week': week,
        'away_team': away_team,
        'home_team': home_team,
        'home_win_prob': sim_data.get('home_win_prob', 0.5),
        'away_win_prob': sim_data.get('away_win_prob', 0.5),
        'fair_spread': sim_data.get('fair_spread', 0.0),
        'fair_total': sim_data.get('fair_total', 45.0),
        'home_actual_win': None,  # Will be filled if outcome found
        'away_score': None,
        'home_score': None,
    }

    # Match to actual outcome if available
    if not outcomes_df.empty:
        # Try to match by week and team abbreviations
        week_matches = outcomes_df[outcomes_df['week'] == week] if 'week' in outcomes_df.columns else outcomes_df

        for _, outcome_row in week_matches.iterrows():
            # Normalize team names from outcome data
            outcome_away = normalize_team_name(outcome_row.get('away_team', ''))
            outcome_home = normalize_team_name(outcome_row.get('home_team', ''))

            if (outcome_away == away_team and outcome_home == home_team) or \
               (outcome_away == home_team and outcome_home == away_team):  # Handle swapped

                # Get actual scores
                away_score = outcome_row.get('away_score') or outcome_row.get('away_team_score')
                home_score = outcome_row.get('home_score') or outcome_row.get('home_team_score')

                if pd.notna(away_score) and pd.notna(home_score):
                    result['away_score'] = float(away_score)
                    result['home_score'] = float(home_score)
                    result['home_actual_win'] = 1 if home_score > away_score else 0
                    break

    return result

def collect_game_line_calibration_data(weeks: List[int], season: int = None) -> pd.DataFrame:
    """Collect all game line predictions and prepare for calibration analysis."""
    if season is None:
        season = get_current_season()

    print("=" * 80)
    print("COLLECTING GAME LINE CALIBRATION DATA")
    print("=" * 80)
    print()

    # Load actual outcomes
    print("Loading actual game outcomes...")
    outcomes_df = get_actual_game_outcomes(season)
    if outcomes_df.empty:
        print("⚠️  No outcomes available - will collect simulation data only")
    else:
        print(f"✅ Loaded {len(outcomes_df)} completed games")

    print()

    # Collect simulation data
    all_data = []
    for week in weeks:
        print(f"Week {week}:")
        sim_results = load_simulation_results(week, season)
        print(f"  Found {len(sim_results)} simulation files")

        for sim_data in sim_results:
            matched = match_simulation_to_outcome(sim_data, outcomes_df)
            if matched:
                all_data.append(matched)

        if all_data:
            print(f"  ✅ Collected {len([d for d in all_data if d.get('season') == season and d.get('week') == week])} games for week {week}")

    if not all_data:
        print("⚠️  No data collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print()
    print(f"✅ Collected {len(df)} game predictions")
    print(f"   Weeks: {weeks}")
    print(f"   Games: {len(df)} unique games")

    # Report outcome matching
    if 'home_actual_win' in df.columns:
        matched = df['home_actual_win'].notna().sum()
        print(f"   Matched outcomes: {matched}/{len(df)} ({matched/len(df)*100:.1f}%)")

    return df

def main():
    # Collect data for weeks 1-8 (completed weeks)
    weeks = list(range(1, 9))
    df = collect_game_line_calibration_data(weeks)  # season defaults to get_current_season()

    if df.empty:
        print("\n❌ No data collected - cannot proceed")
        return

    # Save to CSV for analysis
    output_path = Path('reports/game_line_calibration_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/analyze/analyze_game_line_calibration.py")
    print("  2. Review calibration quality metrics")
    print("  3. Train calibrator if needed: python scripts/train/train_game_line_calibrator.py")

if __name__ == '__main__':
    main()
