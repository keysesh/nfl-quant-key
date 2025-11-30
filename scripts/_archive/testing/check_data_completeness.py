#!/usr/bin/env python3
"""
Data Completeness Checker
==========================

Checks for all required data files and identifies gaps:
- Required data files for pipeline
- Historical vs current season data
- Missing weeks or gaps
- Data fetching recommendations

Usage:
    python scripts/testing/check_data_completeness.py [--week WEEK] [--season SEASON]

Example:
    python scripts/testing/check_data_completeness.py --week 10 --season 2025
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DataCompletenessChecker:
    """Check data completeness and identify gaps."""

    def __init__(self, week: Optional[int] = None, season: int = 2025):
        self.week = week
        self.season = season
        self.base_dir = Path.cwd()
        self.gaps = []
        self.recommendations = []

    def check_sleeper_stats(self) -> Dict:
        """Check Sleeper stats coverage."""
        print("="*80)
        print("SLEEPER STATS CHECK")
        print("="*80)

        stats_dir = self.base_dir / "data/sleeper_stats"
        stats_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for season in [2024, 2025]:
            files = sorted(stats_dir.glob(f"stats_week*_{season}.csv"))
            weeks = sorted([int(f.stem.split('_')[1].replace('week', '')) for f in files])

            results[season] = {
                'files': len(files),
                'weeks': weeks,
                'coverage': f"Week {min(weeks)} to Week {max(weeks)}" if weeks else "No data"
            }

            print(f"\n{season} Season:")
            print(f"  Files: {len(files)}")
            if weeks:
                print(f"  Weeks: {weeks}")
                print(f"  Coverage: Week {min(weeks)} to Week {max(weeks)}")

                # Check for gaps
                if len(weeks) > 1:
                    expected_weeks = set(range(min(weeks), max(weeks) + 1))
                    missing_weeks = sorted(expected_weeks - set(weeks))
                    if missing_weeks:
                        print(f"  ⚠️  Missing weeks: {missing_weeks}")
                        self.gaps.append({
                            'type': 'sleeper_stats',
                            'season': season,
                            'missing_weeks': missing_weeks
                        })
            else:
                print(f"  ❌ No data found")
                self.gaps.append({
                    'type': 'sleeper_stats',
                    'season': season,
                    'missing_weeks': 'all'
                })

        return results

    def check_nflverse_data(self) -> Dict:
        """Check NFLverse data coverage."""
        print("\n" + "="*80)
        print("NFLVERSE DATA CHECK")
        print("="*80)

        nflverse_dir = self.base_dir / "data/nflverse"
        results = {}

        # Check weekly stats
        print("\nWeekly Stats:")
        for season in [2024, 2025]:
            weekly_file = nflverse_dir / f"weekly_{season}.parquet"
            if weekly_file.exists():
                try:
                    df = pd.read_parquet(weekly_file)
                    weeks = sorted(df['week'].unique().tolist())
                    results[f'weekly_{season}'] = {
                        'exists': True,
                        'weeks': weeks,
                        'player_weeks': len(df)
                    }
                    print(f"  {season}: ✅ {len(df):,} player-weeks, weeks {weeks[0]}-{weeks[-1]}")
                except Exception as e:
                    results[f'weekly_{season}'] = {'exists': True, 'error': str(e)}
                    print(f"  {season}: ⚠️  File exists but error reading: {e}")
            else:
                results[f'weekly_{season}'] = {'exists': False}
                print(f"  {season}: ❌ MISSING")
                if season == 2025:
                    self.gaps.append({
                        'type': 'nflverse_weekly',
                        'season': season,
                        'note': 'NFLverse weekly stats lag by ~1 season (expected)'
                    })

        # Check play-by-play data
        print("\nPlay-by-Play Data:")
        for season in [2024, 2025]:
            pbp_file = nflverse_dir / f"pbp_{season}.parquet"
            if pbp_file.exists():
                try:
                    df = pd.read_parquet(pbp_file)
                    weeks = sorted(df['week'].unique().tolist()) if 'week' in df.columns else []
                    results[f'pbp_{season}'] = {
                        'exists': True,
                        'plays': len(df),
                        'weeks': weeks
                    }
                    print(f"  {season}: ✅ {len(df):,} plays" + (f", weeks {weeks[0]}-{weeks[-1]}" if weeks else ""))
                except Exception as e:
                    results[f'pbp_{season}'] = {'exists': True, 'error': str(e)}
                    print(f"  {season}: ⚠️  File exists but error reading: {e}")
            else:
                results[f'pbp_{season}'] = {'exists': False}
                print(f"  {season}: ❌ MISSING")
                self.gaps.append({
                    'type': 'nflverse_pbp',
                    'season': season,
                    'action': f'Run: python scripts/fetch/fetch_nflverse_data.py'
                })

        return results

    def check_odds_data(self) -> Dict:
        """Check odds and prop data."""
        print("\n" + "="*80)
        print("ODDS & PROP DATA CHECK")
        print("="*80)

        results = {}

        # Check game odds
        if self.week:
            odds_file = self.base_dir / f"data/odds_week{self.week}_draftkings.csv"
            if odds_file.exists():
                df = pd.read_csv(odds_file)
                results['game_odds'] = {'exists': True, 'games': len(df)}
                print(f"\nGame Odds (Week {self.week}): ✅ {len(df)} games")
            else:
                results['game_odds'] = {'exists': False}
                print(f"\nGame Odds (Week {self.week}): ❌ MISSING")
                self.gaps.append({
                    'type': 'game_odds',
                    'week': self.week,
                    'action': f'Run: python scripts/fetch/fetch_live_odds.py {self.week}'
                })

        # Check player props
        props_file = self.base_dir / "data/nfl_player_props_draftkings.csv"
        if props_file.exists():
            df = pd.read_csv(props_file)
            results['player_props'] = {'exists': True, 'props': len(df)}
            print(f"Player Props: ✅ {len(df)} props")
        else:
            results['player_props'] = {'exists': False}
            print(f"Player Props: ❌ MISSING")
            self.gaps.append({
                'type': 'player_props',
                'action': 'Run: python scripts/fetch/fetch_nfl_player_props.py'
            })

        # Check historical props
        backfill_dir = self.base_dir / "data/historical/backfill"
        if backfill_dir.exists():
            files_2024 = list(backfill_dir.glob("player_props_history_2024*.csv"))
            files_2025 = list(backfill_dir.glob("player_props_history_2025*.csv"))

            results['historical_props'] = {
                '2024': len(files_2024),
                '2025': len(files_2025)
            }
            print(f"\nHistorical Props:")
            print(f"  2024: {len(files_2024)} files")
            print(f"  2025: {len(files_2025)} files")

        return results

    def check_models(self) -> Dict:
        """Check model files."""
        print("\n" + "="*80)
        print("MODEL FILES CHECK")
        print("="*80)

        models_dir = self.base_dir / "data/models"
        results = {}

        required_models = {
            'Usage Predictor': 'usage_predictor_v4_defense.joblib',
            'Efficiency Predictor': 'efficiency_predictor_v2_defense.joblib',
            'TD Calibrator (Improved)': 'td_calibrator_v2_improved.joblib',
        }

        for name, filename in required_models.items():
            model_path = models_dir / filename
            exists = model_path.exists()
            results[name] = {'exists': exists, 'path': str(model_path)}

            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {'Found' if exists else 'MISSING'}")

            if not exists:
                self.gaps.append({
                    'type': 'model',
                    'name': name,
                    'file': filename
                })

        return results

    def check_calibrators(self) -> Dict:
        """Check calibrator files."""
        print("\n" + "="*80)
        print("CALIBRATOR FILES CHECK")
        print("="*80)

        configs_dir = self.base_dir / "configs"
        results = {}

        market_calibrators = [
            'calibrator_player_reception_yds.json',
            'calibrator_player_rush_yds.json',
            'calibrator_player_receptions.json',
            'calibrator_player_pass_yds.json',
        ]

        print("\nMarket-Specific Calibrators:")
        for filename in market_calibrators:
            cal_path = configs_dir / filename
            exists = cal_path.exists()
            results[filename] = {'exists': exists}

            status = "✅" if exists else "❌"
            print(f"  {status} {filename}")

            if not exists:
                self.gaps.append({
                    'type': 'calibrator',
                    'file': filename,
                    'action': 'Run: python scripts/integration/convert_improved_calibrators_to_json.py'
                })

        return results

    def generate_recommendations(self):
        """Generate data fetching recommendations."""
        print("\n" + "="*80)
        print("DATA GAPS & RECOMMENDATIONS")
        print("="*80)

        if not self.gaps:
            print("\n✅ No data gaps found! All required data is present.")
            return

        print(f"\nFound {len(self.gaps)} data gaps:\n")

        for i, gap in enumerate(self.gaps, 1):
            print(f"{i}. [{gap['type'].upper()}]")
            if 'season' in gap:
                print(f"   Season: {gap['season']}")
            if 'week' in gap:
                print(f"   Week: {gap['week']}")
            if 'missing_weeks' in gap:
                print(f"   Missing weeks: {gap['missing_weeks']}")
            if 'action' in gap:
                print(f"   Action: {gap['action']}")
            if 'note' in gap:
                print(f"   Note: {gap['note']}")
            print()

        # Generate fetch commands
        print("="*80)
        print("RECOMMENDED FETCH COMMANDS")
        print("="*80)

        fetch_commands = []

        sleeper_gaps = [g for g in self.gaps if g['type'] == 'sleeper_stats']
        if sleeper_gaps:
            for gap in sleeper_gaps:
                if gap['missing_weeks'] != 'all':
                    for week in gap.get('missing_weeks', []):
                        fetch_commands.append(
                            f"python scripts/fetch/fetch_sleeper_week9_stats.py --week {week} --season {gap['season']}"
                        )

        nflverse_gaps = [g for g in self.gaps if g['type'] == 'nflverse_pbp']
        if nflverse_gaps:
            fetch_commands.append("python scripts/fetch/fetch_nflverse_data.py")

        odds_gaps = [g for g in self.gaps if g['type'] in ['game_odds', 'player_props']]
        if odds_gaps:
            if self.week:
                fetch_commands.append(f"python scripts/fetch/fetch_live_odds.py {self.week}")
            fetch_commands.append("python scripts/fetch/fetch_nfl_player_props.py")

        if fetch_commands:
            print("\nRun these commands to fill gaps:\n")
            for cmd in set(fetch_commands):
                print(f"  {cmd}")
        else:
            print("\nNo fetch commands needed (gaps are expected or require manual intervention)")

    def run_check(self) -> Dict:
        """Run complete data completeness check."""
        print("="*80)
        print("DATA COMPLETENESS CHECK")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.week:
            print(f"Week: {self.week}")
        print(f"Season: {self.season}")
        print("="*80)

        results = {
            'sleeper_stats': self.check_sleeper_stats(),
            'nflverse_data': self.check_nflverse_data(),
            'odds_data': self.check_odds_data(),
            'models': self.check_models(),
            'calibrators': self.check_calibrators(),
            'gaps': self.gaps
        }

        self.generate_recommendations()

        return results


def main():
    parser = argparse.ArgumentParser(description='Check data completeness')
    parser.add_argument('--week', type=int, help='Week number to check (optional)')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')

    args = parser.parse_args()

    checker = DataCompletenessChecker(args.week, args.season)
    results = checker.run_check()

    sys.exit(0 if len(checker.gaps) == 0 else 1)


if __name__ == '__main__':
    main()
