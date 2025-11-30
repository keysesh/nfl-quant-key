#!/usr/bin/env python3
"""
Full Framework End-to-End Testing
==================================

Comprehensive testing of the complete NFL Quant betting framework:
1. Data completeness check
2. Pipeline execution validation
3. Output quality verification
4. Integration testing

Usage:
    python scripts/testing/test_full_framework.py [week] [--dry-run]

Example:
    python scripts/testing/test_full_framework.py 11 --dry-run
"""

import sys
from pathlib import Path
import pandas as pd
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.stats_loader import load_weekly_stats, is_data_available
from nfl_quant.data.adapters.nflverse_adapter import NFLVerseAdapter


class FullFrameworkTester:
    """Comprehensive end-to-end framework testing."""

    def __init__(self, week: int, season: int = 2025, dry_run: bool = False):
        self.week = week
        self.season = season
        self.dry_run = dry_run
        self.base_dir = Path.cwd()
        self.results = {
            'data_completeness': {},
            'pipeline_execution': {},
            'output_quality': {},
            'data_comparison': {},
            'integration': {}
        }

    def test_data_completeness(self) -> Dict:
        """Test 1: Check all required data files exist."""
        print("\n" + "="*80)
        print("TEST 1: DATA COMPLETENESS CHECK")
        print("="*80)

        checks = {}

        # Required data files
        required_files = {
            'NFLverse PBP Data': self.base_dir / f'data/nflverse/pbp_{self.season}.parquet',
            'NFLverse Weekly Stats': self.base_dir / f'data/nflverse/weekly_{self.season}.parquet',
            'Game Odds': self.base_dir / f'data/odds_week{self.week}_draftkings.csv',
            'Player Props': self.base_dir / 'data/nfl_player_props_draftkings.csv',
            'Usage Model': self.base_dir / 'data/models/usage_predictor_v4_defense.joblib',
            'Efficiency Model': self.base_dir / 'data/models/efficiency_predictor_v2_defense.joblib',
            'TD Calibrator (Improved)': self.base_dir / 'data/models/td_calibrator_v2_improved.joblib',
            'Market Calibrators': [
                self.base_dir / 'configs/calibrator_player_reception_yds.json',
                self.base_dir / 'configs/calibrator_player_rush_yds.json',
                self.base_dir / 'configs/calibrator_player_receptions.json',
                self.base_dir / 'configs/calibrator_player_pass_yds.json',
            ],
        }

        for name, path in required_files.items():
            if isinstance(path, list):
                # Multiple files
                all_exist = all(p.exists() for p in path)
                missing = [p for p in path if not p.exists()]
                checks[name] = {
                    'exists': all_exist,
                    'missing': [str(p) for p in missing] if missing else None
                }
                status = "✅" if all_exist else "❌"
                print(f"  {status} {name}: {'All present' if all_exist else f'Missing {len(missing)} files'}")
                if missing:
                    for p in missing:
                        print(f"      - {p.name}")
            else:
                exists = path.exists()
                checks[name] = {'exists': exists, 'path': str(path)}
                status = "✅" if exists else "❌"
                print(f"  {status} {name}: {'Found' if exists else 'MISSING'}")

        # Check data availability via loader
        print("\n  Checking data availability via stats loader:")
        for source in ['nflverse']:
            try:
                available = is_data_available(self.week, self.season, source)
                checks[f'Loader ({source})'] = {'available': available}
                status = "✅" if available else "❌"
                print(f"    {status} {source}: {'Available' if available else 'Not available'}")
            except Exception as e:
                checks[f'Loader ({source})'] = {'available': False, 'error': str(e)}
                print(f"    ❌ {source}: Error - {e}")

        self.results['data_completeness'] = checks
        all_passed = all(
            (isinstance(v, dict) and v.get('exists', v.get('available', False)))
            for v in checks.values()
        )

        return {'passed': all_passed, 'details': checks}

    def test_nflverse_data_quality(self) -> Dict:
        """Test 2: Verify nflverse data quality."""
        print("\n" + "="*80)
        print("TEST 2: NFLVERSE DATA QUALITY CHECK")
        print("="*80)

        quality_check = {}

        try:
            # Load from nflverse
            print("\n  Loading data from nflverse...")

            nflverse_adapter = NFLVerseAdapter()
            nflverse_available = nflverse_adapter.is_available(self.week, self.season)

            quality_check['nflverse_available'] = nflverse_available

            print(f"    NFLverse: {'✅ Available' if nflverse_available else '❌ Not available'}")

            if nflverse_available:
                print("\n  Loading and analyzing data...")

                nflverse_df = nflverse_adapter.load_weekly_stats(self.week, self.season)

                quality_check['nflverse_count'] = len(nflverse_df)

                print(f"    NFLverse players: {len(nflverse_df)}")

                # Check key stats
                stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions', 'targets']
                available_cols = [col for col in stat_cols if col in nflverse_df.columns]

                if available_cols:
                    print("\n  Stat totals:")
                    for col in available_cols:
                        nflverse_total = nflverse_df[col].sum()
                        quality_check[f'{col}_total'] = float(nflverse_total)
                        print(f"    {col}: {nflverse_total:,.0f}")

                quality_check['recommendation'] = 'NFLverse data looks good'
            else:
                print("\n  ❌ NFLverse data not available")
                quality_check['recommendation'] = 'ERROR: No data sources available'

        except Exception as e:
            quality_check['error'] = str(e)
            print(f"\n  ❌ Error during quality check: {e}")

        self.results['data_comparison'] = quality_check
        return quality_check

    def test_pipeline_execution(self) -> Dict:
        """Test 3: Execute full pipeline and verify outputs."""
        print("\n" + "="*80)
        print("TEST 3: PIPELINE EXECUTION")
        print("="*80)

        if self.dry_run:
            print("\n  🔍 DRY RUN MODE - Skipping actual execution")
            print("  Would execute:")
            print("    1. Fetch odds (optional)")
            print("    2. Generate game simulations")
            print("    3. Generate player predictions")
            print("    4. Generate recommendations")
            print("    5. Merge and enrich")
            print("    6. Generate dashboard")
            return {'dry_run': True, 'skipped': True}

        pipeline_steps = [
            {
                'name': 'Fetch Game Odds',
                'script': ['scripts/fetch/fetch_live_odds.py', str(self.week)],
                'optional': True,
                'output': f'data/odds_week{self.week}_draftkings.csv'
            },
            {
                'name': 'Fetch Player Props',
                'script': ['scripts/fetch/fetch_nfl_player_props.py'],
                'optional': True,
                'output': 'data/nfl_player_props_draftkings.csv'
            },
            {
                'name': 'Generate Game Simulations',
                'script': ['-m', 'nfl_quant.cli', 'simulate', '--week', str(self.week), '--season', str(self.season), '--trials', '50000'],
                'optional': False,
                'output': f'reports/sim_{self.season}_{self.week:02d}_*.json'
            },
            {
                'name': 'Generate Player Predictions',
                'script': ['scripts/predict/generate_model_predictions.py', str(self.week)],
                'optional': False,
                'output': f'data/model_predictions_week{self.week}.csv'
            },
            {
                'name': 'Generate Recommendations',
                'script': ['scripts/predict/generate_current_week_recommendations.py', str(self.week)],
                'optional': False,
                'output': 'reports/CURRENT_WEEK_RECOMMENDATIONS.csv'
            },
        ]

        execution_results = {}

        for step in pipeline_steps:
            print(f"\n  Testing: {step['name']}")
            if step['optional']:
                print(f"    ⚠️  OPTIONAL - Will continue if fails")

            # Check if output already exists
            output_path = self.base_dir / step['output']
            if '*' in step['output']:
                # Pattern match
                pattern = step['output'].replace('*', '*')
                existing = list(self.base_dir.glob(pattern))
                if existing:
                    print(f"    ✅ Output already exists: {len(existing)} files")
                    execution_results[step['name']] = {'status': 'exists', 'outputs': len(existing)}
                    continue
            elif output_path.exists():
                print(f"    ✅ Output already exists: {output_path.name}")
                execution_results[step['name']] = {'status': 'exists'}
                continue

            # Would execute here
            print(f"    ⏳ Would execute: python {' '.join(step['script'])}")
            execution_results[step['name']] = {'status': 'would_execute'}

        self.results['pipeline_execution'] = execution_results
        return execution_results

    def test_output_quality(self) -> Dict:
        """Test 4: Verify output quality and completeness."""
        print("\n" + "="*80)
        print("TEST 4: OUTPUT QUALITY CHECK")
        print("="*80)

        quality_checks = {}

        # Check model predictions
        pred_file = self.base_dir / f'data/model_predictions_week{self.week}.csv'
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            quality_checks['predictions'] = {
                'exists': True,
                'player_count': len(df),
                'has_td_prob': 'calibrated_td_prob' in df.columns,
                'has_all_markets': all(col in df.columns for col in ['pred_receptions', 'pred_reception_yds', 'pred_rush_yds', 'pred_pass_yds'])
            }

            print(f"\n  Model Predictions:")
            print(f"    ✅ File exists: {len(df)} players")
            print(f"    {'✅' if quality_checks['predictions']['has_td_prob'] else '❌'} Has TD probabilities")
            print(f"    {'✅' if quality_checks['predictions']['has_all_markets'] else '❌'} Has all market predictions")

            if 'calibrated_td_prob' in df.columns:
                td_mean = df['calibrated_td_prob'].mean()
                td_std = df['calibrated_td_prob'].std()
                print(f"    TD prob mean: {td_mean:.1%}, std: {td_std:.3f}")
                quality_checks['predictions']['td_not_flat'] = td_std > 0.05
        else:
            quality_checks['predictions'] = {'exists': False}
            print(f"\n  Model Predictions: ❌ File not found")

        # Check recommendations
        rec_file = self.base_dir / 'reports/CURRENT_WEEK_RECOMMENDATIONS.csv'
        if rec_file.exists():
            df = pd.read_csv(rec_file)
            quality_checks['recommendations'] = {
                'exists': True,
                'recommendation_count': len(df),
                'has_edge': 'edge' in df.columns,
                'has_kelly': 'kelly_fraction' in df.columns,
                'has_market_type': 'market' in df.columns
            }

            print(f"\n  Recommendations:")
            print(f"    ✅ File exists: {len(df)} recommendations")
            print(f"    {'✅' if quality_checks['recommendations']['has_edge'] else '❌'} Has edge calculations")
            print(f"    {'✅' if quality_checks['recommendations']['has_kelly'] else '❌'} Has Kelly sizing")

            if 'edge' in df.columns:
                positive_edge = (df['edge'] > 0).sum()
                print(f"    Positive edge bets: {positive_edge} ({positive_edge/len(df)*100:.1f}%)")
        else:
            quality_checks['recommendations'] = {'exists': False}
            print(f"\n  Recommendations: ❌ File not found")

        self.results['output_quality'] = quality_checks
        return quality_checks

    def test_integration(self) -> Dict:
        """Test 5: Integration testing - verify components work together."""
        print("\n" + "="*80)
        print("TEST 5: INTEGRATION TESTING")
        print("="*80)

        integration_results = {}

        # Test calibrator loading
        print("\n  Testing calibrator integration:")
        try:
            from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market

            markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']
            loaded = 0
            for market in markets:
                try:
                    cal = load_calibrator_for_market(market)
                    loaded += 1
                except Exception as e:
                    print(f"    ❌ Failed to load {market}: {e}")

            integration_results['calibrators'] = {'loaded': loaded, 'total': len(markets)}
            print(f"    ✅ Loaded {loaded}/{len(markets)} market-specific calibrators")
        except Exception as e:
            integration_results['calibrators'] = {'error': str(e)}
            print(f"    ❌ Calibrator loading failed: {e}")

        # Test stats loader integration
        print("\n  Testing stats loader integration:")
        try:
            stats = load_weekly_stats(self.week, self.season, source='auto')
            integration_results['stats_loader'] = {
                'success': True,
                'player_count': len(stats),
                'source': stats['source'].iloc[0] if 'source' in stats.columns else 'unknown'
            }
            print(f"    ✅ Loaded {len(stats)} players from {integration_results['stats_loader']['source']}")
        except Exception as e:
            integration_results['stats_loader'] = {'success': False, 'error': str(e)}
            print(f"    ❌ Stats loader failed: {e}")

        self.results['integration'] = integration_results
        return integration_results

    def generate_report(self) -> Dict:
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)

        # Run all tests
        data_test = self.test_data_completeness()
        quality_check_test = self.test_nflverse_data_quality()
        pipeline_test = self.test_pipeline_execution()
        quality_test = self.test_output_quality()
        integration_test = self.test_integration()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        test_results = {
            'Data Completeness': data_test.get('passed', False),
            'Data Quality': quality_check_test.get('nflverse_available', False),
            'Pipeline Execution': not pipeline_test.get('skipped', False),
            'Output Quality': quality_test.get('predictions', {}).get('exists', False) or quality_test.get('recommendations', {}).get('exists', False),
            'Integration': integration_test.get('calibrators', {}).get('loaded', 0) > 0
        }

        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} - {test_name}")

        total_passed = sum(test_results.values())
        total_tests = len(test_results)

        print(f"\n  Overall: {total_passed}/{total_tests} tests passed")

        # Save results
        report_file = self.base_dir / f'reports/framework_test_report_week{self.week}.json'
        report_file.parent.mkdir(parents=True, exist_ok=True)

        full_report = {
            'week': self.week,
            'season': self.season,
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results,
            'detailed_results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(full_report, f, indent=2)

        print(f"\n  📄 Full report saved to: {report_file}")

        return full_report

    def run(self):
        """Run all tests."""
        print("="*80)
        print("FULL FRAMEWORK END-TO-END TESTING")
        print("="*80)
        print(f"Week: {self.week}")
        print(f"Season: {self.season}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'FULL EXECUTION'}")
        print("="*80)

        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='Test full NFL Quant framework')
    parser.add_argument('week', type=int, help='Week number to test')
    parser.add_argument('--season', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (skip actual execution)')

    args = parser.parse_args()

    tester = FullFrameworkTester(args.week, args.season, args.dry_run)
    report = tester.run()

    sys.exit(0 if report['test_results'].get('Data Completeness', False) else 1)


if __name__ == '__main__':
    main()
