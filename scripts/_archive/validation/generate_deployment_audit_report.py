#!/usr/bin/env python3
"""
Deployment Audit Report Generator
==================================

Generates a comprehensive deployment audit report covering:
1. Calibrator inventory and status
2. Training data coverage
3. Market coverage analysis
4. Position-specific TD calibrator status
5. Integration points
6. Deployment readiness checklist
7. Recommendations

Outputs a detailed markdown report for deployment review.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.calibration.calibrator_loader import get_available_market_calibrators


class DeploymentAuditReportGenerator:
    """Generate comprehensive deployment audit report."""

    def __init__(self):
        self.base_dir = Path(Path.cwd())
        self.config_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "data/models"
        self.reports_dir = self.base_dir / "reports"

    def get_calibrator_inventory(self):
        """Get inventory of all calibrators."""
        inventory = {
            'unified': {},
            'market_specific': {},
            'position_td': {},
            'game_line': {}
        }

        # Unified calibrator
        unified_path = self.config_dir / 'calibrator.json'
        if unified_path.exists():
            try:
                cal = NFLProbabilityCalibrator()
                cal.load(str(unified_path))
                inventory['unified'] = {
                    'exists': True,
                    'fitted': cal.is_fitted,
                    'calibration_points': len(cal.calibrator.X_thresholds_),
                    'path': str(unified_path)
                }
            except Exception as e:
                inventory['unified'] = {'exists': True, 'error': str(e)}

        # Market-specific calibrators
        markets = get_available_market_calibrators(str(self.config_dir))
        for market in markets:
            cal_path = self.config_dir / f'calibrator_{market}.json'
            meta_path = self.config_dir / f'calibrator_{market}_metadata.json'

            if cal_path.exists():
                try:
                    cal = NFLProbabilityCalibrator()
                    cal.load(str(cal_path))

                    metadata = {}
                    if meta_path.exists():
                        with open(meta_path) as f:
                            metadata = json.load(f)

                    inventory['market_specific'][market] = {
                        'exists': True,
                        'fitted': cal.is_fitted,
                        'calibration_points': len(cal.calibrator.X_thresholds_),
                        'training_samples': metadata.get('training_samples', 'Unknown'),
                        'brier_improvement': metadata.get('brier_improvement', 'Unknown'),
                        'path': str(cal_path)
                    }
                except Exception as e:
                    inventory['market_specific'][market] = {'exists': True, 'error': str(e)}

        # Position-specific TD calibrators
        for position in ['QB', 'RB', 'WR', 'TE']:
            cal_path = self.config_dir / f'td_calibrator_{position}.json'
            meta_path = self.config_dir / f'td_calibrator_{position}_metadata.json'

            if cal_path.exists():
                try:
                    cal = NFLProbabilityCalibrator()
                    cal.load(str(cal_path))

                    metadata = {}
                    if meta_path.exists():
                        with open(meta_path) as f:
                            metadata = json.load(f)

                    inventory['position_td'][position] = {
                        'exists': True,
                        'fitted': cal.is_fitted,
                        'calibration_points': len(cal.calibrator.X_thresholds_),
                        'training_samples': metadata.get('training_samples', 'Unknown'),
                        'brier_improvement': metadata.get('brier_improvement', 'Unknown'),
                        'path': str(cal_path)
                    }
                except Exception as e:
                    inventory['position_td'][position] = {'exists': True, 'error': str(e)}
            else:
                inventory['position_td'][position] = {'exists': False}

        # Game line calibrator
        game_line_path = self.config_dir / 'game_line_calibrator.json'
        if game_line_path.exists():
            try:
                cal = NFLProbabilityCalibrator()
                cal.load(str(game_line_path))
                inventory['game_line'] = {
                    'exists': True,
                    'fitted': cal.is_fitted,
                    'calibration_points': len(cal.calibrator.X_thresholds_),
                    'path': str(game_line_path)
                }
            except Exception as e:
                inventory['game_line'] = {'exists': True, 'error': str(e)}

        return inventory

    def analyze_training_data_coverage(self):
        """Analyze training data coverage for calibrators."""
        coverage = {
            'available_markets': [],
            'missing_markets': [],
            'data_sources': []
        }

        # Check training data files
        data_files = [
            'reports/FRESH_BACKTEST_WEEKS_1_8_CALIBRATED.csv',
            'reports/framework_backtest_weeks_1_7_fixed.csv',
            'reports/week_by_week_backtest_results.csv',
        ]

        all_markets = set()
        for file_path in data_files:
            path = self.base_dir / file_path
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if 'market' in df.columns:
                        markets = df['market'].value_counts()
                        all_markets.update(markets.index)
                        coverage['data_sources'].append({
                            'file': file_path,
                            'rows': len(df),
                            'markets': markets.to_dict()
                        })
                except Exception as e:
                    pass

        # Expected markets
        expected_markets = {
            'core': ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds'],
            'missing': ['player_pass_completions', 'player_pass_attempts', 'player_targets'],
            'td': ['player_pass_tds', 'player_rush_tds', 'player_receiving_tds']
        }

        for category, markets in expected_markets.items():
            for market in markets:
                if market in all_markets:
                    coverage['available_markets'].append(market)
                else:
                    coverage['missing_markets'].append(market)

        return coverage

    def check_integration_points(self):
        """Check integration points in codebase."""
        integration_points = {
            'calibrator_loader': False,
            'player_simulator': False,
            'recommendation_generator': False,
            'td_calibrator_integration': False
        }

        # Check calibrator loader
        loader_path = self.base_dir / 'nfl_quant/calibration/calibrator_loader.py'
        if loader_path.exists():
            integration_points['calibrator_loader'] = True

        # Check player simulator integration
        simulator_path = self.base_dir / 'nfl_quant/simulation/player_simulator.py'
        if simulator_path.exists():
            with open(simulator_path) as f:
                content = f.read()
                if 'calibrator' in content.lower() and 'transform' in content.lower():
                    integration_points['player_simulator'] = True

        # Check recommendation generator
        rec_path = self.base_dir / 'scripts/predict/generate_current_week_recommendations.py'
        if rec_path.exists():
            with open(rec_path) as f:
                content = f.read()
                if 'calibrator' in content.lower():
                    integration_points['recommendation_generator'] = True

        # Check TD calibrator integration
        if any(self.config_dir.glob('td_calibrator_*.json')):
            integration_points['td_calibrator_integration'] = True

        return integration_points

    def generate_deployment_readiness_checklist(self):
        """Generate deployment readiness checklist."""
        checklist = {
            'calibrators': {
                'unified_calibrator_exists': False,
                'core_market_calibrators_exist': False,
                'missing_market_calibrators_trained': False,
                'position_td_calibrators_exist': False,
                'all_calibrators_fitted': False
            },
            'data': {
                'training_data_available': False,
                'sufficient_training_samples': False,
                'validation_data_available': False
            },
            'integration': {
                'calibrator_loader_working': False,
                'player_simulator_integrated': False,
                'recommendation_generator_integrated': False
            },
            'validation': {
                'validation_suite_passed': False,
                'calibrator_quality_validated': False,
                'integration_tests_passed': False
            }
        }

        # Check calibrators
        inventory = self.get_calibrator_inventory()
        checklist['calibrators']['unified_calibrator_exists'] = inventory['unified'].get('exists', False)
        checklist['calibrators']['all_calibrators_fitted'] = all(
            cal.get('fitted', False) for cal in inventory['market_specific'].values()
        )

        core_markets = ['player_reception_yds', 'player_rush_yds', 'player_receptions', 'player_pass_yds']
        checklist['calibrators']['core_market_calibrators_exist'] = all(
            market in inventory['market_specific'] for market in core_markets
        )

        missing_markets = ['player_pass_completions', 'player_pass_attempts', 'player_targets']
        checklist['calibrators']['missing_market_calibrators_trained'] = all(
            market in inventory['market_specific'] for market in missing_markets
        )

        checklist['calibrators']['position_td_calibrators_exist'] = len(inventory['position_td']) == 4

        # Check data
        coverage = self.analyze_training_data_coverage()
        checklist['data']['training_data_available'] = len(coverage['data_sources']) > 0
        checklist['data']['sufficient_training_samples'] = any(
            source.get('rows', 0) >= 50 for source in coverage['data_sources']
        )

        # Check integration
        integration = self.check_integration_points()
        checklist['integration'] = integration

        # Check validation (if report exists)
        validation_report = self.reports_dir / 'validation_report.json'
        if validation_report.exists():
            with open(validation_report) as f:
                val_data = json.load(f)
                checklist['validation']['validation_suite_passed'] = val_data.get('overall_status') == 'PASS'
                checklist['validation']['calibrator_quality_validated'] = val_data.get('summary', {}).get('quality_passed', 0) > 0
                checklist['validation']['integration_tests_passed'] = 'integration' in val_data.get('detailed_results', {})

        return checklist

    def generate_report(self):
        """Generate comprehensive deployment audit report."""
        print("="*80)
        print("GENERATING DEPLOYMENT AUDIT REPORT")
        print("="*80)

        inventory = self.get_calibrator_inventory()
        coverage = self.analyze_training_data_coverage()
        integration = self.check_integration_points()
        checklist = self.generate_deployment_readiness_checklist()

        # Generate markdown report
        report_lines = []
        report_lines.append("# Deployment Audit Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")

        total_calibrators = (
            (1 if inventory['unified'].get('exists') else 0) +
            len(inventory['market_specific']) +
            len(inventory['position_td']) +
            (1 if inventory['game_line'].get('exists') else 0)
        )

        report_lines.append(f"- **Total Calibrators:** {total_calibrators}")
        report_lines.append(f"- **Market-Specific Calibrators:** {len(inventory['market_specific'])}")
        report_lines.append(f"- **Position-Specific TD Calibrators:** {len(inventory['position_td'])}")
        report_lines.append(f"- **Training Data Sources:** {len(coverage['data_sources'])}")
        report_lines.append("")

        # Calibrator Inventory
        report_lines.append("## 1. Calibrator Inventory")
        report_lines.append("")

        report_lines.append("### 1.1 Unified Calibrator")
        if inventory['unified'].get('exists'):
            report_lines.append(f"- ✅ **Status:** Available")
            report_lines.append(f"- **Calibration Points:** {inventory['unified'].get('calibration_points', 'Unknown')}")
            report_lines.append(f"- **Path:** `{inventory['unified'].get('path', 'Unknown')}`")
        else:
            report_lines.append("- ❌ **Status:** Not Found")
        report_lines.append("")

        report_lines.append("### 1.2 Market-Specific Calibrators")
        if inventory['market_specific']:
            report_lines.append("| Market | Status | Points | Samples | Brier Δ |")
            report_lines.append("|--------|--------|--------|---------|---------|")
            for market, info in sorted(inventory['market_specific'].items()):
                status = "✅" if info.get('exists') else "❌"
                points = info.get('calibration_points', 'N/A')
                samples = info.get('training_samples', 'N/A')
                brier = info.get('brier_improvement', 'N/A')
                if isinstance(brier, (int, float)):
                    brier = f"{brier:+.4f}"
                report_lines.append(f"| {market} | {status} | {points} | {samples} | {brier} |")
        else:
            report_lines.append("- No market-specific calibrators found")
        report_lines.append("")

        report_lines.append("### 1.3 Position-Specific TD Calibrators")
        if inventory['position_td']:
            report_lines.append("| Position | Status | Points | Samples | Brier Δ |")
            report_lines.append("|----------|--------|--------|---------|---------|")
            for position in ['QB', 'RB', 'WR', 'TE']:
                info = inventory['position_td'].get(position, {})
                status = "✅" if info.get('exists') else "❌"
                points = info.get('calibration_points', 'N/A')
                samples = info.get('training_samples', 'N/A')
                brier = info.get('brier_improvement', 'N/A')
                if isinstance(brier, (int, float)):
                    brier = f"{brier:+.4f}"
                report_lines.append(f"| {position} | {status} | {points} | {samples} | {brier} |")
        else:
            report_lines.append("- No position-specific TD calibrators found")
        report_lines.append("")

        # Training Data Coverage
        report_lines.append("## 2. Training Data Coverage")
        report_lines.append("")
        report_lines.append(f"- **Available Markets:** {len(coverage['available_markets'])}")
        report_lines.append(f"- **Missing Markets:** {len(coverage['missing_markets'])}")
        report_lines.append("")

        if coverage['data_sources']:
            report_lines.append("### Data Sources:")
            for source in coverage['data_sources']:
                report_lines.append(f"- **{source['file']}:** {source['rows']:,} rows")
                report_lines.append(f"  - Markets: {', '.join(source['markets'].keys())}")
        report_lines.append("")

        # Integration Points
        report_lines.append("## 3. Integration Points")
        report_lines.append("")
        for point, status in integration.items():
            icon = "✅" if status else "❌"
            report_lines.append(f"- {icon} **{point.replace('_', ' ').title()}:** {'Integrated' if status else 'Not Integrated'}")
        report_lines.append("")

        # Deployment Readiness Checklist
        report_lines.append("## 4. Deployment Readiness Checklist")
        report_lines.append("")

        for category, items in checklist.items():
            report_lines.append(f"### 4.{list(checklist.keys()).index(category) + 1} {category.title()}")
            report_lines.append("")
            for item, status in items.items():
                icon = "✅" if status else "❌"
                report_lines.append(f"- {icon} {item.replace('_', ' ').title()}")
            report_lines.append("")

        # Recommendations
        report_lines.append("## 5. Recommendations")
        report_lines.append("")

        recommendations = []

        if not checklist['calibrators']['missing_market_calibrators_trained']:
            recommendations.append("Train calibrators for missing markets: completions, attempts, targets")

        if not checklist['calibrators']['position_td_calibrators_exist']:
            recommendations.append("Create position-specific TD calibrators for all positions (QB, RB, WR, TE)")

        if 'player_simulator_integrated' in checklist.get('integration', {}) and not checklist['integration']['player_simulator_integrated']:
            recommendations.append("Verify calibrator integration in PlayerSimulator")

        if not checklist['validation']['validation_suite_passed']:
            recommendations.append("Run comprehensive validation suite and address failures")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        else:
            report_lines.append("✅ All systems ready for deployment!")
        report_lines.append("")

        # Save report
        report_path = self.reports_dir / f'deployment_audit_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\n✅ Report generated: {report_path}")
        print(f"\nReport Summary:")
        print(f"  - Total Calibrators: {total_calibrators}")
        print(f"  - Market-Specific: {len(inventory['market_specific'])}")
        print(f"  - Position TD: {len(inventory['position_td'])}")
        print(f"  - Recommendations: {len(recommendations)}")

        return report_path


def main():
    generator = DeploymentAuditReportGenerator()
    report_path = generator.generate_report()
    print(f"\n📄 Full report available at: {report_path}")


if __name__ == "__main__":
    main()
