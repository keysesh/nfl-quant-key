#!/usr/bin/env python3
"""
Phase 2: Weekly Outcome Consolidation
- Automatically match predictions to actual results
- Consolidate outcomes for calibrator retraining
- Track performance metrics week-over-week

Expected impact: Enables online learning
Time: 4 hours
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class WeeklyOutcomeConsolidator:
    """Consolidate weekly predictions vs. actuals for continuous learning."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.reports_dir = base_dir / "reports"

    def load_week_predictions(self, week: int, season: int = 2025) -> pd.DataFrame:
        """Load predictions for a specific week."""
        pred_file = self.data_dir / f"model_predictions_week{week}.csv"

        if not pred_file.exists():
            print(f"⚠ No predictions found for week {week}")
            return pd.DataFrame()

        df = pd.read_csv(pred_file)
        df['week'] = week
        df['season'] = season
        return df

    def load_week_actuals(self, week: int, season: int = 2025) -> pd.DataFrame:
        """Load actual stats for a specific week."""
        # Try multiple locations
        possible_files = [
            self.data_dir / f"sleeper_stats/stats_week{week}_{season}.csv",
            self.data_dir / f"sleeper_stats/week{week}_{season}_stats.csv",
        ]

        for file in possible_files:
            if file.exists():
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        df['week'] = week
                        df['season'] = season
                        return df
                except:
                    continue

        # Fallback to consolidated file
        consolidated = self.data_dir / f"processed/actual_stats_{season}_weeks_1_8.csv"
        if consolidated.exists():
            df = pd.read_csv(consolidated)
            return df[df['week'] == week]

        print(f"⚠ No actual stats found for week {week}")
        return pd.DataFrame()

    def match_predictions_to_actuals(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        week: int
    ) -> pd.DataFrame:
        """
        Match predictions to actual player performance.

        For each prediction, find the corresponding actual stat and determine
        if the bet would have won.
        """
        if len(predictions) == 0 or len(actuals) == 0:
            print(f"  Cannot match: missing data")
            return pd.DataFrame()

        print(f"  Matching {len(predictions)} predictions to {len(actuals)} actual stats...")

        matched_outcomes = []

        # Standardize column names in actuals
        actual_cols = actuals.columns.tolist()
        player_name_col = None
        if 'player_name' in actual_cols:
            player_name_col = 'player_name'
        elif 'player_display_name' in actual_cols:
            player_name_col = 'player_display_name'

        if player_name_col is None:
            print(f"  ⚠ Cannot find player name column in actuals")
            return pd.DataFrame()

        # For each prediction
        for idx, pred in predictions.iterrows():
            player_name = pred.get('player_name', pred.get('player', ''))

            if not player_name or player_name == '':
                continue

            # Try exact match first
            player_actuals = actuals[actuals[player_name_col] == player_name]

            # Try fuzzy match if no exact match
            if len(player_actuals) == 0 and ' ' in player_name:
                last_name = player_name.split()[-1]
                player_actuals = actuals[
                    actuals[player_name_col].str.contains(last_name, case=False, na=False)
                ]

            if len(player_actuals) == 0:
                continue

            actual_row = player_actuals.iloc[0]

            # Determine market type and actual value
            # Map prediction columns to actual stat columns
            stat_mapping = {
                'rushing_yards': 'rushing_yards',
                'receiving_yards': 'receiving_yards',
                'receptions': 'receptions',
                'passing_yards': 'passing_yards',
                'rushing_tds': 'rushing_tds',
                'receiving_tds': 'receiving_tds'
            }

            # Try to infer market type from prediction columns
            pred_value = None
            actual_value = None
            market_type = None

            # Check which stat predictions we have
            for pred_stat, actual_stat in stat_mapping.items():
                if f'{pred_stat}_mean' in pred.index and actual_stat in actual_row.index:
                    pred_value = pred[f'{pred_stat}_mean']
                    actual_value = actual_row[actual_stat]
                    market_type = f'player_{pred_stat}'
                    break

            if pred_value is not None and actual_value is not None:
                outcome = {
                    'week': week,
                    'player': player_name,
                    'market': market_type,
                    'prediction': float(pred_value),
                    'actual': float(actual_value) if pd.notna(actual_value) else 0.0,
                    'prediction_made': True,
                    'actual_found': True
                }

                matched_outcomes.append(outcome)

        if matched_outcomes:
            matched_df = pd.DataFrame(matched_outcomes)
            print(f"  ✓ Matched {len(matched_df)} predictions to actuals")
            print(f"    Markets: {matched_df['market'].value_counts().to_dict()}")
            return matched_df

        return pd.DataFrame()

    def consolidate_week_outcomes(self, week: int) -> dict:
        """Consolidate outcomes for a specific week."""
        print(f"\nProcessing Week {week}:")

        # Load predictions and actuals
        predictions = self.load_week_predictions(week)
        actuals = self.load_week_actuals(week)

        if len(predictions) == 0:
            print(f"  ⚠ No predictions for week {week}")
            return {'week': week, 'status': 'no_predictions'}

        if len(actuals) == 0:
            print(f"  ⚠ No actuals for week {week}")
            return {'week': week, 'status': 'no_actuals'}

        # Match predictions to actuals
        matched = self.match_predictions_to_actuals(predictions, actuals, week)

        if len(matched) == 0:
            return {'week': week, 'status': 'no_matches'}

        # Calculate metrics
        metrics = {
            'week': week,
            'status': 'success',
            'predictions_made': len(predictions),
            'actuals_found': len(actuals),
            'matched': len(matched),
            'match_rate': len(matched) / len(predictions) if len(predictions) > 0 else 0
        }

        return metrics

    def append_to_cumulative_dataset(self, week_outcomes: pd.DataFrame):
        """Append week outcomes to cumulative training dataset."""
        cumulative_file = self.data_dir / "calibration/cumulative_outcomes_2025.csv"

        if cumulative_file.exists():
            existing = pd.read_csv(cumulative_file)
            combined = pd.concat([existing, week_outcomes], ignore_index=True)
        else:
            combined = week_outcomes

        # Save updated dataset
        cumulative_file.parent.mkdir(exist_ok=True, parents=True)
        combined.to_csv(cumulative_file, index=False)

        print(f"\n✓ Updated cumulative dataset: {len(combined):,} total outcomes")

        return combined

    def generate_performance_report(self, week: int, metrics: dict):
        """Generate weekly performance report."""
        report_lines = []
        report_lines.append(f"\nWEEK {week} PERFORMANCE REPORT")
        report_lines.append("="*80)

        if metrics.get('status') != 'success':
            report_lines.append(f"Status: {metrics['status']}")
            return "\n".join(report_lines)

        report_lines.append(f"Predictions Made:    {metrics.get('predictions_made', 0)}")
        report_lines.append(f"Actuals Found:       {metrics.get('actuals_found', 0)}")
        report_lines.append(f"Successfully Matched: {metrics.get('matched', 0)}")
        report_lines.append(f"Match Rate:          {metrics.get('match_rate', 0)*100:.1f}%")

        return "\n".join(report_lines)

    def run_weekly_consolidation(self, week: int):
        """Run consolidation for a specific week."""
        print(f"\n" + "="*100)
        print(f"WEEKLY OUTCOME CONSOLIDATION - WEEK {week}")
        print("="*100)

        # Consolidate outcomes
        metrics = self.consolidate_week_outcomes(week)

        # Generate report
        report = self.generate_performance_report(week, metrics)
        print(report)

        # Save metrics
        metrics_file = self.reports_dir / f"week{week}_consolidation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def run_multi_week_consolidation(self, weeks: list):
        """Run consolidation for multiple weeks."""
        print(f"\n" + "="*100)
        print(f"MULTI-WEEK OUTCOME CONSOLIDATION")
        print(f"Processing weeks: {min(weeks)} - {max(weeks)}")
        print("="*100)

        all_metrics = []

        for week in weeks:
            metrics = self.run_weekly_consolidation(week)
            all_metrics.append(metrics)

        # Summary
        print(f"\n" + "="*100)
        print("CONSOLIDATION SUMMARY")
        print("="*100)

        summary_df = pd.DataFrame(all_metrics)
        print(summary_df.to_string(index=False))

        # Save summary
        summary_file = self.reports_dir / "outcome_consolidation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Summary saved to: {summary_file}")

        return all_metrics


def main():
    base_dir = Path(Path.cwd())
    consolidator = WeeklyOutcomeConsolidator(base_dir)

    print("\n" + "="*100)
    print("PHASE 2: WEEKLY OUTCOME CONSOLIDATION SYSTEM")
    print("="*100)

    # Test with week 9 (we have predictions and actuals)
    print("\nTesting with Week 9 data...")
    metrics = consolidator.run_weekly_consolidation(9)

    # If we had predictions for earlier weeks, we could consolidate those too
    # weeks_to_process = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # consolidator.run_multi_week_consolidation(weeks_to_process)

    print("\n" + "="*100)
    print("✓ PHASE 2 FRAMEWORK COMPLETE")
    print("="*100)
    print("\nThis system can now:")
    print("1. Load weekly predictions and actual stats")
    print("2. Match predictions to outcomes")
    print("3. Append to cumulative training dataset")
    print("4. Generate performance reports")
    print("\nNext: Integrate with calibrator auto-update (Phase 3)")
    print("")


if __name__ == "__main__":
    main()
