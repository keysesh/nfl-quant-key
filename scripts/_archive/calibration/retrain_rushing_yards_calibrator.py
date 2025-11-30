"""
Retrain Rushing Yards Calibrator - Fix overconfidence issue

Based on betting ROI diagnostics showing severe probability overconfidence:
- When model says 70% probability, only wins 41% of the time
- Need much more aggressive shrinkage for rushing yards

This script:
1. Loads backtest predictions and actual outcomes
2. Calculates raw probabilities vs actual win rates
3. Fits isotonic calibrator with aggressive shrinkage
4. Saves updated calibrator to configs/

Usage:
    python scripts/calibration/retrain_rushing_yards_calibrator.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
from typing import List, Tuple

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


def load_backtest_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and actuals from weeks 1-8."""

    # Load predictions
    pred_file = Path("reports/v3_backtest_predictions.csv")
    if not pred_file.exists():
        raise FileNotFoundError(
            "Predictions file not found. Run backtest first:\n"
            "python scripts/backtest/backtest_v3_weeks_1_8.py --weeks 1-8"
        )

    predictions = pd.read_csv(pred_file)

    # Parse samples
    predictions['samples_array'] = predictions['samples'].apply(
        lambda x: np.array([float(v) for v in x.split(',')]) if pd.notna(x) else np.array([])
    )

    # Load actuals
    all_actuals = []
    for week in range(1, 9):
        file_path = Path(f"data/sleeper_stats/stats_week{week}_2025.csv")
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['week'] = week

            # Standardize column names
            df = df.rename(columns={
                'rec_yd': 'receiving_yards',
                'rush_yd': 'rushing_yards',
                'pass_yd': 'passing_yards',
            })

            all_actuals.append(df)

    if not all_actuals:
        raise FileNotFoundError("No actual stats files found in data/sleeper_stats/")

    actuals = pd.concat(all_actuals, ignore_index=True)

    return predictions, actuals


def load_historical_props(week: int) -> pd.DataFrame:
    """Load historical player props for a given week."""
    WEEK_DATES = {
        1: "20250909",
        2: "20250916",
        3: "20250923",
        4: "20250930",
        5: "20251007",
        6: "20251014",
        7: "20251021",
        8: "20251028",
    }

    date_str = WEEK_DATES.get(week)
    if not date_str:
        return pd.DataFrame()

    file_path = Path(f"data/historical/backfill/player_props_history_{date_str}T000000Z.csv")

    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Filter to rushing yards only
    df = df[df['market'] == 'player_rush_yds'].copy()
    df['week'] = week

    # Parse into over/under pairs
    lines = []
    for (player, market), group in df.groupby(['player', 'market']):
        over_row = group[group['prop_type'] == 'over']
        under_row = group[group['prop_type'] == 'under']

        if len(over_row) == 0 or len(under_row) == 0:
            continue

        over_line = over_row.iloc[0]['line']
        under_line = under_row.iloc[0]['line']

        if pd.isna(over_line) or pd.isna(under_line) or over_line != under_line:
            continue

        lines.append({
            'week': week,
            'player': player,
            'market': market,
            'line': over_line,
            'over_odds': over_row.iloc[0]['price'],
            'under_odds': under_row.iloc[0]['price'],
        })

    return pd.DataFrame(lines)


def prepare_calibration_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for calibration training.

    Returns:
        raw_probabilities: Array of model probabilities (0-1)
        actual_outcomes: Array of actual binary outcomes (0 or 1)
    """

    print("="*80)
    print("PREPARING RUSHING YARDS CALIBRATION DATA")
    print("="*80)
    print()

    predictions, actuals = load_backtest_data()

    # Filter to rushing yards only
    rushing_preds = predictions[predictions['stat_type'] == 'rushing_yards'].copy()

    print(f"✅ Loaded {len(rushing_preds)} rushing yards predictions")

    # Collect all (raw_prob, outcome) pairs
    raw_probs_over = []
    raw_probs_under = []
    outcomes_over = []
    outcomes_under = []

    total_matched = 0

    for week in range(1, 9):
        props = load_historical_props(week)
        if props.empty:
            continue

        week_preds = rushing_preds[rushing_preds['week'] == week]

        for _, prop in props.iterrows():
            # Match prediction
            pred_match = week_preds[week_preds['player_name'] == prop['player']]
            if len(pred_match) == 0:
                continue

            # Match actual
            actual_match = actuals[
                (actuals['player_name'] == prop['player']) &
                (actuals['week'] == week)
            ]
            if len(actual_match) == 0:
                continue

            actual_value = actual_match.iloc[0].get('rushing_yards')
            if pd.isna(actual_value):
                continue

            # Get prediction samples
            pred = pred_match.iloc[0]
            samples = pred['samples_array']

            if len(samples) == 0:
                continue

            # Calculate raw probabilities
            line = prop['line']
            raw_prob_over = np.mean(samples > line)
            raw_prob_under = np.mean(samples < line)

            # Actual outcomes
            outcome_over = 1 if actual_value > line else 0
            outcome_under = 1 if actual_value < line else 0

            # Collect
            raw_probs_over.append(raw_prob_over)
            outcomes_over.append(outcome_over)

            raw_probs_under.append(raw_prob_under)
            outcomes_under.append(outcome_under)

            total_matched += 1

    print(f"✅ Matched {total_matched} props with predictions and actuals")
    print()

    # Combine over and under (they're complementary, so we get 2x data)
    all_raw_probs = np.array(raw_probs_over + raw_probs_under)
    all_outcomes = np.array(outcomes_over + outcomes_under)

    print(f"📊 Total calibration samples: {len(all_raw_probs)}")
    print(f"   Mean raw probability: {all_raw_probs.mean():.3f}")
    print(f"   Mean actual outcome: {all_outcomes.mean():.3f}")
    print()

    return all_raw_probs, all_outcomes


def analyze_current_calibration(raw_probs: np.ndarray, outcomes: np.ndarray):
    """Analyze how badly calibrated the current model is."""

    print("="*80)
    print("CURRENT CALIBRATION ANALYSIS")
    print("="*80)
    print()

    # Bin by probability ranges
    bins = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 0.85),
        (0.85, 0.90),
        (0.90, 0.95),
        (0.95, 1.00),
    ]

    print(f"{'Range':<15} {'Model Says':<12} {'Actual Rate':<12} {'Error':<10} {'Count':<8}")
    print("-" * 65)

    total_error = 0
    total_count = 0

    for bin_min, bin_max in bins:
        mask = (raw_probs >= bin_min) & (raw_probs < bin_max)
        bin_probs = raw_probs[mask]
        bin_outcomes = outcomes[mask]

        if len(bin_probs) == 0:
            continue

        avg_prob = bin_probs.mean()
        actual_rate = bin_outcomes.mean()
        error = avg_prob - actual_rate

        total_error += abs(error) * len(bin_probs)
        total_count += len(bin_probs)

        indicator = "❌" if abs(error) > 0.10 else ("⚠️" if abs(error) > 0.05 else "✅")

        print(f"{bin_min:.2f}-{bin_max:.2f}    {avg_prob:<12.3f} {actual_rate:<12.3f} {error:+10.3f} {len(bin_probs):<8} {indicator}")

    mae = total_error / total_count if total_count > 0 else 0
    print("-" * 65)
    print(f"Mean Absolute Error: {mae:.3f}")
    print()


def train_new_calibrator(
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
    high_prob_threshold: float = 0.60,
    high_prob_shrinkage: float = 0.15
) -> NFLProbabilityCalibrator:
    """
    Train new calibrator with aggressive shrinkage.

    Args:
        raw_probs: Raw model probabilities
        outcomes: Actual binary outcomes
        high_prob_threshold: Apply shrinkage above this probability (default 0.60)
        high_prob_shrinkage: Shrinkage factor (default 0.15 = 85% shrinkage toward 50%)

    Returns:
        Fitted calibrator
    """

    print("="*80)
    print("TRAINING NEW CALIBRATOR")
    print("="*80)
    print()
    print(f"⚙️  Settings:")
    print(f"   High Probability Threshold: {high_prob_threshold:.2f}")
    print(f"   High Probability Shrinkage: {high_prob_shrinkage:.2f}")
    print(f"   (Lower shrinkage = more aggressive dampening)")
    print()

    calibrator = NFLProbabilityCalibrator(
        high_prob_threshold=high_prob_threshold,
        high_prob_shrinkage=high_prob_shrinkage
    )

    calibrator.fit(raw_probs, outcomes)

    return calibrator


def validate_new_calibrator(
    calibrator: NFLProbabilityCalibrator,
    raw_probs: np.ndarray,
    outcomes: np.ndarray
):
    """Validate that new calibrator improves calibration."""

    print("="*80)
    print("VALIDATING NEW CALIBRATOR")
    print("="*80)
    print()

    # Apply calibration
    calibrated_probs = calibrator.transform(raw_probs)

    # Bin analysis
    bins = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 0.85),
        (0.85, 0.90),
    ]

    print(f"{'Range':<15} {'Calibrated':<12} {'Actual Rate':<12} {'Error':<10} {'Count':<8}")
    print("-" * 65)

    total_error = 0
    total_count = 0

    for bin_min, bin_max in bins:
        mask = (calibrated_probs >= bin_min) & (calibrated_probs < bin_max)
        bin_probs = calibrated_probs[mask]
        bin_outcomes = outcomes[mask]

        if len(bin_probs) == 0:
            continue

        avg_prob = bin_probs.mean()
        actual_rate = bin_outcomes.mean()
        error = avg_prob - actual_rate

        total_error += abs(error) * len(bin_probs)
        total_count += len(bin_probs)

        indicator = "❌" if abs(error) > 0.10 else ("⚠️" if abs(error) > 0.05 else "✅")

        print(f"{bin_min:.2f}-{bin_max:.2f}    {avg_prob:<12.3f} {actual_rate:<12.3f} {error:+10.3f} {len(bin_probs):<8} {indicator}")

    mae = total_error / total_count if total_count > 0 else 0
    print("-" * 65)
    print(f"Mean Absolute Error: {mae:.3f}")
    print()


def main():
    print("\n")

    # Step 1: Prepare data
    raw_probs, outcomes = prepare_calibration_data()

    # Step 2: Analyze current calibration issues
    analyze_current_calibration(raw_probs, outcomes)

    # Step 3: Train new calibrator with aggressive shrinkage
    # Based on diagnostics: 70% prob only wins 41%, need to pull way down
    calibrator = train_new_calibrator(
        raw_probs,
        outcomes,
        high_prob_threshold=0.60,  # Start shrinking at 60%
        high_prob_shrinkage=0.15   # Very aggressive (85% shrinkage)
    )

    # Step 4: Validate
    validate_new_calibrator(calibrator, raw_probs, outcomes)

    # Step 5: Save
    output_path = "configs/calibrator_player_rush_yds.json"

    # Backup old calibrator
    backup_path = "configs/calibrator_player_rush_yds_pre_fix.json"
    if Path(output_path).exists():
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"📦 Backed up old calibrator to {backup_path}")

    calibrator.save(output_path)

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Re-run backtest with new calibrator to generate new predictions
2. Re-run ROI analysis to measure improvement
3. Expected: Rushing yards ROI should improve from -1.3% to positive

Commands:
    python scripts/backtest/backtest_v3_weeks_1_8.py --weeks 1-8
    python scripts/backtest/betting_roi_analysis.py --min-edge 0.05
    """)


if __name__ == '__main__':
    main()
