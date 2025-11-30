#!/usr/bin/env python3
"""
Phase 3: Incremental Model Training
- Retrain models weekly with weighted 2024 + 2025 data
- 60% weight on 2024 (stable baseline)
- 40% weight on 2025 (current patterns)

Expected impact: +3 percentage points ROI
Time: 6 hours
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class IncrementalModelTrainer:
    """Retrain models incrementally with growing 2025 data."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.models_dir = base_dir / "data/models"
        self.reports_dir = base_dir / "reports"

    def load_training_data(self, through_week: int = None):
        """
        Load combined 2024 + 2025 training data.

        Args:
            through_week: Include 2025 data through this week (exclusive)
        """
        print("\n" + "="*100)
        print("LOADING TRAINING DATA")
        print("="*100)

        # Load 2024 baseline
        pbp_2024_file = self.data_dir / "nflverse/pbp_2024.parquet"
        if not pbp_2024_file.exists():
            print(f"❌ 2024 PBP data not found: {pbp_2024_file}")
            return None, None

        pbp_2024 = pd.read_parquet(pbp_2024_file)
        print(f"\n✓ Loaded 2024 PBP data: {len(pbp_2024):,} plays")

        # Load 2025 data
        pbp_2025_file = self.data_dir / "nflverse/pbp_2025.parquet"
        if not pbp_2025_file.exists():
            print(f"⚠ 2025 PBP data not found, using 2024 only")
            return pbp_2024, np.ones(len(pbp_2024))

        pbp_2025 = pd.read_parquet(pbp_2025_file)

        if through_week:
            pbp_2025 = pbp_2025[pbp_2025['week'] < through_week]

        print(f"✓ Loaded 2025 PBP data: {len(pbp_2025):,} plays (through week {through_week-1 if through_week else 'all'})")

        # Combine datasets
        pbp_combined = pd.concat([pbp_2024, pbp_2025], ignore_index=True)

        # Create sample weights: 60% for 2024, 40% for 2025
        weights_2024 = np.ones(len(pbp_2024)) * 0.6
        weights_2025 = np.ones(len(pbp_2025)) * 0.4

        sample_weights = np.concatenate([weights_2024, weights_2025])

        print(f"\n✓ Combined dataset: {len(pbp_combined):,} plays")
        print(f"  2024: {len(pbp_2024):,} plays (60% weight)")
        print(f"  2025: {len(pbp_2025):,} plays (40% weight)")

        return pbp_combined, sample_weights

    def prepare_features(self, pbp_data: pd.DataFrame):
        """
        Prepare features for model training.

        This is a simplified version - your actual feature engineering
        would be more sophisticated.
        """
        print("\n" + "="*100)
        print("PREPARING FEATURES")
        print("="*100)

        # Filter to relevant plays
        relevant_plays = pbp_data[
            (pbp_data['play_type'].isin(['run', 'pass'])) &
            (pbp_data['down'].isin([1, 2, 3, 4]))
        ].copy()

        print(f"\n✓ Filtered to {len(relevant_plays):,} relevant plays")

        # Basic feature engineering
        # (In production, you'd use your actual feature engineering pipeline)
        features = pd.DataFrame()

        # Example features - replace with your actual features
        if 'yards_gained' in relevant_plays.columns:
            features['target_yards'] = relevant_plays['yards_gained']

        # Add more features as needed
        # features['down'] = relevant_plays['down']
        # features['ydstogo'] = relevant_plays['ydstogo']
        # etc.

        # Remove rows with missing values
        features = features.dropna()

        print(f"✓ Prepared {len(features):,} training samples with {features.shape[1]} features")

        return features

    def retrain_usage_predictor(
        self,
        pbp_combined: pd.DataFrame,
        sample_weights: np.ndarray,
        through_week: int
    ):
        """Retrain usage predictor with combined data."""
        print("\n" + "="*100)
        print("RETRAINING USAGE PREDICTOR")
        print("="*100)

        # Prepare features (simplified - use your actual pipeline)
        print("\n⚠ Using simplified training for demonstration")
        print("  In production, use your actual feature engineering pipeline")

        # For now, just save a placeholder model with metadata
        model_metadata = {
            'model_type': 'usage_predictor_v4_adaptive',
            'trained_date': datetime.now().isoformat(),
            'training_samples_2024': int((sample_weights == 0.6).sum()),
            'training_samples_2025': int((sample_weights == 0.4).sum()),
            'through_week': through_week,
            'sample_weighting': '60% 2024, 40% 2025',
            'status': 'framework_only'
        }

        # Save metadata
        metadata_file = self.models_dir / f"usage_predictor_v4_adaptive_week{through_week}.json"
        self.models_dir.mkdir(exist_ok=True, parents=True)

        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        print(f"\n✓ Model metadata saved to: {metadata_file}")
        print(f"\n📝 NOTE: Full model retraining requires:")
        print(f"   1. Feature engineering pipeline from existing training scripts")
        print(f"   2. Hyperparameter configuration")
        print(f"   3. Validation set evaluation")
        print(f"   4. Model serialization")
        print(f"\n   See: scripts/train/train_usage_predictor_v4_with_defense.py")

        return model_metadata

    def generate_training_report(self, metadata: dict):
        """Generate model retraining report."""
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("PHASE 3: INCREMENTAL MODEL TRAINING REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*100)

        report_lines.append("\n## TRAINING CONFIGURATION")
        report_lines.append("-"*100)
        report_lines.append(f"Model Type:          {metadata.get('model_type', 'N/A')}")
        report_lines.append(f"Training Date:       {metadata.get('trained_date', 'N/A')}")
        report_lines.append(f"2024 Samples:        {metadata.get('training_samples_2024', 0):,}")
        report_lines.append(f"2025 Samples:        {metadata.get('training_samples_2025', 0):,}")
        report_lines.append(f"Sample Weighting:    {metadata.get('sample_weighting', 'N/A')}")
        report_lines.append(f"Through Week:        {metadata.get('through_week', 'N/A')}")

        report_lines.append("\n## IMPLEMENTATION STATUS")
        report_lines.append("-"*100)
        report_lines.append("✓ Data loading complete")
        report_lines.append("✓ Sample weighting configured")
        report_lines.append("⚠ Full model training integration needed")

        report_lines.append("\n## NEXT STEPS TO COMPLETE PHASE 3")
        report_lines.append("-"*100)
        report_lines.append("1. [ ] Integrate with existing training pipeline:")
        report_lines.append("       scripts/train/train_usage_predictor_v4_with_defense.py")
        report_lines.append("2. [ ] Add sample_weight parameter to model.fit()")
        report_lines.append("3. [ ] Validate on holdout set from 2025 data")
        report_lines.append("4. [ ] Save versioned models")
        report_lines.append("5. [ ] Update prediction scripts to load adaptive models")

        report_lines.append("\n## EXPECTED IMPACT")
        report_lines.append("-"*100)
        report_lines.append("ROI Improvement: +3 percentage points")
        report_lines.append("  From: 45.9% (with improved calibrators)")
        report_lines.append("  To:   48.9% (with adaptive learning)")

        report_lines.append("\n## CODE INTEGRATION EXAMPLE")
        report_lines.append("-"*100)
        report_lines.append("```python")
        report_lines.append("# In your existing training script, modify:")
        report_lines.append("")
        report_lines.append("# Load combined data")
        report_lines.append("pbp_combined, sample_weights = load_training_data(through_week=10)")
        report_lines.append("")
        report_lines.append("# Train with weights")
        report_lines.append("model.fit(X_train, y_train, sample_weight=sample_weights)")
        report_lines.append("")
        report_lines.append("# Save with version")
        report_lines.append("joblib.dump(model, f'models/usage_predictor_v4_adaptive_week{week}.joblib')")
        report_lines.append("```")

        report_lines.append("\n" + "="*100)

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        # Save report
        report_file = self.reports_dir / "PHASE3_TRAINING_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Report saved to: {report_file}")

        return report_text

    def run_incremental_training(self, through_week: int = 10):
        """Run incremental training for specified week."""
        print("\n" + "="*100)
        print("PHASE 3: INCREMENTAL MODEL TRAINING")
        print(f"Training through Week {through_week-1} (to predict Week {through_week})")
        print("="*100)

        # Load data
        pbp_combined, sample_weights = self.load_training_data(through_week)

        if pbp_combined is None:
            print("\n❌ Cannot proceed without training data")
            return None

        # Retrain models (simplified for now)
        metadata = self.retrain_usage_predictor(pbp_combined, sample_weights, through_week)

        # Generate report
        self.generate_training_report(metadata)

        print("\n" + "="*100)
        print("✓ PHASE 3 FRAMEWORK COMPLETE")
        print("="*100)
        print("\nFramework ready for integration with existing training scripts.")
        print("See PHASE3_TRAINING_REPORT.txt for integration instructions.")
        print("")

        return metadata


def main():
    base_dir = Path(Path.cwd())
    trainer = IncrementalModelTrainer(base_dir)

    # Run incremental training through week 10
    # (uses weeks 1-9 of 2025 data to train for week 10 predictions)
    metadata = trainer.run_incremental_training(through_week=10)


if __name__ == "__main__":
    main()
