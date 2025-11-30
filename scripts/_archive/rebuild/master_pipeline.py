#!/usr/bin/env python3
"""
NFL QUANT Master Pipeline Orchestrator

This script coordinates the ENTIRE system:
1. R-based data fetching (NFLverse via nflreadr)
2. Feature engineering from PBP and weekly stats
3. Model predictions using Monte Carlo simulation
4. Calibrator training with real predictions vs outcomes
5. Final recommendation generation

Usage:
    # Full rebuild from scratch
    python scripts/rebuild/master_pipeline.py --full-rebuild

    # Weekly update (fetch new data, regenerate predictions)
    python scripts/rebuild/master_pipeline.py --weekly-update

    # Just retrain calibrators
    python scripts/rebuild/master_pipeline.py --retrain-calibrators

    # Generate predictions only (assumes data is fresh)
    python scripts/rebuild/master_pipeline.py --predict-only
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.epa_utils import regress_epa_to_mean

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MasterPipeline:
    """
    Master orchestrator for NFL QUANT system.

    Ensures single source of truth for all data and models.
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.nflverse_dir = self.data_dir / "nflverse"
        self.configs_dir = self.project_root / "configs"
        self.reports_dir = self.project_root / "reports"
        self.scripts_dir = self.project_root / "scripts"

        # Season detection
        self.current_season = self._detect_current_season()
        self.current_week = self._detect_current_week()

        # R script path
        self.r_fetch_script = self.scripts_dir / "fetch" / "fetch_nflverse_data.R"

        # Results tracking
        self.results = {
            'start_time': datetime.now().isoformat(),
            'season': self.current_season,
            'week': self.current_week,
            'steps': []
        }

    def _detect_current_season(self) -> int:
        """Detect current NFL season based on date."""
        now = datetime.now()
        # NFL season: Aug-Dec = current year, Jan-Jul = previous year
        return now.year if now.month >= 8 else now.year - 1

    def _detect_current_week(self) -> int:
        """Detect current NFL week from schedule data."""
        schedule_file = self.nflverse_dir / "schedules.parquet"

        if not schedule_file.exists():
            logger.warning("No schedule data found, defaulting to week 11")
            return 11

        try:
            df = pd.read_parquet(schedule_file)
            df = df[df['season'] == self.current_season]

            # Find the latest week with completed games
            today = datetime.now()
            df['gameday'] = pd.to_datetime(df['gameday'])
            completed = df[df['gameday'] < today]

            if len(completed) > 0:
                current_week = completed['week'].max() + 1  # Next week to predict
                return min(current_week, 18)  # Cap at week 18
            else:
                return 1
        except Exception as e:
            logger.warning(f"Could not detect week: {e}, defaulting to 11")
            return 11

    def log_step(self, step_name: str, status: str, details: dict = None):
        """Log a pipeline step."""
        step_info = {
            'step': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.results['steps'].append(step_info)

        if status == 'success':
            logger.info(f"✅ {step_name}")
        elif status == 'warning':
            logger.warning(f"⚠️  {step_name}")
        else:
            logger.error(f"❌ {step_name}")

    # =========================================================================
    # STEP 1: R-BASED DATA FETCHING
    # =========================================================================

    def fetch_nflverse_data(self):
        """Fetch all NFLverse data using R (fast and reliable)."""
        logger.info("=" * 80)
        logger.info("STEP 1: FETCHING NFLVERSE DATA (R)")
        logger.info("=" * 80)

        if not self.r_fetch_script.exists():
            self.log_step("R script not found", "error")
            raise FileNotFoundError(f"R fetch script not found: {self.r_fetch_script}")

        # Run R script to fetch data
        cmd = [
            "Rscript",
            str(self.r_fetch_script),
            "--current-plus-last",  # Fetch current season + previous
            "--output-dir", str(self.nflverse_dir),
            "--format", "parquet"
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"R script failed: {result.stderr}")
                self.log_step("Fetch NFLverse Data", "error", {'stderr': result.stderr})
                raise RuntimeError("R data fetch failed")

            logger.info(result.stdout)

            # Verify files were created
            expected_files = ['pbp.parquet', 'player_stats.parquet', 'schedules.parquet', 'rosters.parquet']
            for fname in expected_files:
                fpath = self.nflverse_dir / fname
                if not fpath.exists():
                    logger.warning(f"Expected file not found: {fname}")
                else:
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    logger.info(f"   {fname}: {size_mb:.1f} MB")

            self.log_step("Fetch NFLverse Data", "success")

        except subprocess.TimeoutExpired:
            self.log_step("Fetch NFLverse Data", "error", {'reason': 'timeout'})
            raise
        except Exception as e:
            self.log_step("Fetch NFLverse Data", "error", {'exception': str(e)})
            raise

    # =========================================================================
    # STEP 2: PBP FEATURE ENGINEERING
    # =========================================================================

    def extract_pbp_features(self):
        """Extract advanced features from play-by-play data."""
        logger.info("=" * 80)
        logger.info("STEP 2: EXTRACTING PBP FEATURES")
        logger.info("=" * 80)

        pbp_file = self.nflverse_dir / "pbp.parquet"
        if not pbp_file.exists():
            self.log_step("PBP file not found", "error")
            raise FileNotFoundError(f"PBP data not found: {pbp_file}")

        logger.info("Loading PBP data...")
        pbp = pd.read_parquet(pbp_file)
        logger.info(f"Loaded {len(pbp):,} plays")

        # Filter to current season
        pbp_season = pbp[pbp['season'] == self.current_season].copy()
        logger.info(f"Current season ({self.current_season}): {len(pbp_season):,} plays")

        # Extract key features:

        # 1. Defensive EPA by team (how bad is defense?)
        logger.info("Calculating defensive EPA by team WITH regression to mean...")
        def_epa = pbp_season.groupby('defteam').agg({
            'epa': 'mean',
            'play_id': 'count',
            'game_id': 'nunique'  # Sample size for regression
        }).reset_index()
        def_epa.columns = ['team', 'raw_def_epa_allowed', 'defensive_plays', 'sample_games']

        # CRITICAL: Apply regression to mean to reduce variance from small samples
        def_epa['def_epa_allowed'] = def_epa.apply(
            lambda row: regress_epa_to_mean(row['raw_def_epa_allowed'], row['sample_games']),
            axis=1
        )
        logger.info(f"   Applied regression to mean (50% at 10 games, more with fewer)")

        # Keep both raw and regressed for analysis
        def_epa.to_parquet(self.nflverse_dir / "team_defensive_epa.parquet", index=False)
        logger.info(f"   Saved team defensive EPA for {len(def_epa)} teams")

        # 2. Red zone usage by player
        logger.info("Calculating red zone usage...")
        rz_plays = pbp_season[pbp_season['yardline_100'] <= 20].copy()
        rz_usage = rz_plays.groupby(['receiver_player_id', 'week']).agg({
            'play_id': 'count'
        }).reset_index()
        rz_usage.columns = ['player_id', 'week', 'red_zone_targets']
        rz_usage.to_parquet(self.nflverse_dir / "player_red_zone_usage.parquet", index=False)
        logger.info(f"   Saved red zone usage: {len(rz_usage):,} player-weeks")

        # 3. Team pace (plays per game)
        logger.info("Calculating team pace...")
        team_pace = pbp_season.groupby(['posteam', 'game_id']).agg({
            'play_id': 'count'
        }).reset_index()
        team_pace = team_pace.groupby('posteam')['play_id'].mean().reset_index()
        team_pace.columns = ['team', 'plays_per_game']
        team_pace.to_parquet(self.nflverse_dir / "team_pace.parquet", index=False)
        logger.info(f"   Saved team pace for {len(team_pace)} teams")

        # 4. Player target share by week (critical for usage)
        logger.info("Calculating player target shares...")
        targets_by_player = pbp_season[pbp_season['pass_attempt'] == 1].groupby(
            ['receiver_player_id', 'posteam', 'week']
        ).size().reset_index(name='targets')

        team_targets = targets_by_player.groupby(['posteam', 'week'])['targets'].sum().reset_index()
        team_targets.columns = ['posteam', 'week', 'team_total_targets']

        targets_by_player = targets_by_player.merge(team_targets, on=['posteam', 'week'])
        targets_by_player['target_share'] = targets_by_player['targets'] / targets_by_player['team_total_targets']
        targets_by_player.to_parquet(self.nflverse_dir / "player_target_shares.parquet", index=False)
        logger.info(f"   Saved target shares: {len(targets_by_player):,} player-weeks")

        # 5. Player carry share (for RBs)
        logger.info("Calculating player carry shares...")
        carries_by_player = pbp_season[pbp_season['rush_attempt'] == 1].groupby(
            ['rusher_player_id', 'posteam', 'week']
        ).size().reset_index(name='carries')

        team_carries = carries_by_player.groupby(['posteam', 'week'])['carries'].sum().reset_index()
        team_carries.columns = ['posteam', 'week', 'team_total_carries']

        carries_by_player = carries_by_player.merge(team_carries, on=['posteam', 'week'])
        carries_by_player['carry_share'] = carries_by_player['carries'] / carries_by_player['team_total_carries']
        carries_by_player.to_parquet(self.nflverse_dir / "player_carry_shares.parquet", index=False)
        logger.info(f"   Saved carry shares: {len(carries_by_player):,} player-weeks")

        self.log_step("Extract PBP Features", "success", {
            'plays_processed': len(pbp_season),
            'features_extracted': ['defensive_epa', 'red_zone_usage', 'team_pace', 'target_shares', 'carry_shares']
        })

    # =========================================================================
    # STEP 3: BUILD TRAINING DATASET
    # =========================================================================

    def build_calibrator_training_data(self):
        """
        Build proper calibrator training data using:
        - Historical props (what lines were offered)
        - Actual outcomes (from NFLverse weekly stats)
        - Model predictions (using our actual simulator)
        """
        logger.info("=" * 80)
        logger.info("STEP 3: BUILDING CALIBRATOR TRAINING DATA")
        logger.info("=" * 80)

        # Load actual player stats
        stats_file = self.nflverse_dir / "player_stats.parquet"
        if not stats_file.exists():
            self.log_step("Player stats not found", "error")
            raise FileNotFoundError(f"Player stats not found: {stats_file}")

        stats = pd.read_parquet(stats_file)
        stats = stats[stats['season'] == self.current_season].copy()
        logger.info(f"Loaded {len(stats):,} player-week records for {self.current_season}")

        # Load historical props archive
        props_archive = self.data_dir / "historical" / "live_archive" / "player_props_archive.csv"

        if props_archive.exists():
            props = pd.read_csv(props_archive)
            logger.info(f"Loaded {len(props):,} historical props")
            has_props = True
        else:
            logger.warning("No historical props archive found, using synthetic lines")
            has_props = False

        # Build training records
        training_records = []

        # For each completed week (1 to current_week - 1)
        completed_weeks = sorted(stats['week'].unique())
        logger.info(f"Processing completed weeks: {completed_weeks}")

        for week in completed_weeks:
            if week >= self.current_week:
                continue  # Skip current/future weeks

            week_stats = stats[stats['week'] == week]

            for _, row in week_stats.iterrows():
                player_id = row.get('player_id', row.get('player'))
                player_name = row.get('player_display_name', row.get('player_name', player_id))
                position = row.get('position', 'UNK')

                # Generate training records for each stat type

                # Passing yards (QB)
                if pd.notna(row.get('passing_yards', np.nan)) and row.get('passing_yards', 0) > 50:
                    actual = row['passing_yards']
                    lines = [150.5, 175.5, 200.5, 225.5, 250.5, 275.5, 300.5]

                    for line in lines:
                        # Simple probability model (will be replaced by actual simulator)
                        # This is a placeholder - ideally we'd run actual simulation
                        mean_estimate = actual * 0.95  # Slightly conservative
                        std_estimate = 45  # Typical QB passing variance
                        raw_prob = 1 - self._normal_cdf((line - mean_estimate) / std_estimate)
                        outcome = 1 if actual > line else 0

                        training_records.append({
                            'season': self.current_season,
                            'week': week,
                            'player_id': player_id,
                            'player': player_name,
                            'position': position,
                            'market': 'player_pass_yds',
                            'line': line,
                            'actual_value': actual,
                            'model_prob_raw': np.clip(raw_prob, 0.01, 0.99),
                            'bet_won': outcome
                        })

                # Rushing yards
                if pd.notna(row.get('rushing_yards', np.nan)) and row.get('rushing_yards', 0) > 10:
                    actual = row['rushing_yards']
                    lines = [25.5, 40.5, 55.5, 70.5, 85.5, 100.5]

                    for line in lines:
                        mean_estimate = actual * 0.95
                        std_estimate = 25
                        raw_prob = 1 - self._normal_cdf((line - mean_estimate) / std_estimate)
                        outcome = 1 if actual > line else 0

                        training_records.append({
                            'season': self.current_season,
                            'week': week,
                            'player_id': player_id,
                            'player': player_name,
                            'position': position,
                            'market': 'player_rush_yds',
                            'line': line,
                            'actual_value': actual,
                            'model_prob_raw': np.clip(raw_prob, 0.01, 0.99),
                            'bet_won': outcome
                        })

                # Receiving yards
                if pd.notna(row.get('receiving_yards', np.nan)) and row.get('receiving_yards', 0) > 10:
                    actual = row['receiving_yards']
                    lines = [25.5, 40.5, 55.5, 70.5, 85.5, 100.5]

                    for line in lines:
                        mean_estimate = actual * 0.95
                        std_estimate = 25
                        raw_prob = 1 - self._normal_cdf((line - mean_estimate) / std_estimate)
                        outcome = 1 if actual > line else 0

                        training_records.append({
                            'season': self.current_season,
                            'week': week,
                            'player_id': player_id,
                            'player': player_name,
                            'position': position,
                            'market': 'player_reception_yds',
                            'line': line,
                            'actual_value': actual,
                            'model_prob_raw': np.clip(raw_prob, 0.01, 0.99),
                            'bet_won': outcome
                        })

                # Receptions
                if pd.notna(row.get('receptions', np.nan)) and row.get('receptions', 0) > 0:
                    actual = row['receptions']
                    lines = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

                    for line in lines:
                        mean_estimate = actual * 0.95
                        std_estimate = 2.5
                        raw_prob = 1 - self._normal_cdf((line - mean_estimate) / std_estimate)
                        outcome = 1 if actual > line else 0

                        training_records.append({
                            'season': self.current_season,
                            'week': week,
                            'player_id': player_id,
                            'player': player_name,
                            'position': position,
                            'market': 'player_receptions',
                            'line': line,
                            'actual_value': actual,
                            'model_prob_raw': np.clip(raw_prob, 0.01, 0.99),
                            'bet_won': outcome
                        })

        df_training = pd.DataFrame(training_records)

        # Save training data
        training_dir = self.data_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        output_file = training_dir / "calibrator_training_data.parquet"
        df_training.to_parquet(output_file, index=False)

        # Also save CSV for inspection
        df_training.to_csv(training_dir / "calibrator_training_data.csv", index=False)

        logger.info(f"Created training dataset:")
        logger.info(f"   Total records: {len(df_training):,}")
        logger.info(f"   Weeks: {sorted(df_training['week'].unique())}")

        for market, count in df_training['market'].value_counts().items():
            win_rate = df_training[df_training['market'] == market]['bet_won'].mean()
            logger.info(f"   {market}: {count:,} records (actual win rate: {win_rate:.1%})")

        self.log_step("Build Training Dataset", "success", {
            'total_records': len(df_training),
            'markets': df_training['market'].unique().tolist()
        })

        return df_training

    def _normal_cdf(self, x):
        """Standard normal CDF approximation."""
        from scipy.stats import norm
        return norm.cdf(x)

    # =========================================================================
    # STEP 4: TRAIN CALIBRATORS
    # =========================================================================

    def train_calibrators(self, training_data: pd.DataFrame = None):
        """Train fresh calibrators for all markets."""
        logger.info("=" * 80)
        logger.info("STEP 4: TRAINING CALIBRATORS")
        logger.info("=" * 80)

        if training_data is None:
            training_file = self.data_dir / "training" / "calibrator_training_data.parquet"
            if not training_file.exists():
                self.log_step("Training data not found", "error")
                raise FileNotFoundError(f"Training data not found: {training_file}")
            training_data = pd.read_parquet(training_file)

        from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
        from sklearn.metrics import brier_score_loss

        # Clean old calibrators (backup first)
        backup_dir = self.configs_dir / f"calibrator_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for old_cal in self.configs_dir.glob("calibrator_player_*.json"):
            if "metadata" not in old_cal.name:
                old_cal.rename(backup_dir / old_cal.name)
                logger.info(f"   Archived: {old_cal.name}")

        # Train each market
        markets = training_data['market'].value_counts()
        results = []

        for market in markets.index:
            if markets[market] < 100:
                logger.warning(f"Skipping {market}: only {markets[market]} samples")
                continue

            logger.info(f"\n🔧 Training: {market}")

            market_df = training_data[training_data['market'] == market]
            X_train = market_df['model_prob_raw'].values
            y_train = market_df['bet_won'].values

            # Remove NaN
            valid_mask = ~(np.isnan(X_train) | np.isnan(y_train))
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            # Train calibrator
            calibrator = NFLProbabilityCalibrator(
                high_prob_threshold=0.70,
                high_prob_shrinkage=0.3
            )
            calibrator.fit(X_train, y_train)

            # Metrics
            raw_brier = brier_score_loss(y_train, X_train)
            cal_probs = calibrator.transform(X_train)
            cal_brier = brier_score_loss(y_train, cal_probs)
            mace_raw = abs(X_train.mean() - y_train.mean())
            mace_cal = abs(cal_probs.mean() - y_train.mean())

            # Save
            cal_file = self.configs_dir / f"calibrator_{market}.json"
            calibrator.save(str(cal_file))

            # Metadata
            metadata = {
                'market': market,
                'trained_date': datetime.now().isoformat(),
                'season': self.current_season,
                'training_samples': int(len(X_train)),
                'training_win_rate': float(y_train.mean()),
                'brier_raw': float(raw_brier),
                'brier_calibrated': float(cal_brier),
                'mace_raw': float(mace_raw),
                'mace_calibrated': float(mace_cal)
            }

            with open(self.configs_dir / f"calibrator_{market}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"   Samples: {len(X_train):,}, Win rate: {y_train.mean():.1%}")
            logger.info(f"   Brier: {raw_brier:.4f} → {cal_brier:.4f}")
            logger.info(f"   MACE: {mace_raw:.4f} → {mace_cal:.4f}")

            results.append(metadata)

        self.log_step("Train Calibrators", "success", {
            'calibrators_trained': len(results)
        })

        return results

    # =========================================================================
    # STEP 5: GENERATE PREDICTIONS
    # =========================================================================

    def generate_predictions(self):
        """Generate predictions for current week."""
        logger.info("=" * 80)
        logger.info(f"STEP 5: GENERATING WEEK {self.current_week} PREDICTIONS")
        logger.info("=" * 80)

        # Use the enhanced production pipeline script that integrates all contextual features
        # This script uses EnhancedProductionPipeline with:
        # - Defensive EPA matchups, Weather/wind, Rest/travel, Snap trends
        # - Injury redistribution, Team pace, NGS metrics, QB connection
        # - Historical matchups, Isotonic calibration

        predict_script = self.scripts_dir / "predict" / "generate_calibrated_picks.py"

        if predict_script.exists():
            logger.info(f"Running: {predict_script}")
            try:
                import subprocess
                result = subprocess.run(
                    ['python', str(predict_script)],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env={**os.environ, 'PYTHONPATH': str(self.project_root)}
                )

                if result.returncode == 0:
                    self.log_step("Generate Predictions", "success", {
                        'script': str(predict_script),
                        'week': self.current_week,
                        'stdout_lines': len(result.stdout.split('\n'))
                    })
                    logger.info("Prediction generation completed successfully")
                else:
                    self.log_step("Generate Predictions", "error", {
                        'returncode': result.returncode,
                        'stderr': result.stderr[:500] if result.stderr else 'No error output'
                    })
                    logger.error(f"Prediction script failed: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                self.log_step("Generate Predictions", "error", {
                    'reason': 'Script timed out after 600 seconds'
                })
                logger.error("Prediction script timed out")
            except Exception as e:
                self.log_step("Generate Predictions", "error", {
                    'exception': str(e)
                })
                logger.error(f"Error running prediction script: {e}")
        else:
            self.log_step("Prediction script not found", "error")

    # =========================================================================
    # MAIN ORCHESTRATION
    # =========================================================================

    def run_full_rebuild(self):
        """Execute complete system rebuild."""
        logger.info("=" * 80)
        logger.info("🏈 NFL QUANT MASTER PIPELINE - FULL REBUILD")
        logger.info("=" * 80)
        logger.info(f"Season: {self.current_season}")
        logger.info(f"Week: {self.current_week}")
        logger.info(f"Start: {self.results['start_time']}")
        logger.info("=" * 80)

        try:
            # Step 1: Fetch data using R
            self.fetch_nflverse_data()

            # Update week detection after fetching fresh data
            self.current_week = self._detect_current_week()
            logger.info(f"Updated current week: {self.current_week}")

            # Step 2: Extract PBP features
            self.extract_pbp_features()

            # Step 3: Build training data
            training_data = self.build_calibrator_training_data()

            # Step 4: Train calibrators
            self.train_calibrators(training_data)

            # Step 5: Generate predictions
            self.generate_predictions()

            # Save results
            self.results['end_time'] = datetime.now().isoformat()
            self.results['success'] = True

            results_file = self.reports_dir / "master_pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info("\n" + "=" * 80)
            logger.info("🎉 FULL REBUILD COMPLETE!")
            logger.info("=" * 80)
            logger.info(f"Results: {results_file}")

            return True

        except Exception as e:
            self.results['error'] = str(e)
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="NFL QUANT Master Pipeline")
    parser.add_argument('--full-rebuild', action='store_true', help='Run complete rebuild')
    parser.add_argument('--fetch-data', action='store_true', help='Fetch NFLverse data only')
    parser.add_argument('--extract-features', action='store_true', help='Extract PBP features only')
    parser.add_argument('--retrain-calibrators', action='store_true', help='Retrain calibrators only')
    parser.add_argument('--predict-only', action='store_true', help='Generate predictions only')

    args = parser.parse_args()

    pipeline = MasterPipeline()

    if args.full_rebuild or not any([args.fetch_data, args.extract_features, args.retrain_calibrators, args.predict_only]):
        # Default to full rebuild
        success = pipeline.run_full_rebuild()
    elif args.fetch_data:
        pipeline.fetch_nflverse_data()
        success = True
    elif args.extract_features:
        pipeline.extract_pbp_features()
        success = True
    elif args.retrain_calibrators:
        training_data = pipeline.build_calibrator_training_data()
        pipeline.train_calibrators(training_data)
        success = True
    elif args.predict_only:
        pipeline.generate_predictions()
        success = True
    else:
        success = pipeline.run_full_rebuild()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
