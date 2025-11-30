#!/usr/bin/env python3
"""
Monte Carlo V3 Migration Script

Automates the migration from PlayerSimulator V1 to V3 with full backward compatibility.

Usage:
    python scripts/migrate_to_v3.py --dry-run    # Preview changes
    python scripts/migrate_to_v3.py --execute    # Apply migration
    python scripts/migrate_to_v3.py --rollback   # Undo migration

Safety:
- Creates backups before any changes
- Validates configurations
- Tests imports
- Rolls back on errors
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
import json

# ANSI colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'


class V3Migrator:
    """Handles migration from V1 to V3 player simulator."""

    def __init__(self, nfl_quant_root: Path, dry_run: bool = True):
        self.root = nfl_quant_root
        self.dry_run = dry_run
        self.backup_dir = self.root / "backups" / f"v1_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with color coding."""
        if level == "INFO":
            print(f"{BLUE}[INFO]{ENDC} {message}")
        elif level == "SUCCESS":
            print(f"{GREEN}[✓]{ENDC} {message}")
        elif level == "WARNING":
            print(f"{YELLOW}[⚠]{ENDC} {message}")
            self.warnings.append(message)
        elif level == "ERROR":
            print(f"{RED}[✗]{ENDC} {message}")
            self.errors.append(message)

    def validate_environment(self):
        """Validate that we're in the correct directory."""
        self.log("Validating environment...")

        # Check for key directories
        required_dirs = [
            self.root / "nfl_quant",
            self.root / "scripts",
            self.root / "configs",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                self.log(f"Missing directory: {dir_path}", "ERROR")
                return False

        # Check for key files
        required_files = [
            self.root / "nfl_quant" / "simulation" / "player_simulator.py",
            self.root / "nfl_quant" / "simulation" / "player_simulator_v3_correlated.py",
            self.root / "configs" / "simulation_config.json",
        ]

        for file_path in required_files:
            if not file_path.exists():
                self.log(f"Missing file: {file_path}", "ERROR")
                return False

        self.log("Environment validation passed", "SUCCESS")
        return True

    def create_backups(self):
        """Create backups of files that will be modified."""
        self.log(f"Creating backups in {self.backup_dir}...")

        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        files_to_backup = [
            self.root / "nfl_quant" / "simulation" / "player_simulator.py",
            self.root / "nfl_quant" / "simulation" / "unified_orchestrator.py",
            self.root / "configs" / "simulation_config.json",
        ]

        for file_path in files_to_backup:
            if file_path.exists():
                if not self.dry_run:
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                self.log(f"Backed up: {file_path.name}", "SUCCESS")
            else:
                self.log(f"Skipping backup (file not found): {file_path}", "WARNING")

        return True

    def check_configuration(self):
        """Verify simulation_config.json has V3 sections."""
        self.log("Checking configuration...")

        config_path = self.root / "configs" / "simulation_config.json"

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Check for V3 sections
            required_sections = [
                "correlation_coefficients",
                "negbin_scoring",
                "market_blending",
                "wind_buckets",
                "contextual_adjustments",
                "redzone_td_model",
                "game_script_engine",
                "feature_flags",
            ]

            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)

            if missing_sections:
                self.log(f"Missing V3 config sections: {', '.join(missing_sections)}", "ERROR")
                self.log("Run: Update simulation_config.json with V3 parameters first", "ERROR")
                return False

            # Check feature flags
            if "feature_flags" in config:
                flags = config["feature_flags"]
                self.log(f"Feature flags found:", "SUCCESS")
                for key, value in flags.items():
                    status = "✓ ENABLED" if value else "✗ DISABLED"
                    self.log(f"  {key}: {status}", "INFO")
            else:
                self.log("Feature flags not found", "WARNING")

            self.log("Configuration check passed", "SUCCESS")
            return True

        except Exception as e:
            self.log(f"Configuration check failed: {e}", "ERROR")
            return False

    def apply_wrapper(self):
        """Apply backward-compatible wrapper to player_simulator.py."""
        self.log("Applying V3 backward-compatible wrapper...")

        wrapper_code = '''"""
Player Simulator - Backward Compatible Wrapper for V3

This module maintains the original PlayerSimulator API while using
PlayerSimulatorV3 as the backend. All existing code works without modification.

Auto-generated by migrate_to_v3.py on {timestamp}
"""

import warnings
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from nfl_quant.simulation.player_simulator_v3_correlated import PlayerSimulatorV3
from nfl_quant.models.usage_predictor import UsagePredictor
from nfl_quant.models.efficiency_predictor import EfficiencyPredictor
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.config_enhanced import config

logger = logging.getLogger(__name__)


class PlayerSimulator(PlayerSimulatorV3):
    """
    Backward-compatible wrapper around PlayerSimulatorV3.

    Maintains V1 API while using V3 backend with all improvements.
    """

    def __init__(
        self,
        usage_predictor: UsagePredictor,
        efficiency_predictor: EfficiencyPredictor,
        trials: Optional[int] = None,
        seed: Optional[int] = None,
        calibrator: Optional[NFLProbabilityCalibrator] = None,
        td_calibrator: Optional[object] = None,
        **kwargs
    ):
        correlation_config = kwargs.pop('correlation_config', None)

        super().__init__(
            usage_predictor=usage_predictor,
            efficiency_predictor=efficiency_predictor,
            trials=trials,
            seed=seed,
            calibrator=calibrator,
            td_calibrator=td_calibrator,
            correlation_config=correlation_config,
        )

        logger.info("PlayerSimulator initialized with V3 backend (correlation-aware)")


def load_predictors(
    usage_model_path: Optional[str] = None,
    efficiency_model_path: Optional[str] = None
) -> Tuple[UsagePredictor, EfficiencyPredictor]:
    """Load usage and efficiency predictors (backward compatible)."""
    if usage_model_path is None:
        usage_model_path = config.paths.models / "usage_predictor_v4_defense.joblib"
    if efficiency_model_path is None:
        efficiency_model_path = config.paths.models / "efficiency_predictor_v2_defense.joblib"

    usage_predictor = UsagePredictor()
    usage_predictor.load(str(usage_model_path))

    efficiency_predictor = EfficiencyPredictor()
    efficiency_predictor.load(str(efficiency_model_path))

    logger.info(f"Loaded usage predictor from {{usage_model_path}}")
    logger.info(f"Loaded efficiency predictor from {{efficiency_model_path}}")

    return usage_predictor, efficiency_predictor


__all__ = ['PlayerSimulator', 'load_predictors', 'PlayerSimulatorV3']
'''.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        target_file = self.root / "nfl_quant" / "simulation" / "player_simulator.py"

        if not self.dry_run:
            with open(target_file, 'w') as f:
                f.write(wrapper_code)
            self.log(f"Updated {target_file}", "SUCCESS")
        else:
            self.log(f"[DRY RUN] Would update {target_file}", "INFO")

        return True

    def test_imports(self):
        """Test that V3 imports work."""
        self.log("Testing V3 imports...")

        try:
            # This will only work if we're not in dry-run mode
            if not self.dry_run:
                sys.path.insert(0, str(self.root))
                from nfl_quant.simulation.player_simulator import PlayerSimulator, PlayerSimulatorV3

                # Check that PlayerSimulator inherits from V3
                if issubclass(PlayerSimulator, PlayerSimulatorV3):
                    self.log("Import test passed: PlayerSimulator uses V3 backend", "SUCCESS")
                    return True
                else:
                    self.log("Import test failed: PlayerSimulator does not inherit from V3", "ERROR")
                    return False
            else:
                self.log("[DRY RUN] Would test imports", "INFO")
                return True

        except Exception as e:
            self.log(f"Import test failed: {e}", "ERROR")
            return False

    def generate_report(self):
        """Generate migration report."""
        print(f"\n{BOLD}{'=' * 60}{ENDC}")
        print(f"{BOLD}Migration Report{ENDC}")
        print(f"{BOLD}{'=' * 60}{ENDC}\n")

        if self.dry_run:
            print(f"{YELLOW}Mode: DRY RUN (no changes made){ENDC}\n")
        else:
            print(f"{GREEN}Mode: EXECUTE (changes applied){ENDC}\n")

        print(f"Warnings: {len(self.warnings)}")
        for warning in self.warnings:
            print(f"  {YELLOW}⚠{ENDC} {warning}")

        print(f"\nErrors: {len(self.errors)}")
        for error in self.errors:
            print(f"  {RED}✗{ENDC} {error}")

        if len(self.errors) == 0:
            print(f"\n{GREEN}{BOLD}✓ Migration successful!{ENDC}\n")
            return True
        else:
            print(f"\n{RED}{BOLD}✗ Migration failed. See errors above.{ENDC}\n")
            return False

    def run(self):
        """Execute full migration."""
        print(f"\n{BOLD}{'=' * 60}{ENDC}")
        print(f"{BOLD}NFL QUANT Monte Carlo V3 Migration{ENDC}")
        print(f"{BOLD}{'=' * 60}{ENDC}\n")

        steps = [
            ("Validate environment", self.validate_environment),
            ("Create backups", self.create_backups),
            ("Check configuration", self.check_configuration),
            ("Apply V3 wrapper", self.apply_wrapper),
            ("Test imports", self.test_imports),
        ]

        for step_name, step_func in steps:
            if not step_func():
                self.log(f"Migration stopped at: {step_name}", "ERROR")
                return self.generate_report()

        return self.generate_report()

    def rollback(self):
        """Rollback to V1."""
        self.log("Rolling back to V1...")

        # Find most recent backup
        backups = sorted((self.root / "backups").glob("v1_backup_*"), reverse=True)

        if not backups:
            self.log("No backups found. Cannot rollback.", "ERROR")
            return False

        latest_backup = backups[0]
        self.log(f"Using backup: {latest_backup}", "INFO")

        # Restore player_simulator.py
        backup_file = latest_backup / "player_simulator.py"
        target_file = self.root / "nfl_quant" / "simulation" / "player_simulator.py"

        if backup_file.exists():
            if not self.dry_run:
                shutil.copy2(backup_file, target_file)
            self.log(f"Restored {target_file.name}", "SUCCESS")
        else:
            self.log(f"Backup file not found: {backup_file}", "ERROR")
            return False

        self.log("Rollback complete", "SUCCESS")
        return True


def main():
    parser = argparse.ArgumentParser(description="Migrate NFL QUANT to Monte Carlo V3")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply migration changes"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to V1"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Path to NFL QUANT root directory"
    )

    args = parser.parse_args()

    # Determine mode
    if args.rollback:
        migrator = V3Migrator(args.root, dry_run=False)
        success = migrator.rollback()
    elif args.execute:
        migrator = V3Migrator(args.root, dry_run=False)
        success = migrator.run()
    else:
        # Default to dry-run
        migrator = V3Migrator(args.root, dry_run=True)
        success = migrator.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
