#!/usr/bin/env python3
"""
NFL QUANT System Cleanup

This script:
1. Archives duplicate/old calibrator files
2. Consolidates data directories
3. Removes deprecated code
4. Creates a clean, unified system

Run with --dry-run first to see what will be changed.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent


def archive_old_calibrators(dry_run=False):
    """Move duplicate/old calibrator files to archive."""
    configs_dir = PROJECT_ROOT / "configs"
    archive_dir = configs_dir / f"calibrator_archive_{datetime.now().strftime('%Y%m%d')}"

    # Files to archive (keep only the main .json versions)
    patterns_to_archive = [
        "*_full.joblib",  # Old full versions
        "*_improved.joblib",  # Old improved versions
        "*_v2.json",  # Old v2 versions
        "*_backup.json",  # Backups
        "*_pre_fix.json",  # Pre-fix versions
        "calibrator_player-*",  # Hyphenated versions (old naming)
    ]

    files_to_archive = []
    for pattern in patterns_to_archive:
        files_to_archive.extend(configs_dir.glob(pattern))

    # Filter to only calibrator files
    files_to_archive = [f for f in files_to_archive if "calibrator" in f.name]

    if not files_to_archive:
        print("No old calibrator files to archive")
        return

    print(f"\n📁 Archiving {len(files_to_archive)} old calibrator files to:")
    print(f"   {archive_dir}")

    for f in sorted(files_to_archive):
        print(f"   - {f.name}")

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in files_to_archive:
            shutil.move(str(f), str(archive_dir / f.name))
        print(f"\n✅ Archived {len(files_to_archive)} files")
    else:
        print("\n⚠️  DRY RUN - no files moved")


def consolidate_data_directories(dry_run=False):
    """Consolidate scattered data into single locations."""
    data_dir = PROJECT_ROOT / "data"

    # Remove duplicate cache directories
    duplicate_caches = [
        data_dir / "nflverse_cache",  # Old cache location
    ]

    print("\n📁 Checking for duplicate data directories...")

    for cache_dir in duplicate_caches:
        if cache_dir.exists():
            # Count files
            files = list(cache_dir.glob("*"))
            print(f"   Found: {cache_dir.name} ({len(files)} files)")

            if not dry_run:
                # Don't delete yet, just report
                print(f"   ⚠️  Consider manually reviewing and archiving: {cache_dir}")
            else:
                print(f"   ⚠️  Would need manual review: {cache_dir}")


def remove_deprecated_venvs(dry_run=False):
    """Remove old virtual environments."""
    venvs_to_remove = [
        PROJECT_ROOT / "venv311",
    ]

    print("\n📁 Checking for deprecated virtual environments...")

    for venv_dir in venvs_to_remove:
        if venv_dir.exists():
            size_mb = sum(f.stat().st_size for f in venv_dir.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"   Found: {venv_dir.name} ({size_mb:.1f} MB)")

            if not dry_run:
                print(f"   ⚠️  Would remove {venv_dir} - run without --dry-run to execute")
                # Don't actually delete - too dangerous
            else:
                print(f"   ⚠️  DRY RUN - would delete {venv_dir}")


def clean_backup_directories(dry_run=False):
    """Clean up backup directories."""
    configs_dir = PROJECT_ROOT / "configs"

    backup_dirs = [
        configs_dir / "backup_pre_phase1",
        configs_dir / "backup_before_improved",
        configs_dir / "calibrator_backup_rebuild",
    ]

    print("\n📁 Checking for backup directories to consolidate...")

    for backup_dir in backup_dirs:
        if backup_dir.exists():
            files = list(backup_dir.glob("*"))
            print(f"   Found: {backup_dir.name} ({len(files)} files)")

    if not dry_run:
        # Consolidate all backups into one archive
        master_backup = configs_dir / f"all_backups_pre_{datetime.now().strftime('%Y%m%d')}"
        if any(d.exists() for d in backup_dirs):
            master_backup.mkdir(parents=True, exist_ok=True)
            print(f"\n   Consolidating to: {master_backup}")

            for backup_dir in backup_dirs:
                if backup_dir.exists():
                    # Move contents
                    for f in backup_dir.glob("*"):
                        dest = master_backup / f"{backup_dir.name}_{f.name}"
                        shutil.copy2(str(f), str(dest))
                    # Remove old directory
                    shutil.rmtree(str(backup_dir))
                    print(f"   ✅ Consolidated and removed: {backup_dir.name}")


def generate_current_state_report():
    """Generate a report of current system state."""
    print("\n" + "=" * 80)
    print("CURRENT SYSTEM STATE REPORT")
    print("=" * 80)

    # Check calibrators
    configs_dir = PROJECT_ROOT / "configs"
    calibrators = list(configs_dir.glob("calibrator_player_*.json"))
    calibrators = [c for c in calibrators if "metadata" not in c.name]

    print("\n📊 ACTIVE CALIBRATORS:")
    for cal in sorted(calibrators):
        # Load metadata if exists
        metadata_file = cal.parent / f"{cal.stem}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                meta = json.load(f)
            print(f"   {cal.stem}:")
            print(f"      Samples: {meta.get('training_samples', 'N/A'):,}")
            print(f"      Win Rate: {meta.get('training_win_rate', 0):.1%}")
            print(f"      MACE: {meta.get('mace_calibrated', 0):.4f}")
            print(f"      Trained: {meta.get('trained_date', 'N/A')[:10]}")
        else:
            print(f"   {cal.stem}: (no metadata)")

    # Check data files
    nflverse_dir = PROJECT_ROOT / "data" / "nflverse"
    print("\n📊 NFLVERSE DATA FILES:")
    key_files = ['pbp.parquet', 'player_stats.parquet', 'schedules.parquet', 'rosters.parquet']
    for fname in key_files:
        fpath = nflverse_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            print(f"   {fname}: {size_mb:.1f} MB (updated: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"   {fname}: ❌ MISSING")

    # Check PBP features
    print("\n📊 PBP EXTRACTED FEATURES:")
    pbp_features = [
        'team_defensive_epa.parquet',
        'player_target_shares.parquet',
        'player_carry_shares.parquet',
        'player_red_zone_usage.parquet',
        'team_pace.parquet'
    ]
    for fname in pbp_features:
        fpath = nflverse_dir / fname
        if fpath.exists():
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            print(f"   ✅ {fname} (updated: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"   ❌ {fname} MISSING")

    # Check models
    models_dir = PROJECT_ROOT / "data" / "models"
    print("\n📊 ML MODELS:")
    if models_dir.exists():
        for model_file in sorted(models_dir.glob("*.joblib")):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"   {model_file.name}: {size_mb:.1f} MB (updated: {mtime.strftime('%Y-%m-%d')})")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Clean up NFL QUANT system")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--report-only', action='store_true', help='Only generate state report')

    args = parser.parse_args()

    print("🧹 NFL QUANT SYSTEM CLEANUP")
    print("=" * 80)

    if args.report_only:
        generate_current_state_report()
        return

    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made\n")

    # Run cleanup steps
    archive_old_calibrators(args.dry_run)
    consolidate_data_directories(args.dry_run)
    remove_deprecated_venvs(args.dry_run)
    clean_backup_directories(args.dry_run)

    # Generate final report
    generate_current_state_report()

    print("\n✅ Cleanup complete!")


if __name__ == "__main__":
    main()
