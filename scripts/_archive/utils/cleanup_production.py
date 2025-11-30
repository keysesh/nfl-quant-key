#!/usr/bin/env python3
"""
Production Cleanup Script

Removes unnecessary files while keeping production essentials:
- Keeps: Current week files, essential CSVs, active dashboards
- Removes: Old backups, archived files, duplicate dashboards, stale data
"""

import os
from pathlib import Path
import shutil

def cleanup_reports():
    """Clean up reports directory."""
    reports_dir = Path('reports')

    # Files to KEEP (production essentials)
    keep_files = {
        'unified_betting_recommendations.csv',
        'unified_betting_recommendations_v2.csv',
        'CURRENT_WEEK_PLAYER_PROPS.csv',
        'FINAL_RECOMMENDATIONS.csv',
        'simplified_dashboard.html',
        'BACKTEST_WEEKS_1_8_VALIDATION.csv',  # Validation data
        'calibration_validation_plot.png',  # Validation output
        'data_integrity_validation_report.txt',  # Validation output
    }

    # Directories to REMOVE (backups/archives)
    remove_dirs = [
        'reports/backups',
        'reports/backup_before_clean',
        'reports/archive_stale_20251029',
        'reports/archive_old_dashboards',
    ]

    # Files to REMOVE (old/stale)
    remove_patterns = [
        'recommended_bets*.csv',
        'all_prop_recommendations.csv',
        'alt_line_recommendations.csv',
        'unmatched_players.csv',
        'tonights_picks_only.csv',
        'all_betting_opportunities.csv',
        'week8_*.csv',
        'week8_*.html',
        'week8_*.txt',
        'week8_*.json',
        'week9_dashboard.html',  # Old dashboard
        'clean_dashboard_auto.html',  # Old dashboard
        'WEEK8_*.csv',
        'WEEK9_MODEL_VS_DRAFTKINGS.csv',
        'betting_backtest_*.csv',
        'statistical_analysis.csv',
        'calibration_method_comparison.csv',
        'calibration_cv_results.csv',
        'sim_*.json',  # Old simulation files
        '*.log',  # Log files
        'framework_output.log',
        'tracking_*.txt',
        'EXPERT_SYSTEM_AUDIT.md',
        'TONIGHT_BETTING_CARD.md',
    ]

    removed_count = 0

    # Remove directories
    for dir_path in remove_dirs:
        dir_obj = Path(dir_path)
        if dir_obj.exists():
            print(f"🗑️  Removing directory: {dir_path}")
            shutil.rmtree(dir_path)
            removed_count += 1

    # Remove files matching patterns
    for pattern in remove_patterns:
        for file_path in reports_dir.glob(pattern):
            if file_path.name not in keep_files:
                print(f"🗑️  Removing file: {file_path.name}")
                file_path.unlink()
                removed_count += 1

    # Remove old PNG files (keep only latest validation plot)
    for png_file in reports_dir.glob('*.png'):
        if png_file.name not in ['calibration_validation_plot.png']:
            print(f"🗑️  Removing old plot: {png_file.name}")
            png_file.unlink()
            removed_count += 1

    return removed_count


def cleanup_data():
    """Clean up data directory (keep essentials only)."""
    data_dir = Path('data')

    # Directories to KEEP
    keep_dirs = {
        'models',
        'historical',  # Keep historical data
        'nfl_player_props_draftkings.csv',  # Current odds
    }

    removed_count = 0

    # Remove old backup files
    for backup_file in data_dir.rglob('*.backup'):
        print(f"🗑️  Removing backup: {backup_file}")
        backup_file.unlink()
        removed_count += 1

    return removed_count


def cleanup_configs():
    """Clean up configs directory."""
    configs_dir = Path('configs')

    # Files to KEEP
    keep_files = {
        'bankroll_config.json',
        'risk_modes.json',
        'calibrator.json',
    }

    removed_count = 0

    # Remove old calibrator backups
    for config_file in configs_dir.glob('*.json'):
        if config_file.name not in keep_files:
            if 'backup' in config_file.name.lower() or 'old' in config_file.name.lower():
                print(f"🗑️  Removing old config: {config_file.name}")
                config_file.unlink()
                removed_count += 1

    return removed_count


def main():
    """Run cleanup."""
    print("=" * 80)
    print("🧹 PRODUCTION CLEANUP")
    print("=" * 80)
    print()

    total_removed = 0

    print("📊 Cleaning reports directory...")
    removed = cleanup_reports()
    total_removed += removed
    print(f"   Removed {removed} items\n")

    print("📊 Cleaning data directory...")
    removed = cleanup_data()
    total_removed += removed
    print(f"   Removed {removed} items\n")

    print("📊 Cleaning configs directory...")
    removed = cleanup_configs()
    total_removed += removed
    print(f"   Removed {removed} items\n")

    print("=" * 80)
    print(f"✅ CLEANUP COMPLETE: Removed {total_removed} items")
    print("=" * 80)
    print()
    print("✅ Kept production essentials:")
    print("   - unified_betting_recommendations.csv")
    print("   - simplified_dashboard.html")
    print("   - CURRENT_WEEK_PLAYER_PROPS.csv")
    print("   - FINAL_RECOMMENDATIONS.csv")
    print("   - configs/calibrator.json")
    print("   - configs/bankroll_config.json")


if __name__ == '__main__':
    main()
