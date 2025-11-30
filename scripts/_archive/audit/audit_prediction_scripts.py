#!/usr/bin/env python3
"""
Update Prediction Generation Scripts to Use Unified Data Loading

This script audits and updates all prediction generation scripts to:
1. Use unified trailing stats (2025 + historical)
2. Use stats_loader.py for consistent data loading
3. Prioritize 2025 data for current season predictions
4. Use historical data as fallback
"""

import re
from pathlib import Path
from typing import List, Dict

def audit_prediction_scripts() -> Dict:
    """Audit prediction generation scripts for data usage."""
    scripts_dir = Path('scripts/predict')
    
    findings = {
        'uses_unified_trailing_stats': [],
        'uses_stats_loader': [],
        'hardcoded_data_paths': [],
        'missing_historical_fallback': [],
        'wrong_2025_priority': []
    }
    
    for script_file in scripts_dir.glob('*.py'):
        if script_file.name == '__init__.py':
            continue
        
        try:
            with open(script_file) as f:
                content = f.read()
        except Exception:
            continue
        
        # Check for unified trailing stats usage
        if 'historical_player_stats.json' in content or 'create_unified_historical_player_stats' in content:
            findings['uses_unified_trailing_stats'].append(str(script_file))
        
        # Check for stats_loader usage
        if 'from nfl_quant.data.stats_loader import' in content or 'stats_loader' in content:
            findings['uses_stats_loader'].append(str(script_file))
        
        # Check for hardcoded data paths
        hardcoded_patterns = [
            r"data/sleeper_stats/stats_week\d+_2025\.csv",
            r"data/nflverse/pbp_2025\.parquet",
            r"data/nflverse_cache/stats_player_week_2025\.csv",
        ]
        for pattern in hardcoded_patterns:
            if re.search(pattern, content):
                findings['hardcoded_data_paths'].append({
                    'file': str(script_file),
                    'pattern': pattern
                })
        
        # Check for missing historical fallback
        if '2025' in content and 'sleeper' in content.lower():
            if '2024' not in content and 'historical' not in content.lower():
                findings['missing_historical_fallback'].append(str(script_file))
    
    return findings


def update_load_trailing_stats_function():
    """Update load_trailing_stats to use unified historical_player_stats.json."""
    script_path = Path('scripts/predict/generate_model_predictions.py')
    
    new_function = '''def load_trailing_stats():
    """
    Load unified trailing stats combining 2025 Sleeper + historical NFLverse/Sleeper.
    
    Priority:
    1. historical_player_stats.json (unified, includes 2025 + historical)
    2. week_specific_trailing_stats.json (if exists)
    3. player_database.json (fallback)
    """
    # Primary: Unified historical stats (2025 + historical)
    unified_stats_file = Path('data/historical_player_stats.json')
    trailing_stats = {}
    
    if unified_stats_file.exists():
        import json
        with open(unified_stats_file) as f:
            unified_db = json.load(f)
        
        # Convert unified format to trailing_stats format
        # Unified uses "Player Name" as key
        # trailing_stats uses "Player Name_week{week}" as key
        # We'll create entries for current week (will be overridden by caller if needed)
        default_week = 10  # Default, will be overridden by caller
        
        for player_name, stats in unified_db.items():
            key = f"{player_name}_week{default_week}"
            trailing_stats[key] = {
                'trailing_snap_share': stats.get('trailing_snap_share'),
                'trailing_target_share': stats.get('trailing_target_share'),
                'trailing_carry_share': stats.get('trailing_carry_share'),
                'trailing_yards_per_opportunity': stats.get('trailing_yards_per_opportunity'),
                'trailing_td_rate': stats.get('trailing_td_rate'),
                'team': stats.get('team'),
                'position': stats.get('position'),
                'seasons': stats.get('seasons', []),
                'data_sources': stats.get('data_sources', []),
            }
        
        logger.info(f"   ✅ Loaded {len(trailing_stats)} players from unified historical_player_stats.json")
        logger.info(f"      Data sources: {set(s for stats in unified_db.values() for s in stats.get('data_sources', []))}")
        logger.info(f"      Seasons: {sorted(set(s for stats in unified_db.values() for s in stats.get('seasons', [])))}")
    
    # Fallback: week_specific_trailing_stats.json
    if not trailing_stats:
        week_stats_file = Path('data/week_specific_trailing_stats.json')
        if week_stats_file.exists():
            import json
            with open(week_stats_file) as f:
                trailing_stats = json.load(f)
            logger.info(f"   ✅ Loaded {len(trailing_stats)} player-week combinations from week_specific_trailing_stats.json")
    
    # Final fallback: player_database.json
    if not trailing_stats:
        player_db_file = Path('data/player_database.json')
        if player_db_file.exists():
            import json
            with open(player_db_file) as f:
                player_db = json.load(f)
            
            default_week = 10
            for player_name, stats in player_db.items():
                key = f"{player_name}_week{default_week}"
                trailing_stats[key] = {
                    'trailing_snap_share': stats.get('trailing_snap_share'),
                    'trailing_target_share': stats.get('trailing_target_share'),
                    'trailing_carry_share': stats.get('trailing_carry_share'),
                    'trailing_yards_per_opportunity': stats.get('trailing_yards_per_opportunity'),
                    'trailing_td_rate': stats.get('trailing_td_rate'),
                    'team': stats.get('team'),
                    'position': stats.get('position'),
                }
            
            logger.info(f"   ✅ Loaded {len(trailing_stats)} players from player_database.json (fallback)")
    
    if not trailing_stats:
        logger.warning(f"   ⚠️  No trailing stats found in any file")
        logger.warning(f"   Expected: data/historical_player_stats.json")
        logger.warning(f"   Run: python scripts/utils/create_unified_historical_player_stats.py")
    
    return trailing_stats'''
    
    return new_function


def main():
    """Run audit and generate update recommendations."""
    print("="*80)
    print("PREDICTION GENERATION SCRIPTS AUDIT")
    print("="*80)
    print()
    
    findings = audit_prediction_scripts()
    
    print("📊 Audit Results:")
    print()
    print(f"✅ Uses Unified Trailing Stats: {len(findings['uses_unified_trailing_stats'])} scripts")
    for script in findings['uses_unified_trailing_stats']:
        print(f"   - {script}")
    
    print()
    print(f"✅ Uses stats_loader: {len(findings['uses_stats_loader'])} scripts")
    for script in findings['uses_stats_loader']:
        print(f"   - {script}")
    
    print()
    print(f"⚠️  Hardcoded Data Paths: {len(findings['hardcoded_data_paths'])} instances")
    for item in findings['hardcoded_data_paths']:
        print(f"   - {item['file']}: {item['pattern']}")
    
    print()
    print(f"⚠️  Missing Historical Fallback: {len(findings['missing_historical_fallback'])} scripts")
    for script in findings['missing_historical_fallback']:
        print(f"   - {script}")
    
    print()
    print("="*80)
    print("RECOMMENDED UPDATES")
    print("="*80)
    print()
    print("1. Update load_trailing_stats() in generate_model_predictions.py")
    print("   to use unified historical_player_stats.json")
    print()
    print("2. Replace hardcoded data paths with stats_loader.py")
    print()
    print("3. Ensure all scripts prioritize 2025 data with historical fallback")
    print()
    
    # Generate updated function
    updated_function = update_load_trailing_stats_function()
    print("="*80)
    print("UPDATED load_trailing_stats() FUNCTION")
    print("="*80)
    print()
    print(updated_function)


if __name__ == '__main__':
    main()

