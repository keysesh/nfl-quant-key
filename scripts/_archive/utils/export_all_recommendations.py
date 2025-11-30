"""
Export unified betting recommendations (game bets + player props).

Combines game-level spread/total bets with player prop projections
into a single comprehensive betting sheet.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.season_utils import get_current_week

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_unified_recommendations(week: int = None) -> None:
    """Export all recommendations to a single comprehensive file.

    Args:
        week: NFL week number. If None, auto-detects current week.
    """
    if week is None:
        week = get_current_week()

    logger.info("="*80)
    logger.info(f"EXPORTING UNIFIED BETTING RECOMMENDATIONS - WEEK {week}")
    logger.info("="*80)

    # Load bankroll config
    config_path = Path('configs/bankroll_config.json')
    if not config_path.exists():
        raise FileNotFoundError("Bankroll configuration not found")

    with open(config_path) as f:
        config = json.load(f)

    # Load game bets
    game_bets_path = Path('reports/FINAL_RECOMMENDATIONS.csv')
    if game_bets_path.exists():
        game_bets_df = pd.read_csv(game_bets_path)
        logger.info(f"✅ Loaded {len(game_bets_df)} game bets")
    else:
        game_bets_df = pd.DataFrame()
        logger.warning("⚠️  No game bets found")

    # Load player props
    props_path = Path(f'reports/PLAYER_PROPS_WEEK{week}.csv')
    if props_path.exists():
        props_df = pd.read_csv(props_path)
        logger.info(f"✅ Loaded {len(props_df)} player prop projections")
    else:
        props_df = pd.DataFrame()
        logger.warning("⚠️  No player props found")

    # Create unified export
    output_path = Path(f'reports/UNIFIED_BETTING_SHEET_WEEK{week}.csv')

    # Prepare game bets section
    if not game_bets_df.empty:
        game_export = game_bets_df.copy()
        game_export['category'] = 'GAME_BET'
        game_export['week'] = week
    else:
        game_export = pd.DataFrame()

    # Prepare props section
    if not props_df.empty:
        props_export = props_df.copy()
        props_export['category'] = 'PLAYER_PROP'
        props_export['week'] = week
        # Add placeholder columns to match game bets structure
        props_export['wager'] = ''
        props_export['potential_profit'] = ''
        props_export['edge'] = ''
    else:
        props_export = pd.DataFrame()

    # Combine sections
    if not game_export.empty and not props_export.empty:
        # Find common columns
        common_cols = list(set(game_export.columns) & set(props_export.columns))
        unified_df = pd.concat([
            game_export[common_cols],
            props_export[common_cols]
        ], ignore_index=True)
    elif not game_export.empty:
        unified_df = game_export
    elif not props_export.empty:
        unified_df = props_export
    else:
        logger.error("❌ No recommendations to export")
        return

    # Add metadata
    unified_df.insert(0, 'export_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Save unified export
    unified_df.to_csv(output_path, index=False)

    # Also save separate detailed exports
    if not game_bets_df.empty:
        game_detail_path = Path(f'reports/GAME_BETS_WEEK{week}.csv')
        game_bets_df.to_csv(game_detail_path, index=False)
        logger.info(f"📁 Game bets: {game_detail_path}")

    if not props_df.empty:
        props_detail_path = Path(f'reports/PLAYER_PROPS_WEEK{week}.csv')
        props_df.to_csv(props_detail_path, index=False)
        logger.info(f"📁 Player props: {props_detail_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ UNIFIED EXPORT COMPLETE")
    logger.info("="*80)
    logger.info(f"📁 Unified sheet: {output_path}")
    logger.info(f"\n📊 Summary:")
    logger.info(f"   • Game Bets: {len(game_bets_df)}")
    logger.info(f"   • Player Props: {len(props_df)}")
    logger.info(f"   • Total Recommendations: {len(unified_df)}")
    logger.info(f"\n💰 Bankroll Allocation:")
    logger.info(f"   • Total Bankroll: ${config['total_bankroll']:,.2f}")
    logger.info(f"   • Game Bets: {config.get('game_bets_allocation', 1.0)*100:.0f}%")
    logger.info(f"   • Player Props: {config.get('prop_bets_allocation', 0.0)*100:.0f}%")


if __name__ == '__main__':
    # Auto-detect current week (no hardcoded value)
    export_unified_recommendations()  # Uses get_current_week() internally
