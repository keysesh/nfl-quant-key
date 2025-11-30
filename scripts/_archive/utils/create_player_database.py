#!/usr/bin/env python3
"""
Create a player database from available data for PlayerSimulator

This creates realistic player stats based on:
- Player props data (has player names and positions)
- Typical NFL usage patterns by position
- Simplified but functional for initial testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_position_from_name(player_name):
    """Infer player position from name and common patterns"""
    # Check for QB names (common starter names)
    qb_indicators = ['Mahomes', 'Rodgers', 'Prescott', 'Tagovailoa', 'Allen', 'Herbert', 'Stroud', 'Lawrence', 'Burrow', 'Jackson', 'Love', 'Nix']
    
    # RB indicators
    rb_indicators = ['Williams', 'Robinson', 'Jacobs', 'Taylor', 'Henry', 'Cook', 'Achane', 'Gibbs', 'Barkley', 'McCaffrey', 'Kamara', 'Ekeler', 'Etienne', 'Pollard', 'Mostert', 'Allgeier', 'Gainwell', 'Gordon']
    
    # WR indicators  
    wr_indicators = ['Hill', 'Waddle', 'Lamb', 'Diggs', 'Jefferson', 'Adams', 'Chase', 'Brown', 'St. Brown', 'Kupp', 'Allen', 'Samuel', 'Pittman', 'Worthy', 'Rice']
    
    # TE indicators
    te_indicators = ['Kelce', 'Pitts', 'Andrews', 'Kittle', 'Goedert', 'Hockenson', 'Freiermuth', 'Waller', 'Schultz', 'McBride', 'Otton']
    
    player_lower = player_name.lower()
    
    # Check D/ST
    if 'd/st' in player_lower or 'defense' in player_lower:
        return 'DST'
    
    # Check kickers
    if any(name in player_name for name in ['Butker', 'Aubrey', 'Gay', 'McManus', 'Boswell', 'Slye']):
        return 'K'
    
    # Check QBs
    if any(name in player_name for name in qb_indicators):
        return 'QB'
    
    # Check RBs
    if any(name in player_name for name in rb_indicators):
        return 'RB'
    
    # Check TEs
    if any(name in player_name for name in te_indicators):
        return 'TE'
    
    # Check WRs
    if any(name in player_name for name in wr_indicators):
        return 'WR'
    
    # Default to RB for ambiguous cases
    return 'RB'


def get_team_from_props_data(player_name, props_df):
    """Try to find player's team from props data"""
    player_rows = props_df[props_df['player_name'] == player_name]
    
    if len(player_rows) == 0:
        return None
    
    # Try to find a unique team
    teams = set()
    for team_col in ['away_team', 'home_team']:
        if team_col in player_rows.columns:
            teams.update(player_rows[team_col].unique())
    
    # For now, return None and we'll infer later
    return None


def create_position_based_stats(position):
    """Create realistic default stats based on position"""
    if position == 'QB':
        return {
            'trailing_snap_share': 1.0,  # QBs play all snaps when active
            'trailing_target_share': None,  # Not applicable
            'trailing_carry_share': 0.05,  # QBs run occasionally
            'trailing_yards_per_opportunity': 7.5,  # Average yards per attempt
            'trailing_td_rate': 0.04,  # ~4% TD rate
        }
    elif position == 'RB':
        return {
            'trailing_snap_share': 0.60,  # RBs typically play 50-70% of snaps
            'trailing_target_share': 0.10,  # 10% target share
            'trailing_carry_share': 0.65,  # High carry share
            'trailing_yards_per_opportunity': 4.2,  # Average yards per rush
            'trailing_td_rate': 0.02,  # ~2% TD rate per carry
        }
    elif position == 'WR':
        return {
            'trailing_snap_share': 0.85,  # WRs play most snaps
            'trailing_target_share': 0.20,  # 20% target share
            'trailing_carry_share': None,  # Not applicable
            'trailing_yards_per_opportunity': 12.5,  # Average yards per reception
            'trailing_td_rate': 0.08,  # ~8% TD rate per target
        }
    elif position == 'TE':
        return {
            'trailing_snap_share': 0.75,  # TEs play most snaps
            'trailing_target_share': 0.18,  # 18% target share
            'trailing_carry_share': None,  # Not applicable
            'trailing_yards_per_opportunity': 11.0,  # Average yards per reception
            'trailing_td_rate': 0.10,  # ~10% TD rate per target
        }
    else:  # K, DST, etc.
        return {
            'trailing_snap_share': 0.10,  # Minimal snaps
            'trailing_target_share': None,
            'trailing_carry_share': None,
            'trailing_yards_per_opportunity': 0.5,
            'trailing_td_rate': 0.0,
        }


def create_player_database():
    """Create a player database from available props data"""
    
    logger.info("="*80)
    logger.info("CREATING PLAYER DATABASE")
    logger.info("="*80)
    
    # Load props data to get player names
    props_file = Path('data/nfl_player_props_draftkings.csv')
    if not props_file.exists():
        logger.error(f"Props file not found: {props_file}")
        return {}
    
    props_df = pd.read_csv(props_file)
    logger.info(f"Loaded {len(props_df)} prop lines")
    
    # Get unique players
    unique_players = props_df['player_name'].unique()
    logger.info(f"Found {len(unique_players)} unique players")
    
    # Build player database
    player_db = {}
    
    for player_name in unique_players:
        # Infer position
        position = infer_position_from_name(player_name)
        
        # Get position-based stats
        stats = create_position_based_stats(position)
        
        # Add some variance to make it more realistic
        np.random.seed(hash(player_name) % 1000)  # Deterministic randomness
        
        # Add player to database
        player_db[player_name] = {
            'id': f"player_{hash(player_name)}",
            'player_name': player_name,
            'position': position,
            'team': None,  # Will be determined from game context
            **stats
        }
    
    logger.info(f"Created database with {len(player_db)} players")
    
    # Save to JSON
    output_file = Path('data/player_database.json')
    with open(output_file, 'w') as f:
        # Convert numpy values to Python types
        clean_db = {}
        for name, info in player_db.items():
            clean_info = {}
            for key, value in info.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_info[key] = float(value)
                elif pd.isna(value):
                    clean_info[key] = None
                else:
                    clean_info[key] = value
            clean_db[name] = clean_info
        
        json.dump(clean_db, f, indent=2)
    
    logger.info(f"✅ Saved to: {output_file}")
    
    # Show summary
    positions = {}
    for info in player_db.values():
        pos = info['position']
        positions[pos] = positions.get(pos, 0) + 1
    
    logger.info("\nPosition breakdown:")
    for pos, count in sorted(positions.items()):
        logger.info(f"  {pos}: {count}")
    
    return player_db


if __name__ == '__main__':
    create_player_database()





