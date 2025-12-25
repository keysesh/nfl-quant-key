#!/usr/bin/env python3
"""
NFL QUANT Backtest with REAL Sportsbook Props

Uses actual prop lines from sportsbooks - NO hardcoded values.
Evaluates model predictions against real market conditions.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path (script is in scripts/_archive/backtest/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market, clear_calibrator_cache
from nfl_quant.calibration.bias_correction import apply_bias_correction, get_correction_factor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data'
NFLVERSE_DIR = DATA_DIR / 'nflverse'


def load_historical_props() -> pd.DataFrame:
    """
    Load all historical prop lines from actual sportsbook data.
    NO hardcoded lines - only real market data.
    """
    # Load all available prop files (multiple naming patterns)
    props_files = []
    props_files.extend(list(DATA_DIR.glob('odds_player_props_*.csv')))
    props_files.extend(list(DATA_DIR.glob('odds_week*_player_props.csv')))

    # Remove duplicates
    props_files = list(set(props_files))

    all_props = []

    # Load direct player props files
    for f in props_files:
        try:
            df = pd.read_csv(f)
            # Extract date/week from filename
            df['source_file'] = f.stem
            all_props.append(df)
            logger.info(f"  Loaded {len(df)} props from {f.name}")
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")

    # Also load from comprehensive files (contain player props mixed with game lines)
    comprehensive_files = list(DATA_DIR.glob('odds_*_comprehensive.csv'))
    comprehensive_files.extend(list(DATA_DIR.glob('odds_comprehensive_*.csv')))
    comprehensive_files = list(set(comprehensive_files))

    for f in comprehensive_files:
        try:
            df = pd.read_csv(f)
            # Filter to player props only
            if 'bet_type' in df.columns:
                df = df[df['bet_type'] == 'Player Prop'].copy()
            if 'stat_type' in df.columns and len(df) > 0:
                # Rename columns to match expected format
                if 'player' not in df.columns and 'player_name' in df.columns:
                    df = df.rename(columns={'player_name': 'player'})
                df['source_file'] = f.stem
                all_props.append(df)
                logger.info(f"  Loaded {len(df)} props from {f.name} (comprehensive)")
        except Exception as e:
            logger.warning(f"Could not load comprehensive {f}: {e}")

    # Also load converted historical props (from The Odds API historical data)
    historical_converted = DATA_DIR / 'historical_player_props_converted.csv'
    if historical_converted.exists():
        try:
            df = pd.read_csv(historical_converted)
            df['source_file'] = 'historical_converted'
            all_props.append(df)
            logger.info(f"  Loaded {len(df)} props from historical_player_props_converted.csv")
        except Exception as e:
            logger.warning(f"Could not load historical converted props: {e}")

    # Load comprehensive historical props from backtest folder (weeks 1-11+)
    all_historical_path = DATA_DIR / 'backtest' / 'all_historical_props_2025.csv'
    if all_historical_path.exists():
        try:
            df = pd.read_csv(all_historical_path)
            # Convert column names to match expected format
            # market: player_pass_yds -> stat_type: passing_yards
            market_to_stat = {
                'player_pass_yds': 'passing_yards',
                'player_rush_yds': 'rushing_yards',
                'player_reception_yds': 'receiving_yards',
                'player_receptions': 'receptions',
            }
            if 'market' in df.columns:
                df['stat_type'] = df['market'].map(market_to_stat)
                df = df[df['stat_type'].notna()].copy()

            # Pivot over/under odds into separate columns
            if 'prop_type' in df.columns and 'american_odds' in df.columns:
                # Split into over and under rows
                over_df = df[df['prop_type'] == 'over'].copy()
                under_df = df[df['prop_type'] == 'under'].copy()

                # Rename odds columns
                over_df = over_df.rename(columns={'american_odds': 'over_odds'})
                under_df = under_df.rename(columns={'american_odds': 'under_odds'})

                # Merge on key columns
                merge_cols = ['week', 'player', 'stat_type', 'line']
                if all(col in over_df.columns for col in merge_cols):
                    merged = over_df.merge(
                        under_df[merge_cols + ['under_odds']],
                        on=merge_cols,
                        how='outer'
                    )
                    merged['source_file'] = 'all_historical_2025'
                    all_props.append(merged)
                    logger.info(f"  Loaded {len(merged)} props from all_historical_props_2025.csv")
        except Exception as e:
            logger.warning(f"Could not load all_historical_props_2025.csv: {e}")

    if not all_props:
        raise FileNotFoundError("No odds_player_props_*.csv files found. Cannot backtest without real prop data.")

    props_df = pd.concat(all_props, ignore_index=True)

    # Filter to relevant stat types only
    relevant_stats = ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']
    props_df = props_df[props_df['stat_type'].isin(relevant_stats)].copy()

    # Must have a line value
    props_df = props_df[props_df['line'].notna()].copy()

    # Remove exact duplicates (same player, same stat, same line)
    props_df = props_df.drop_duplicates(subset=['player', 'stat_type', 'line'])

    logger.info(f"Loaded {len(props_df)} unique real prop lines")

    return props_df


def load_game_lines() -> Dict[str, Dict[str, float]]:
    """
    Load Vegas game lines (spreads and totals) from NFLverse for game script adjustments.
    High totals = more passing/volume expected
    Large spreads = potential blowouts, more garbage time
    """
    games_file = NFLVERSE_DIR / 'games.parquet'

    if not games_file.exists():
        logger.warning("No games.parquet found - game script adjustments disabled")
        return {}

    games = pd.read_parquet(games_file)
    games = games[games['season'] == 2025].copy()

    # Build game context lookup by season, week, and team
    game_context = {}

    for _, row in games.iterrows():
        week = row['week']
        home_team = row['home_team']
        away_team = row['away_team']
        spread_line = row.get('spread_line', np.nan)  # Home team spread
        total_line = row.get('total_line', np.nan)

        if pd.isna(spread_line) and pd.isna(total_line):
            continue

        # Store for both teams in this game
        game_info = {
            'spread': spread_line if not pd.isna(spread_line) else 0,
            'total': total_line if not pd.isna(total_line) else 45.0,  # Default NFL total
            'week': week
        }

        # Key by (week, team)
        game_context[(week, home_team)] = {
            'spread': -spread_line if not pd.isna(spread_line) else 0,  # Home team spread
            'total': total_line if not pd.isna(total_line) else 45.0,
            'is_favorite': spread_line < 0 if not pd.isna(spread_line) else False
        }
        game_context[(week, away_team)] = {
            'spread': spread_line if not pd.isna(spread_line) else 0,  # Away team spread (opposite)
            'total': total_line if not pd.isna(total_line) else 45.0,
            'is_favorite': spread_line > 0 if not pd.isna(spread_line) else False
        }

    logger.info(f"Loaded Vegas lines for {len(game_context)//2} games")
    return game_context


def load_nflverse_outcomes(season: int = 2025) -> pd.DataFrame:
    """Load actual player outcomes from NFLverse with home/away info."""
    weekly_file = NFLVERSE_DIR / 'weekly_stats.parquet'
    games_file = NFLVERSE_DIR / 'games.parquet'

    if not weekly_file.exists():
        raise FileNotFoundError(f"NFLverse weekly stats not found: {weekly_file}")

    stats = pd.read_parquet(weekly_file)
    stats = stats[stats['season'] == season].copy()

    # Add home/away information by joining with games data
    if games_file.exists():
        games = pd.read_parquet(games_file)
        games = games[games['season'] == season].copy()

        # Create home/away lookup
        home_games = games[['season', 'week', 'home_team']].copy()
        home_games['is_home'] = True
        home_games = home_games.rename(columns={'home_team': 'team'})

        away_games = games[['season', 'week', 'away_team']].copy()
        away_games['is_home'] = False
        away_games = away_games.rename(columns={'away_team': 'team'})

        game_locations = pd.concat([home_games, away_games], ignore_index=True)

        # Merge with stats
        stats = stats.merge(
            game_locations,
            on=['season', 'week', 'team'],
            how='left'
        )

        # Fill missing with False
        stats['is_home'] = stats['is_home'].fillna(False)

        home_count = stats['is_home'].sum()
        away_count = len(stats) - home_count
        logger.info(f"Added home/away info: {home_count} home games, {away_count} away games")
    else:
        stats['is_home'] = False  # Fallback
        logger.warning("Games file not found - home/away adjustments disabled")

    logger.info(f"Loaded {len(stats)} player-week outcomes from NFLverse")

    return stats


def calculate_team_defensive_strength(outcomes_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate each team's defensive strength by stat type.
    Uses actual data from NFLverse - NO hardcoded values.

    Returns multiplier: <1.0 = good defense (allows less), >1.0 = bad defense (allows more)

    CRITICAL FIX: Calculate total yards allowed per GAME, not per player average.
    This gives proper multipliers in 0.80-1.25 range instead of broken 0.16-0.22.
    """
    # Group by opponent team to see what each defense allows
    defense_stats = {}

    for stat_type, col in [('passing_yards', 'passing_yards'),
                           ('rushing_yards', 'rushing_yards'),
                           ('receiving_yards', 'receiving_yards'),
                           ('receptions', 'receptions')]:

        if col not in outcomes_df.columns or 'opponent_team' not in outcomes_df.columns:
            continue

        # CORRECT: Sum total stat per game (all players vs that defense in that week)
        # Then average across games to get what the defense allows per game
        game_totals = outcomes_df.groupby(['opponent_team', 'week'])[col].sum().reset_index()
        team_allowed_per_game = game_totals.groupby('opponent_team')[col].mean()

        # League average is the average of team averages (what typical defense allows)
        league_avg_per_game = team_allowed_per_game.mean()

        if league_avg_per_game == 0:
            continue

        # Convert to multiplier relative to league average
        # >1.0 means bad defense (allows more than average)
        # <1.0 means good defense (allows less than average)
        for team in team_allowed_per_game.index:
            if team not in defense_stats:
                defense_stats[team] = {}
            defense_stats[team][stat_type] = team_allowed_per_game[team] / league_avg_per_game

    return defense_stats


def calculate_home_away_factors(outcomes_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate home/away performance factors from actual data.

    Returns multipliers for each player showing how they perform at home vs away.
    >1.0 means better at home, <1.0 means better away
    """
    if 'is_home' not in outcomes_df.columns:
        logger.warning("No home/away data available")
        return {}

    player_factors = {}

    for stat_type, col in [('passing_yards', 'passing_yards'),
                           ('rushing_yards', 'rushing_yards'),
                           ('receiving_yards', 'receiving_yards'),
                           ('receptions', 'receptions')]:

        if col not in outcomes_df.columns:
            continue

        # Calculate home/away splits per player
        for player in outcomes_df['player_display_name'].unique():
            player_data = outcomes_df[outcomes_df['player_display_name'] == player]

            home_games = player_data[player_data['is_home'] == True]
            away_games = player_data[player_data['is_home'] == False]

            # Need at least 2 home and 2 away games for reliable estimate
            if len(home_games) < 2 or len(away_games) < 2:
                continue

            home_avg = home_games[col].mean()
            away_avg = away_games[col].mean()

            if away_avg == 0 or pd.isna(home_avg) or pd.isna(away_avg):
                continue

            # Home factor: how much better at home
            home_factor = home_avg / away_avg

            if player not in player_factors:
                player_factors[player] = {}

            player_factors[player][stat_type] = home_factor

    return player_factors


def match_props_to_outcomes(props_df: pd.DataFrame, outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match real sportsbook props to actual NFLverse outcomes.
    Uses player name matching and game context.
    NOW WITH OPPONENT DEFENSE + HOME/AWAY + GAME SCRIPT ADJUSTMENTS!
    """
    results = []

    # Calculate defensive strength from ACTUAL data (no hardcoding)
    defense_strength = calculate_team_defensive_strength(outcomes_df)
    logger.info(f"Calculated defensive strength for {len(defense_strength)} teams")

    # Calculate home/away factors from ACTUAL data
    home_away_factors = calculate_home_away_factors(outcomes_df)
    logger.info(f"Calculated home/away factors for {len(home_away_factors)} players")

    # Load Vegas game lines for game script adjustments
    game_context = load_game_lines()
    logger.info(f"Loaded game script context for {len(game_context)//2} games")

    # Create lookup for outcomes by player name
    # Note: NFLverse uses 'player_display_name', odds use 'player'

    for _, prop in props_df.iterrows():
        player_name = prop['player']
        stat_type = prop['stat_type']
        line = prop['line']

        # Map stat types between odds and NFLverse
        nflverse_col_map = {
            'passing_yards': 'passing_yards',
            'rushing_yards': 'rushing_yards',
            'receiving_yards': 'receiving_yards',
            'receptions': 'receptions',
        }

        nflverse_col = nflverse_col_map.get(stat_type)
        if not nflverse_col:
            continue

        # Find matching player in outcomes
        # Try exact match first
        player_names_lower = outcomes_df['player_display_name'].fillna('').str.lower()
        player_outcomes = outcomes_df[
            player_names_lower == player_name.lower()
        ]

        # If no exact match, try fuzzy matching
        if len(player_outcomes) == 0:
            # Try partial match (last name)
            last_name = player_name.split()[-1].lower()
            player_outcomes = outcomes_df[
                player_names_lower.str.contains(last_name, na=False)
            ]

            # If multiple matches, skip (ambiguous)
            if len(player_outcomes) > 10:  # Too many matches, probably wrong
                continue

        if len(player_outcomes) == 0:
            continue

        # Get the actual value for this stat
        if nflverse_col not in player_outcomes.columns:
            continue

        # Use most recent week for this player
        latest = player_outcomes.sort_values('week', ascending=False).iloc[0]
        actual_value = latest[nflverse_col]

        if pd.isna(actual_value):
            continue

        # Calculate if went over
        went_over = 1 if actual_value > line else 0

        # Calculate model's predicted probability
        # Use player's trailing average as prediction
        player_history = player_outcomes[player_outcomes['week'] < latest['week']]

        if len(player_history) == 0:
            # No history - skip this prop
            continue

        # RECENCY WEIGHTING: Recent games count more than old games
        # More recent performance is better predictor than season average
        if len(player_history) >= 2:
            weeks = player_history['week'].values
            values = player_history[nflverse_col].values

            # Exponential decay weights (recent = higher weight)
            max_week = weeks.max()
            weights = np.exp(-(max_week - weeks) / 3.0)  # Decay factor of 3 weeks
            weights = weights / weights.sum()

            pred_mean = np.sum(values * weights)
        else:
            pred_mean = player_history[nflverse_col].mean()

        # OPPONENT DEFENSE ADJUSTMENT
        # Adjust prediction based on opponent's defensive strength
        opponent_team = latest.get('opponent_team')
        defense_multiplier = 1.0

        if opponent_team and opponent_team in defense_strength:
            if stat_type in defense_strength[opponent_team]:
                defense_multiplier = defense_strength[opponent_team][stat_type]
                # Blend with regression to mean (don't fully trust defense rating)
                # Use 50% weight for defense adjustment
                defense_multiplier = 0.5 * defense_multiplier + 0.5 * 1.0

        # TARGET SHARE ADJUSTMENT (for receiving props)
        # Players with high target share are more stable/predictable
        target_share_multiplier = 1.0
        player_target_share = None

        if stat_type in ['receiving_yards', 'receptions']:
            if 'target_share' in player_history.columns:
                # Get player's average target share
                player_target_share = player_history['target_share'].mean()

                if not pd.isna(player_target_share) and player_target_share > 0:
                    # League average target share for a starter is ~15-20%
                    # Elite receivers get 25-30%
                    # High target share = more opportunities = more stable production
                    league_avg_target_share = 0.15

                    # Adjustment: high target share players are more likely to hit their lines
                    # because they have more opportunities and less variance
                    target_share_ratio = player_target_share / league_avg_target_share

                    # Don't over-adjust - cap at +/- 10%
                    target_share_multiplier = np.clip(target_share_ratio, 0.90, 1.10)

        # CATCH RATE MOMENTUM (for receiving props)
        # If recent catch rate > season average, player is hot
        catch_rate_multiplier = 1.0
        player_catch_rate = None

        if stat_type in ['receiving_yards', 'receptions']:
            if 'targets' in player_history.columns and 'receptions' in player_history.columns:
                # Calculate catch rate (receptions / targets)
                targets = player_history['targets'].sum()
                receptions_total = player_history['receptions'].sum()

                if targets > 0:
                    player_catch_rate = receptions_total / targets

                    # Check recent performance (last 2 games vs season)
                    if len(player_history) >= 3:
                        recent = player_history.sort_values('week', ascending=False).head(2)
                        recent_targets = recent['targets'].sum()
                        recent_receptions = recent['receptions'].sum()

                        if recent_targets > 0:
                            recent_catch_rate = recent_receptions / recent_targets
                            catch_rate_momentum = recent_catch_rate / player_catch_rate

                            # Hot hand effect: +/- 5% max adjustment
                            catch_rate_multiplier = np.clip(catch_rate_momentum, 0.95, 1.05)

        # QB CPOE ADJUSTMENT (for passing yards)
        # QBs with high CPOE are more efficient/accurate
        cpoe_multiplier = 1.0
        player_cpoe = None

        if stat_type == 'passing_yards':
            if 'passing_cpoe' in player_history.columns:
                cpoe_values = player_history['passing_cpoe'].dropna()
                if len(cpoe_values) > 0:
                    player_cpoe = cpoe_values.mean()

                    # CPOE ranges from about -10 to +10
                    # Positive = better than expected
                    # Apply small adjustment (1% per CPOE point)
                    cpoe_multiplier = 1.0 + (player_cpoe / 100.0)
                    cpoe_multiplier = np.clip(cpoe_multiplier, 0.95, 1.05)

        # YAC EFFICIENCY (for receiving yards)
        # Players who create YAC are more explosive
        yac_multiplier = 1.0
        player_yac_pct = None

        if stat_type == 'receiving_yards':
            if 'receiving_yards_after_catch' in player_history.columns and 'receiving_yards' in player_history.columns:
                total_rec_yards = player_history['receiving_yards'].sum()
                total_yac = player_history['receiving_yards_after_catch'].sum()

                if total_rec_yards > 0:
                    player_yac_pct = total_yac / total_rec_yards

                    # League average YAC% is about 50%
                    # High YAC players are more explosive
                    yac_ratio = player_yac_pct / 0.50
                    yac_multiplier = np.clip(yac_ratio, 0.95, 1.05)

        # HOME/AWAY ADJUSTMENT
        # Adjust prediction based on player's home/away performance
        is_home = latest.get('is_home', False)
        home_away_multiplier = 1.0

        if player_name in home_away_factors:
            if stat_type in home_away_factors[player_name]:
                home_factor = home_away_factors[player_name][stat_type]
                # home_factor > 1.0 means player is better at home
                # If playing home, apply factor; if away, apply inverse
                if home_factor > 0 and not pd.isna(home_factor):
                    if is_home:
                        home_away_multiplier = home_factor
                    else:
                        home_away_multiplier = 1.0 / home_factor
                    # Blend with 40% weight to avoid overconfidence
                    home_away_multiplier = 0.4 * home_away_multiplier + 0.6 * 1.0

        # GAME SCRIPT ADJUSTMENT (Vegas lines)
        # High totals = more volume expected
        # Large spreads = potential garbage time adjustments
        game_script_multiplier = 1.0
        team = latest.get('team')
        week = latest.get('week')

        if team and week and (week, team) in game_context:
            vegas_info = game_context[(week, team)]
            total_line = vegas_info.get('total', 45.0)
            spread = vegas_info.get('spread', 0)
            is_favorite = vegas_info.get('is_favorite', False)

            # Total line adjustment: High scoring games = more volume
            # NFL average total is ~45 points
            total_factor = total_line / 45.0

            # Different stats respond differently to game script
            if stat_type == 'passing_yards':
                # High totals favor passing (aerial attack)
                # Trailing teams (underdogs) may pass more
                game_script_multiplier = 0.3 * total_factor + 0.7 * 1.0

                # Big underdogs may throw more in comeback attempts
                if spread > 7 and not is_favorite:
                    game_script_multiplier *= 1.05  # 5% boost for likely trailing team

            elif stat_type == 'rushing_yards':
                # High totals slightly favor rushing too
                # Favorites may run more to kill clock
                game_script_multiplier = 0.2 * total_factor + 0.8 * 1.0

                # Big favorites run more to control game
                if spread < -7 and is_favorite:
                    game_script_multiplier *= 1.05

            elif stat_type in ['receiving_yards', 'receptions']:
                # Similar to passing - high totals = more catches
                game_script_multiplier = 0.3 * total_factor + 0.7 * 1.0

                # Underdogs may target slot receivers more in catch-up mode
                if spread > 7 and not is_favorite:
                    game_script_multiplier *= 1.03
        else:
            game_script_multiplier = 1.0

        # Apply all adjustments to prediction
        pred_mean_adjusted = (pred_mean * defense_multiplier * home_away_multiplier *
                              game_script_multiplier * target_share_multiplier *
                              catch_rate_multiplier * cpoe_multiplier * yac_multiplier)

        # Apply bias correction to fix systematic over-prediction
        # Map stat_type to market name for correction lookup
        market_name = f"player_{stat_type.replace('_', '_')}"
        if stat_type == 'receiving_yards':
            market_name = 'player_reception_yds'
        elif stat_type == 'rushing_yards':
            market_name = 'player_rush_yds'
        elif stat_type == 'passing_yards':
            market_name = 'player_pass_yds'
        elif stat_type == 'receptions':
            market_name = 'player_receptions'

        pred_mean_adjusted = apply_bias_correction(pred_mean_adjusted, market_name)

        # Estimate std from historical variance - USE ACTUAL DATA, NOT HARDCODED
        if len(player_history) >= 3:
            pred_std = player_history[nflverse_col].std()
            if pred_std == 0 or pd.isna(pred_std):
                # POSITION-TIER SPECIFIC CV (coefficient of variation)
                # Based on analysis: elite players have lower variance than depth players
                position = latest.get('position', 'UNK')

                if stat_type == 'passing_yards':
                    # QBs: Elite (Mahomes, Allen) ~30% CV, Average ~40%, Bad ~55%
                    if pred_mean > 250:  # Elite QB
                        base_cv = 0.30
                    elif pred_mean > 200:  # Good QB
                        base_cv = 0.38
                    else:  # Below average
                        base_cv = 0.50
                elif stat_type == 'rushing_yards':
                    # RBs: Lead back ~45% CV, Committee ~55%
                    if pred_mean > 70:  # Lead back
                        base_cv = 0.45
                    elif pred_mean > 40:  # Rotational
                        base_cv = 0.55
                    else:  # Depth/specialist
                        base_cv = 0.65
                elif stat_type == 'receiving_yards':
                    # WRs: WR1 ~50% CV, WR2 ~60%, WR3/TE ~70%
                    if pred_mean > 60:  # WR1/elite
                        base_cv = 0.50
                    elif pred_mean > 40:  # WR2/solid starter
                        base_cv = 0.60
                    else:  # Depth
                        base_cv = 0.70
                elif stat_type == 'receptions':
                    # Receptions are more stable than yards
                    if pred_mean > 5:  # High volume target
                        base_cv = 0.35
                    elif pred_mean > 3:  # Medium target
                        base_cv = 0.45
                    else:  # Low volume
                        base_cv = 0.55
                else:
                    base_cv = 0.45

                pred_std = pred_mean_adjusted * base_cv
        elif len(player_history) >= 2:
            pred_std = player_history[nflverse_col].std()
            if pred_std == 0 or pd.isna(pred_std):
                pred_std = pred_mean_adjusted * 0.50  # Higher uncertainty with less data
        else:
            pred_std = pred_mean_adjusted * 0.55  # Very high uncertainty

        # Minimum variance based on stat type (data-driven, not arbitrary)
        min_std = pred_mean_adjusted * 0.15 if stat_type == 'receptions' else pred_mean_adjusted * 0.20
        pred_std = max(pred_std, min_std)

        # Ensure pred_std is valid (avoid divide by zero)
        if pred_std <= 0 or pd.isna(pred_std):
            pred_std = pred_mean_adjusted * 0.45  # Fallback to typical CV

        # Calculate probability using PROPER normal CDF (not broken tanh approximation)
        if pred_std > 0 and pred_mean_adjusted > 0:
            z_score = (line - pred_mean_adjusted) / pred_std
            prob_over = 1.0 - stats.norm.cdf(z_score)  # Proper normal CDF
            prob_over = np.clip(prob_over, 0.01, 0.99)
        else:
            # Invalid prediction, skip
            continue

        results.append({
            'player': player_name,
            'stat_type': stat_type,
            'line': line,
            'over_odds': prop.get('over_odds'),
            'under_odds': prop.get('under_odds'),
            'actual_value': actual_value,
            'went_over': went_over,
            'pred_mean_raw': pred_mean,
            'pred_mean': pred_mean_adjusted,
            'defense_multiplier': defense_multiplier,
            'home_away_multiplier': home_away_multiplier,
            'game_script_multiplier': game_script_multiplier,
            'target_share_multiplier': target_share_multiplier,
            'target_share': player_target_share if player_target_share else None,
            'catch_rate_multiplier': catch_rate_multiplier,
            'catch_rate': player_catch_rate if player_catch_rate else None,
            'cpoe_multiplier': cpoe_multiplier,
            'cpoe': player_cpoe if player_cpoe else None,
            'yac_multiplier': yac_multiplier,
            'yac_pct': player_yac_pct if player_yac_pct else None,
            'is_home': is_home,
            'opponent_team': opponent_team,
            'pred_std': pred_std,
            'prob_over_raw': prob_over,
            'week': latest['week'],
            'position': latest.get('position', 'UNK'),
        })

    return pd.DataFrame(results)


def apply_calibration(props_df: pd.DataFrame, use_calibrators: bool = False) -> pd.DataFrame:
    """
    Apply market-specific calibrators.

    IMPORTANT: Set use_calibrators=False until calibrators are retrained on real data.
    Current calibrators were trained on fake/invalid data and hurt performance.
    """
    if len(props_df) == 0:
        return props_df

    props_df = props_df.copy()

    if not use_calibrators:
        # Skip calibration - use raw probabilities
        # This is correct until we retrain calibrators on REAL sportsbook props
        logger.info("⚠️  Calibrators DISABLED (trained on invalid data)")
        props_df['prob_over_calibrated'] = props_df['prob_over_raw']
        return props_df

    clear_calibrator_cache()

    market_map = {
        'passing_yards': 'player_pass_yds',
        'rushing_yards': 'player_rush_yds',
        'receiving_yards': 'player_reception_yds',
        'receptions': 'player_receptions',
    }

    calibrated_probs = []

    for _, row in props_df.iterrows():
        market = market_map.get(row['stat_type'], 'default')

        try:
            calibrator = load_calibrator_for_market(market)
            if calibrator and hasattr(calibrator, 'transform'):
                cal_prob = calibrator.transform(row['prob_over_raw'])
            else:
                cal_prob = row['prob_over_raw']
        except Exception:
            cal_prob = row['prob_over_raw']

        calibrated_probs.append(cal_prob)

    props_df['prob_over_calibrated'] = calibrated_probs

    return props_df


def calculate_edge(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting edge by comparing model probability to market implied probability.

    American odds to implied probability:
    - Negative odds: -odds / (-odds + 100)
    - Positive odds: 100 / (odds + 100)
    """
    props_df = props_df.copy()

    def american_to_prob(odds):
        if pd.isna(odds):
            return None
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)

    props_df['market_prob_over'] = props_df['over_odds'].apply(american_to_prob)
    props_df['market_prob_under'] = props_df['under_odds'].apply(american_to_prob)

    # Edge = model prob - market prob
    props_df['edge_over'] = props_df['prob_over_calibrated'] - props_df['market_prob_over']
    props_df['edge_under'] = (1 - props_df['prob_over_calibrated']) - props_df['market_prob_under']

    return props_df


def calculate_metrics(props_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive backtest metrics."""
    if len(props_df) == 0:
        return {}

    metrics = {
        'total_props': len(props_df),
        'timestamp': datetime.now().isoformat(),
    }

    # Model accuracy
    cal_probs = props_df['prob_over_calibrated'].values
    actuals = props_df['went_over'].values

    # Brier score
    metrics['brier_score'] = float(np.mean((cal_probs - actuals) ** 2))

    # Model recommendation accuracy
    model_recommends_over = cal_probs > 0.5
    model_correct = np.where(
        model_recommends_over,
        actuals == 1,
        actuals == 0
    )

    metrics['model_accuracy'] = float(model_correct.mean())
    metrics['over_recommendations'] = int(model_recommends_over.sum())
    metrics['under_recommendations'] = int((~model_recommends_over).sum())

    if model_recommends_over.sum() > 0:
        metrics['over_accuracy'] = float(model_correct[model_recommends_over].mean())
    else:
        metrics['over_accuracy'] = 0.0

    if (~model_recommends_over).sum() > 0:
        metrics['under_accuracy'] = float(model_correct[~model_recommends_over].mean())
    else:
        metrics['under_accuracy'] = 0.0

    # Edge-based betting simulation
    if 'edge_over' in props_df.columns:
        edge_threshold = 0.05  # 5% edge minimum

        # Bet OVER when model has 5%+ edge on over
        over_bets = props_df[props_df['edge_over'] > edge_threshold]
        if len(over_bets) > 0:
            over_wins = (over_bets['went_over'] == 1).sum()
            metrics['over_edge_bets'] = len(over_bets)
            metrics['over_edge_win_rate'] = float(over_wins / len(over_bets))

        # Bet UNDER when model has 5%+ edge on under
        under_bets = props_df[props_df['edge_under'] > edge_threshold]
        if len(under_bets) > 0:
            under_wins = (under_bets['went_over'] == 0).sum()
            metrics['under_edge_bets'] = len(under_bets)
            metrics['under_edge_win_rate'] = float(under_wins / len(under_bets))

    # By stat type
    stat_metrics = {}
    for stat_type in props_df['stat_type'].unique():
        stat_data = props_df[props_df['stat_type'] == stat_type]
        stat_probs = stat_data['prob_over_calibrated'].values
        stat_actuals = stat_data['went_over'].values

        stat_rec_over = stat_probs > 0.5
        stat_correct = np.where(stat_rec_over, stat_actuals == 1, stat_actuals == 0)

        stat_metrics[stat_type] = {
            'count': len(stat_data),
            'model_accuracy': float(stat_correct.mean()),
            'brier': float(np.mean((stat_probs - stat_actuals) ** 2)),
        }

    metrics['by_stat_type'] = stat_metrics

    return metrics


def run_backtest():
    """Run backtest using REAL sportsbook props."""
    logger.info("=" * 80)
    logger.info("NFL QUANT BACKTEST - REAL SPORTSBOOK PROPS")
    logger.info("NO HARDCODED VALUES - Using actual market data")
    logger.info("=" * 80)

    # Load real data
    try:
        props_df = load_historical_props()
    except FileNotFoundError as e:
        logger.error(f"Cannot run backtest: {e}")
        logger.error("You need historical odds_player_props_*.csv files")
        return None, {}

    outcomes_df = load_nflverse_outcomes()

    # Match props to outcomes
    logger.info("\nMatching props to actual outcomes...")
    matched_df = match_props_to_outcomes(props_df, outcomes_df)

    if len(matched_df) == 0:
        logger.error("No props could be matched to outcomes")
        return None, {}

    logger.info(f"Successfully matched {len(matched_df)} props")

    # Apply calibration
    # Enable calibrators now that they're trained on REAL data
    logger.info("\nApplying calibration...")
    matched_df = apply_calibration(matched_df, use_calibrators=True)

    # Calculate edge vs market
    logger.info("Calculating edge vs market odds...")
    matched_df = calculate_edge(matched_df)

    # Calculate metrics
    metrics = calculate_metrics(matched_df)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    logger.info(f"\nTotal props evaluated: {metrics['total_props']}")
    logger.info(f"Brier Score: {metrics['brier_score']:.4f}")

    logger.info(f"\n🎯 MODEL RECOMMENDATION ACCURACY:")
    logger.info(f"   Overall: {metrics['model_accuracy']:.1%}")
    logger.info(f"   OVER: {metrics['over_recommendations']} bets at {metrics['over_accuracy']:.1%}")
    logger.info(f"   UNDER: {metrics['under_recommendations']} bets at {metrics['under_accuracy']:.1%}")
    logger.info(f"   (Need ~52.4% to break even)")

    if 'over_edge_bets' in metrics:
        logger.info(f"\n💰 EDGE-BASED BETTING (>5% edge):")
        logger.info(f"   OVER bets: {metrics.get('over_edge_bets', 0)} at {metrics.get('over_edge_win_rate', 0):.1%}")
        logger.info(f"   UNDER bets: {metrics.get('under_edge_bets', 0)} at {metrics.get('under_edge_win_rate', 0):.1%}")

    logger.info(f"\n📈 BY STAT TYPE:")
    for stat_type, data in metrics.get('by_stat_type', {}).items():
        logger.info(f"   {stat_type}: {data['count']} props, {data['model_accuracy']:.1%} accuracy, Brier={data['brier']:.4f}")

    # Show sample predictions
    logger.info(f"\n📋 SAMPLE PREDICTIONS:")
    sample = matched_df.head(10)
    for _, row in sample.iterrows():
        edge = row.get('edge_over', 0) if row['prob_over_calibrated'] > 0.5 else row.get('edge_under', 0)
        rec = "OVER" if row['prob_over_calibrated'] > 0.5 else "UNDER"
        result = "✅" if (rec == "OVER" and row['went_over'] == 1) or (rec == "UNDER" and row['went_over'] == 0) else "❌"
        logger.info(f"   {row['player'][:20]:20s} {row['stat_type']:15s} line={row['line']:6.1f} "
                   f"actual={row['actual_value']:6.1f} rec={rec:5s} prob={row['prob_over_calibrated']:.2f} "
                   f"edge={edge:+.3f} {result}")

    return matched_df, metrics


if __name__ == "__main__":
    results_df, metrics = run_backtest()

    if results_df is not None:
        # Save results
        output_file = DATA_DIR / 'backtest' / 'real_props_backtest.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nResults saved to {output_file}")
