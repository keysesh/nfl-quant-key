#!/usr/bin/env python3
"""
Match Historical DraftKings Odds to NFLverse Actual Stats

This script:
1. Loads historical odds from DraftKings
2. Loads actual player stats from NFLverse
3. Matches players and games
4. Determines if each prop bet hit (over) or missed (under)
5. Creates unbiased backtest dataset

Key challenge: Player name matching between DraftKings and NFLverse
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def normalize_player_name(name: str) -> str:
    """
    Normalize player name for matching.

    Handles:
    - Jr., Sr., III, II suffixes
    - Case differences
    - Common nicknames
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()

    # Remove common suffixes
    for suffix in [' Jr.', ' Jr', ' Sr.', ' Sr', ' III', ' II', ' IV', ' V']:
        name = name.replace(suffix, '')

    # Common name mappings
    name_map = {
        'Gabe Davis': 'Gabriel Davis',
        'DJ Moore': 'D.J. Moore',
        'DK Metcalf': 'D.K. Metcalf',
        'AJ Brown': 'A.J. Brown',
        'TJ Hockenson': 'T.J. Hockenson',
        'CeeDee Lamb': 'CeeDee Lamb',
        'Ja\'Marr Chase': 'Ja\'Marr Chase',
        'De\'Von Achane': 'Devon Achane',
        'Ty\'Son Williams': 'Ty\'Son Williams',
        'Van Jefferson': 'Van Jefferson',
        'Velus Jones': 'Velus Jones Jr.',
        'Kenneth Walker': 'Kenneth Walker III',
    }

    # Apply specific mappings
    if name in name_map:
        name = name_map[name]

    # Lowercase for comparison
    return name.lower().strip()


def get_game_id_from_odds(row) -> str:
    """Create standardized game_id from odds data"""
    # Odds data has game_id like "20251102_CHI_CIN"
    return row['game_id']


def get_game_id_from_nflverse(row) -> str:
    """Create standardized game_id from NFLverse schedule"""
    # Format: YYYYMMDD_AWAY_HOME
    date_str = pd.to_datetime(row['game_date']).strftime('%Y%m%d')
    return f"{date_str}_{row['away_team']}_{row['home_team']}"


def load_historical_odds() -> pd.DataFrame:
    """Load historical DraftKings odds"""
    # Try the complete 2024-2025 file first
    odds_file = PROJECT_ROOT / 'data' / 'historical' / 'historical_odds_2024_2025_complete.csv'

    if not odds_file.exists():
        # Fall back to old file
        odds_file = PROJECT_ROOT / 'data' / 'historical' / 'player_props_history_20250905_120000Z.csv'

    if not odds_file.exists():
        raise FileNotFoundError(f"Historical odds file not found: {odds_file}")

    df = pd.read_csv(odds_file)
    print(f"Loaded {len(df)} historical odds records")
    print(f"Markets: {df['market'].unique().tolist()}")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")

    # Add normalized player name
    df['player_normalized'] = df['player'].apply(normalize_player_name)

    return df


def load_nflverse_stats() -> pd.DataFrame:
    """Load NFLverse actual player stats"""
    stats_file = PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv'

    if not stats_file.exists():
        raise FileNotFoundError(f"NFLverse stats file not found: {stats_file}")

    df = pd.read_csv(stats_file)
    print(f"Loaded {len(df)} NFLverse player stat records")

    # Add normalized player name
    df['player_normalized'] = df['player_display_name'].apply(normalize_player_name)

    return df


def load_schedule() -> pd.DataFrame:
    """Load NFL schedule for game matching"""
    schedule_file = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules_2024_2025.csv'

    if not schedule_file.exists():
        raise FileNotFoundError(f"Schedule file not found: {schedule_file}")

    df = pd.read_csv(schedule_file)
    df['game_date'] = pd.to_datetime(df['gameday'])

    # Create game_id
    df['game_id_custom'] = df.apply(get_game_id_from_nflverse, axis=1)

    return df


def match_game_to_week(game_id: str, schedule_df: pd.DataFrame) -> dict:
    """
    Match a game_id (YYYYMMDD_AWAY_HOME) to season/week using schedule

    Returns: dict with season, week, and game_id_nflverse
    """
    try:
        match = schedule_df[schedule_df['game_id_custom'] == game_id]
        if not match.empty:
            row = match.iloc[0]
            return {
                'season': int(row['season']),
                'week': int(row['week']),
                'game_id_nflverse': row['game_id']
            }
    except Exception as e:
        pass

    # Try to match by teams only (ignore date discrepancies due to timezone)
    parts = game_id.split('_')
    if len(parts) >= 3:
        date_str = parts[0]
        away_team = parts[1]
        home_team = parts[2]

        try:
            # Parse the date from game_id
            game_date = datetime.strptime(date_str, '%Y%m%d')
            year = game_date.year
            month = game_date.month

            # Determine season (September = new season)
            if month >= 9:
                season = year
            else:
                season = year - 1

            # Try to find game by teams within +/- 1 day
            schedule_df['gameday_dt'] = pd.to_datetime(schedule_df['gameday'])
            mask = (
                (schedule_df['away_team'] == away_team) &
                (schedule_df['home_team'] == home_team) &
                (schedule_df['season'] == season) &
                (abs((schedule_df['gameday_dt'] - game_date).dt.days) <= 1)
            )
            matches = schedule_df[mask]

            if not matches.empty:
                row = matches.iloc[0]
                return {
                    'season': int(row['season']),
                    'week': int(row['week']),
                    'game_id_nflverse': row['game_id']
                }

            # Fallback: just return season, no week
            return {
                'season': season,
                'week': None,
                'game_id_nflverse': None
            }
        except:
            pass

    return {'season': None, 'week': None, 'game_id_nflverse': None}


def get_actual_stat(
    player_name: str,
    team: str,
    season: int,
    week: int,
    stat_type: str,
    stats_df: pd.DataFrame
) -> float:
    """
    Get actual stat value from NFLverse data.

    Args:
        player_name: Normalized player name
        team: Team abbreviation
        season: NFL season
        week: NFL week
        stat_type: Type of stat (passing_yards, rushing_yards, etc.)

    Returns: Actual stat value or NaN if not found
    """
    # Map market types to NFLverse column names
    stat_column_map = {
        'player_pass_yds': 'passing_yards',
        'player_rush_yds': 'rushing_yards',
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_rec_tds': 'receiving_tds',
    }

    col_name = stat_column_map.get(stat_type)
    if not col_name:
        return np.nan

    # Find player's actual stats
    mask = (
        (stats_df['player_normalized'] == player_name) &
        (stats_df['season'] == season) &
        (stats_df['week'] == week)
    )

    matches = stats_df[mask]

    if matches.empty:
        # Try matching by team as fallback
        mask_team = (
            (stats_df['player_normalized'] == player_name) &
            (stats_df['team'] == team) &
            (stats_df['season'] == season) &
            (stats_df['week'] == week)
        )
        matches = stats_df[mask_team]

    if matches.empty:
        return np.nan

    # Get the stat value
    if col_name in matches.columns:
        return matches.iloc[0][col_name]
    else:
        return np.nan


def match_odds_to_actuals(
    odds_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    schedule_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match each odds record to actual player stats.

    Creates a dataset with:
    - Original odds (line, over_odds, under_odds)
    - Actual stat value
    - Hit indicator (1 if over hit, 0 if under hit)
    """
    print("\nMatching odds to actual stats...")

    results = []
    unmatched_players = set()
    unmatched_games = set()

    # Group by game to batch process
    unique_games = odds_df['game_id'].unique()
    print(f"Processing {len(unique_games)} unique games...")

    for i, game_id in enumerate(unique_games):
        if (i + 1) % 20 == 0:
            print(f"  Processing game {i+1}/{len(unique_games)}...")

        # Get game info
        game_info = match_game_to_week(game_id, schedule_df)
        season = game_info['season']
        week = game_info['week']

        if season is None or week is None:
            unmatched_games.add(game_id)
            continue

        # Get all props for this game
        game_odds = odds_df[odds_df['game_id'] == game_id]

        # Group by player and market (one over + one under per player per market)
        for (player, market), group in game_odds.groupby(['player', 'market']):
            # Get over and under odds
            over_row = group[group['prop_type'] == 'over']
            under_row = group[group['prop_type'] == 'under']

            if over_row.empty or under_row.empty:
                continue

            over_row = over_row.iloc[0]
            under_row = under_row.iloc[0]

            # Sanity check - lines should match
            if over_row['line'] != under_row['line']:
                continue

            line = over_row['line']
            player_normalized = normalize_player_name(player)

            # Determine team
            team = over_row['home_team']  # Will try both if needed

            # Get actual stat
            actual = get_actual_stat(
                player_normalized,
                team,
                season,
                week,
                market,
                stats_df
            )

            # If not found with home team, try away team
            if pd.isna(actual):
                actual = get_actual_stat(
                    player_normalized,
                    over_row['away_team'],
                    season,
                    week,
                    market,
                    stats_df
                )

            if pd.isna(actual):
                unmatched_players.add((player, market, game_id))
                continue

            # Determine if bet hit
            over_hit = 1 if actual > line else 0
            under_hit = 1 if actual < line else 0

            # If exactly equal to line, it's a push (treat as neither)
            is_push = 1 if actual == line else 0

            results.append({
                'game_id': game_id,
                'season': season,
                'week': week,
                'away_team': over_row['away_team'],
                'home_team': over_row['home_team'],
                'commence_time': over_row['commence_time'],
                'player': player,
                'player_normalized': player_normalized,
                'market': market,
                'line': line,
                'over_odds': over_row['price'],
                'under_odds': under_row['price'],
                'snapshot_timestamp': over_row['snapshot_timestamp'],
                'actual_stat': actual,
                'over_hit': over_hit,
                'under_hit': under_hit,
                'is_push': is_push,
            })

    print(f"\nMatching complete:")
    print(f"  Matched: {len(results)} prop bets")
    print(f"  Unmatched games: {len(unmatched_games)}")
    print(f"  Unmatched players: {len(unmatched_players)}")

    if unmatched_players:
        print(f"\nSample unmatched players:")
        for player, market, game_id in list(unmatched_players)[:10]:
            print(f"    {player} ({market}) in {game_id}")

    return pd.DataFrame(results)


def calculate_bet_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting outcomes for each prop.

    Adds:
    - implied_prob_over: Implied probability from odds
    - implied_prob_under: Implied probability from odds
    - ev_over: Expected value if betting over
    - ev_under: Expected value if betting under
    """
    df = df.copy()

    # Convert American odds to implied probability
    def american_to_implied_prob(odds):
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    # Convert American odds to decimal
    def american_to_decimal(odds):
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    df['implied_prob_over'] = df['over_odds'].apply(american_to_implied_prob)
    df['implied_prob_under'] = df['under_odds'].apply(american_to_implied_prob)

    # Remove vig for fair probabilities (normalize to sum to 1)
    total_prob = df['implied_prob_over'] + df['implied_prob_under']
    df['fair_prob_over'] = df['implied_prob_over'] / total_prob
    df['fair_prob_under'] = df['implied_prob_under'] / total_prob

    # Calculate vig
    df['vig'] = total_prob - 1

    # Decimal odds
    df['decimal_over'] = df['over_odds'].apply(american_to_decimal)
    df['decimal_under'] = df['under_odds'].apply(american_to_decimal)

    # Calculate profit/loss (assuming $100 bet)
    df['profit_if_over'] = np.where(
        df['is_push'] == 0,
        np.where(df['over_hit'] == 1, (df['decimal_over'] - 1) * 100, -100),
        0  # Push returns stake
    )

    df['profit_if_under'] = np.where(
        df['is_push'] == 0,
        np.where(df['under_hit'] == 1, (df['decimal_under'] - 1) * 100, -100),
        0
    )

    return df


def create_backtest_summary(df: pd.DataFrame):
    """Create summary statistics for the backtest dataset"""
    print("\n" + "="*80)
    print("BACKTEST DATASET SUMMARY")
    print("="*80)

    print(f"\nTotal matched props: {len(df)}")
    print(f"Unique games: {df['game_id'].nunique()}")
    print(f"Unique players: {df['player'].nunique()}")

    print(f"\nSeasons: {sorted(df['season'].unique().tolist())}")
    for season in sorted(df['season'].unique()):
        season_data = df[df['season'] == season]
        print(f"  {season}: {len(season_data)} props, weeks {sorted(season_data['week'].unique().tolist())}")

    print(f"\nMarkets breakdown:")
    for market in df['market'].unique():
        market_data = df[df['market'] == market]
        hit_rate = market_data['over_hit'].mean() * 100
        avg_line = market_data['line'].mean()
        avg_actual = market_data['actual_stat'].mean()
        print(f"  {market}:")
        print(f"    Records: {len(market_data)}")
        print(f"    Avg line: {avg_line:.1f}")
        print(f"    Avg actual: {avg_actual:.1f}")
        print(f"    Over hit rate: {hit_rate:.1f}%")

    print(f"\nOdds analysis:")
    print(f"  Avg vig: {df['vig'].mean()*100:.2f}%")
    print(f"  Avg implied over prob: {df['implied_prob_over'].mean()*100:.1f}%")
    print(f"  Avg implied under prob: {df['implied_prob_under'].mean()*100:.1f}%")

    # If we just bet everything over
    print(f"\nBlind betting analysis:")
    total_profit_over = df['profit_if_over'].sum()
    total_profit_under = df['profit_if_under'].sum()
    roi_over = total_profit_over / (len(df) * 100) * 100
    roi_under = total_profit_under / (len(df) * 100) * 100
    print(f"  Bet all overs: ${total_profit_over:.2f} profit ({roi_over:.2f}% ROI)")
    print(f"  Bet all unders: ${total_profit_under:.2f} profit ({roi_under:.2f}% ROI)")


def main():
    print("="*80)
    print("MATCHING HISTORICAL ODDS TO NFLVERSE ACTUALS")
    print("="*80)

    # Load data
    odds_df = load_historical_odds()
    stats_df = load_nflverse_stats()
    schedule_df = load_schedule()

    # Match odds to actual stats
    matched_df = match_odds_to_actuals(odds_df, stats_df, schedule_df)

    if matched_df.empty:
        print("No matches found!")
        return

    # Calculate betting outcomes
    matched_df = calculate_bet_outcomes(matched_df)

    # Create summary
    create_backtest_summary(matched_df)

    # Save results
    output_dir = PROJECT_ROOT / 'data' / 'backtest'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'matched_odds_actuals.csv'
    matched_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved matched dataset: {output_file}")

    # Also save as parquet for efficiency
    parquet_file = output_dir / 'matched_odds_actuals.parquet'
    matched_df.to_parquet(parquet_file)
    print(f"✅ Saved parquet: {parquet_file}")

    return matched_df


if __name__ == '__main__':
    main()
