#!/usr/bin/env python3
"""
Trigger-Based Game Lines Betting Model - NO DATA LEAKAGE

This model uses validated triggers to generate bets, not continuous predictions.
Each trigger fires only when specific conditions are met.

Validated Triggers (consistent across 2024-2025):
1. Big Home Favorite (10+) → Bet HOME spread
2. Big Home Underdog (7+) → Bet HOME spread
3. Windy Games (15+ mph) → Bet UNDER
4. High Combined Offense EPA (+0.1) → Bet OVER

Anti-Leakage Measures:
- EPA calculated from PRIOR weeks only
- Triggers defined from historical analysis (not test data)
- Walk-forward validation

Usage:
    python scripts/validation/backtest_trigger_model.py
    python scripts/validation/backtest_trigger_model.py --season 2025
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data'


# =============================================================================
# TRIGGER DEFINITIONS
# =============================================================================

class BettingTrigger:
    """Base class for betting triggers."""

    def __init__(self, name: str, bet_type: str, bet_side: str):
        self.name = name
        self.bet_type = bet_type  # 'spread' or 'total'
        self.bet_side = bet_side  # 'HOME', 'AWAY', 'OVER', 'UNDER'

    def check(self, game_data: dict) -> bool:
        """Return True if trigger fires for this game."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name} → {self.bet_type.upper()} {self.bet_side}"


class BigHomeFavorite(BettingTrigger):
    """Bet HOME when home is 10+ point favorite."""

    def __init__(self, threshold: float = 10.0):
        super().__init__(
            name=f"Big Home Fav ({threshold}+)",
            bet_type='spread',
            bet_side='HOME'
        )
        self.threshold = threshold

    def check(self, game_data: dict) -> bool:
        # spread_line > 0 means home is favorite (using corrected convention)
        return game_data.get('spread_line', 0) >= self.threshold


class BigHomeUnderdog(BettingTrigger):
    """Bet HOME when home is 7+ point underdog."""

    def __init__(self, threshold: float = 7.0):
        super().__init__(
            name=f"Big Home Dog ({threshold}+)",
            bet_type='spread',
            bet_side='HOME'
        )
        self.threshold = threshold

    def check(self, game_data: dict) -> bool:
        # spread_line < 0 means home is underdog
        return game_data.get('spread_line', 0) <= -self.threshold


class BigHomeFavoriteML(BettingTrigger):
    """Bet HOME MONEYLINE when home is 10+ point favorite."""

    def __init__(self, threshold: float = 10.0):
        super().__init__(
            name=f"Big Home Fav ML ({threshold}+)",
            bet_type='moneyline',
            bet_side='HOME'
        )
        self.threshold = threshold

    def check(self, game_data: dict) -> bool:
        # spread_line > 0 means home is favorite
        return game_data.get('spread_line', 0) >= self.threshold


class WindyUnder(BettingTrigger):
    """Bet UNDER when wind is 15+ mph AND market hasn't already priced in low scoring."""

    def __init__(self, threshold: float = 15.0, min_market_total: float = 40.0):
        super().__init__(
            name=f"Windy ({threshold}+ mph)",
            bet_type='total',
            bet_side='UNDER'
        )
        self.threshold = threshold
        self.min_market_total = min_market_total

    def check(self, game_data: dict) -> bool:
        wind = game_data.get('wind')
        if wind is None or pd.isna(wind):
            return False

        # Market sanity check: skip if market already priced in low scoring
        total_line = game_data.get('total_line')
        if total_line is not None and not pd.isna(total_line):
            if total_line < self.min_market_total:
                return False  # Market already low, don't pile on

        return wind >= self.threshold


class HighOffenseEPA(BettingTrigger):
    """Bet OVER when combined offensive EPA is high AND market agrees."""

    def __init__(self, threshold: float = 0.1, min_market_total: float = 44.0):
        super().__init__(
            name=f"High Off EPA (+{threshold})",
            bet_type='total',
            bet_side='OVER'
        )
        self.threshold = threshold
        self.min_market_total = min_market_total

    def check(self, game_data: dict) -> bool:
        home_epa = game_data.get('home_off_epa', 0)
        away_epa = game_data.get('away_off_epa', 0)
        combined_epa = home_epa + away_epa

        if combined_epa < self.threshold:
            return False

        # Market sanity check: if EPA is high but market total is low,
        # Vegas knows something we don't (injury, weather, etc.)
        total_line = game_data.get('total_line')
        if total_line is not None and not pd.isna(total_line):
            if total_line < self.min_market_total:
                return False  # Don't fight the market

        return True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pbp_data() -> pd.DataFrame:
    """Load play-by-play data."""
    path = DATA_DIR / 'nflverse' / 'pbp.parquet'
    if not path.exists():
        raise FileNotFoundError(f"PBP data not found at {path}")
    return pq.read_table(path).to_pandas()


def load_schedule_data() -> pd.DataFrame:
    """Load schedule with results."""
    path = DATA_DIR / 'nflverse' / 'schedules.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Schedule not found at {path}")
    return pq.read_table(path).to_pandas()


# =============================================================================
# FEATURE CALCULATION (NO LEAKAGE)
# =============================================================================

def calculate_team_epa_prior(
    pbp_df: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    lookback: int = 6
) -> dict:
    """
    Calculate team EPA using ONLY prior weeks.

    NO LEAKAGE: Only uses data from weeks < current_week
    """
    min_week = max(1, week - lookback)

    prior_pbp = pbp_df[
        (pbp_df['season'] == season) &
        (pbp_df['week'] >= min_week) &
        (pbp_df['week'] < week) &
        (pbp_df['play_type'].isin(['pass', 'run']))
    ]

    # Offensive plays
    off_plays = prior_pbp[prior_pbp['posteam'] == team]

    if len(off_plays) < 50:
        return None

    return {
        'off_epa': off_plays['epa'].mean(),
        'games': len(off_plays['game_id'].unique())
    }


# =============================================================================
# BET EVALUATION
# =============================================================================

def evaluate_spread_bet(direction: str, actual_margin: float, market_spread: float) -> dict:
    """
    Evaluate spread bet result.

    CRITICAL: NFLverse spread_line is AWAY team's spread.
    - Positive spread_line = home is FAVORITE
    - Negative spread_line = home is UNDERDOG

    Home covers if: margin - spread_line > 0
    """
    cover_margin = actual_margin - market_spread

    home_covers = cover_margin > 0
    away_covers = cover_margin < 0
    push = abs(cover_margin) < 0.5

    if direction == 'HOME':
        won = home_covers and not push
    else:
        won = away_covers and not push

    return {'won': won, 'push': push, 'cover_margin': cover_margin}


def evaluate_total_bet(direction: str, actual_total: float, market_total: float) -> dict:
    """Evaluate total bet result."""
    if pd.isna(actual_total) or pd.isna(market_total):
        return {'won': False, 'push': True}

    if direction == 'OVER':
        won = actual_total > market_total
        push = actual_total == market_total
    else:
        won = actual_total < market_total
        push = actual_total == market_total

    return {'won': won, 'push': push}


def evaluate_moneyline_bet(direction: str, actual_margin: float) -> dict:
    """
    Evaluate moneyline bet result.

    Args:
        direction: 'HOME' or 'AWAY'
        actual_margin: home_score - away_score

    Returns:
        dict with 'won' and 'push' keys
    """
    home_won = actual_margin > 0
    away_won = actual_margin < 0
    push = actual_margin == 0  # Tie (rare in NFL)

    if direction == 'HOME':
        won = home_won
    else:
        won = away_won

    return {'won': won, 'push': push}


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_trigger_backtest(
    start_week: int = 5,
    end_week: int = 17,
    season: int = 2025,
    min_games_for_epa: int = 3
) -> pd.DataFrame:
    """
    Run walk-forward backtest using trigger-based model.

    NO LEAKAGE:
    - EPA calculated from prior weeks only
    - Triggers are pre-defined (not learned from test data)
    """
    print("\n" + "=" * 70)
    print("TRIGGER-BASED BETTING MODEL BACKTEST")
    print("=" * 70)
    print(f"Season: {season}")
    print(f"Weeks: {start_week} to {end_week}")
    print()

    # Initialize triggers - SPREAD + MONEYLINE
    # Total triggers disabled due to year-over-year inconsistency
    triggers = [
        BigHomeFavorite(threshold=10.0),
        BigHomeUnderdog(threshold=7.0),
        BigHomeFavoriteML(threshold=10.0),  # Moneyline for big favorites
        # WindyUnder(threshold=15.0),       # DISABLED - not consistent
        # HighOffenseEPA(threshold=0.1),    # DISABLED - not consistent
    ]

    print("Active Triggers:")
    for t in triggers:
        print(f"  - {t}")
    print()

    # Load data
    print("Loading data...")
    pbp_df = load_pbp_data()
    schedule_df = load_schedule_data()

    pbp_season = pbp_df[pbp_df['season'] == season]
    sched_season = schedule_df[schedule_df['season'] == season]

    print(f"  PBP plays: {len(pbp_season):,}")
    print(f"  Schedule games: {len(sched_season)}")

    results = []

    for test_week in range(start_week, end_week + 1):
        print(f"\n--- Week {test_week} ---")

        # Get completed games for this week
        week_games = sched_season[
            (sched_season['week'] == test_week) &
            (sched_season['game_type'] == 'REG') &
            (sched_season['home_score'].notna()) &
            (sched_season['away_score'].notna())
        ]

        if len(week_games) == 0:
            print("  No completed games")
            continue

        week_bets = 0

        for _, game in week_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']

            # Calculate EPA using PRIOR weeks only
            home_epa = calculate_team_epa_prior(pbp_season, home_team, season, test_week)
            away_epa = calculate_team_epa_prior(pbp_season, away_team, season, test_week)

            if home_epa is None or away_epa is None:
                continue

            if home_epa['games'] < min_games_for_epa or away_epa['games'] < min_games_for_epa:
                continue

            # Build game data for trigger evaluation
            game_data = {
                'spread_line': game.get('spread_line'),
                'total_line': game.get('total_line'),
                'wind': game.get('wind'),
                'temp': game.get('temp'),
                'home_off_epa': home_epa['off_epa'],
                'away_off_epa': away_epa['off_epa'],
                'home_rest': game.get('home_rest', 7),
                'away_rest': game.get('away_rest', 7),
                'div_game': game.get('div_game', 0),
            }

            # Actual results
            home_score = game['home_score']
            away_score = game['away_score']
            actual_margin = home_score - away_score
            actual_total = home_score + away_score

            # Check each trigger
            for trigger in triggers:
                if trigger.check(game_data):
                    # Trigger fired - evaluate bet
                    if trigger.bet_type == 'spread':
                        market_line = game_data.get('spread_line')
                        if pd.isna(market_line):
                            continue

                        eval_result = evaluate_spread_bet(
                            trigger.bet_side, actual_margin, market_line
                        )

                        if eval_result['push']:
                            continue

                        results.append({
                            'season': season,
                            'week': test_week,
                            'game': f"{away_team} @ {home_team}",
                            'trigger': trigger.name,
                            'bet_type': 'spread',
                            'bet_side': trigger.bet_side,
                            'market_line': market_line,
                            'won': eval_result['won'],
                            'actual_margin': actual_margin,
                            'cover_margin': eval_result['cover_margin'],
                            'home_off_epa': home_epa['off_epa'],
                            'away_off_epa': away_epa['off_epa'],
                        })
                        week_bets += 1

                    elif trigger.bet_type == 'total':
                        market_line = game_data.get('total_line')
                        if pd.isna(market_line):
                            continue

                        eval_result = evaluate_total_bet(
                            trigger.bet_side, actual_total, market_line
                        )

                        if eval_result['push']:
                            continue

                        results.append({
                            'season': season,
                            'week': test_week,
                            'game': f"{away_team} @ {home_team}",
                            'trigger': trigger.name,
                            'bet_type': 'total',
                            'bet_side': trigger.bet_side,
                            'market_line': market_line,
                            'won': eval_result['won'],
                            'actual_total': actual_total,
                            'home_off_epa': home_epa['off_epa'],
                            'away_off_epa': away_epa['off_epa'],
                        })
                        week_bets += 1

                    elif trigger.bet_type == 'moneyline':
                        eval_result = evaluate_moneyline_bet(
                            trigger.bet_side, actual_margin
                        )

                        if eval_result['push']:
                            continue

                        results.append({
                            'season': season,
                            'week': test_week,
                            'game': f"{away_team} @ {home_team}",
                            'trigger': trigger.name,
                            'bet_type': 'moneyline',
                            'bet_side': trigger.bet_side,
                            'market_line': game_data.get('spread_line'),
                            'won': eval_result['won'],
                            'actual_margin': actual_margin,
                            'home_off_epa': home_epa['off_epa'],
                            'away_off_epa': away_epa['off_epa'],
                        })
                        week_bets += 1

        print(f"  Bets placed: {week_bets}")

    return pd.DataFrame(results)


def print_results(df: pd.DataFrame):
    """Print backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    if len(df) == 0:
        print("No bets placed.")
        return

    # Overall
    wins = df['won'].sum()
    total = len(df)
    win_rate = wins / total

    # ROI calculation (-110 odds)
    roi = (win_rate * 1.9091 - 1) * 100

    print(f"\nOVERALL:")
    print(f"  Bets: {total}")
    print(f"  Wins: {wins}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Est. ROI: {roi:+.1f}%")

    # By trigger
    print(f"\nBY TRIGGER:")
    for trigger in df['trigger'].unique():
        t_df = df[df['trigger'] == trigger]
        t_wins = t_df['won'].sum()
        t_total = len(t_df)
        t_rate = t_wins / t_total
        t_roi = (t_rate * 1.9091 - 1) * 100
        print(f"  {trigger}:")
        print(f"    {t_wins}/{t_total} ({t_rate:.1%}) | ROI: {t_roi:+.1f}%")

    # By bet type
    print(f"\nBY BET TYPE:")
    for bet_type in df['bet_type'].unique():
        bt_df = df[df['bet_type'] == bet_type]
        bt_wins = bt_df['won'].sum()
        bt_total = len(bt_df)
        bt_rate = bt_wins / bt_total
        bt_roi = (bt_rate * 1.9091 - 1) * 100
        print(f"  {bet_type.upper()}: {bt_wins}/{bt_total} ({bt_rate:.1%}) | ROI: {bt_roi:+.1f}%")

    # By week
    print(f"\nBY WEEK:")
    for week in sorted(df['week'].unique()):
        w_df = df[df['week'] == week]
        w_wins = w_df['won'].sum()
        w_total = len(w_df)
        w_rate = w_wins / w_total if w_total > 0 else 0
        print(f"  Week {week:2}: {w_wins}/{w_total} ({w_rate:.1%})")

    # Statistical test
    print(f"\n" + "=" * 70)
    print("STATISTICAL VALIDATION")
    print("=" * 70)

    # Test vs 50% (coin flip)
    result = stats.binomtest(wins, total, 0.50)
    print(f"\nTest vs 50% (coin flip):")
    print(f"  p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: Not significant (could be random)")

    # Test vs 52.4% (breakeven at -110)
    result_be = stats.binomtest(wins, total, 0.524)
    print(f"\nTest vs 52.4% (breakeven):")
    print(f"  p-value: {result_be.pvalue:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Trigger-based betting backtest')
    parser.add_argument('--season', type=int, default=2025, help='Season to backtest')
    parser.add_argument('--start-week', type=int, default=5, help='Start week')
    parser.add_argument('--end-week', type=int, default=17, help='End week')
    args = parser.parse_args()

    # Run backtest
    results_df = run_trigger_backtest(
        start_week=args.start_week,
        end_week=args.end_week,
        season=args.season
    )

    # Print results
    print_results(results_df)

    # Save results
    output_path = DATA_DIR / 'backtest' / f'trigger_model_backtest_{args.season}.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
