"""
NFL Elo Rating System
=====================

Implements FiveThirtyEight-style Elo ratings for NFL teams using nflverse schedule data.

Theory:
-------
Elo ratings measure team strength on a continuous scale where:
- 1500 = league average team
- +25 Elo points ≈ 1 point spread advantage
- Typical range: 1350 (worst) to 1700 (best)

Key Features:
-------------
1. Margin of Victory adjustment - larger wins increase rating change
2. Home field advantage - configurable HFA boost (default: 48 Elo ≈ 1.9 points)
3. Season regression - ratings regress toward mean between seasons
4. Persistence - ratings saved to JSON for weekly updates

Applications:
-------------
- Team power rankings for game line predictions
- Spread prediction as baseline for edge detection
- Feature input for player prop models (team context)

Usage:
------
    from nfl_quant.models.elo_ratings import EloRatingSystem

    elo = EloRatingSystem()
    elo.initialize_from_schedule(schedule_df, seasons=[2023, 2024, 2025])

    # Get predictions
    spread = elo.get_spread_prediction('KC', 'BUF')  # Positive = KC favored
    win_prob = elo.get_win_probability('KC', 'BUF')

    # Save ratings
    elo.save_ratings('data/models/elo_ratings.json')
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class EloRatingSystem:
    """
    NFL Elo rating system using nflverse schedule data.

    Based on FiveThirtyEight's NFL Elo methodology with adjustments
    for nflverse data availability.

    Attributes:
        ratings: Current Elo ratings for all teams
        k_factor: Rating adjustment speed (default: 20)
        home_advantage: Home field advantage in Elo points (default: 48)
        initial_rating: Starting rating for new teams (default: 1500)
        season_regression: Regression to mean between seasons (default: 0.33)
    """

    # NFL team abbreviations (32 teams)
    NFL_TEAMS = [
        'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
        'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
        'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
        'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
    ]

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 48.0,
        initial_rating: float = 1500.0,
        season_regression: float = 0.33,
        mov_multiplier: bool = True,
    ):
        """
        Initialize Elo rating system.

        Args:
            k_factor: Controls how quickly ratings change (higher = more volatile)
            home_advantage: Home field advantage in Elo points (48 ≈ 1.9 point spread)
            initial_rating: Starting rating for all teams
            season_regression: Fraction to regress toward mean between seasons
            mov_multiplier: Whether to adjust K based on margin of victory
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_regression = season_regression
        self.mov_multiplier = mov_multiplier

        # Initialize all teams to average rating
        self.ratings: Dict[str, float] = {
            team: initial_rating for team in self.NFL_TEAMS
        }

        # Track rating history for analysis
        self.history: List[Dict] = []

        # Track last update
        self.last_updated: Optional[str] = None
        self.seasons_processed: List[int] = []

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected win probability for team A.

        Uses logistic function: P(A wins) = 1 / (1 + 10^((Rb - Ra) / 400))

        Args:
            rating_a: Elo rating of team A
            rating_b: Elo rating of team B

        Returns:
            Expected probability that team A wins (0 to 1)
        """
        exponent = (rating_b - rating_a) / 400.0
        return 1.0 / (1.0 + math.pow(10, exponent))

    def get_win_probability(
        self,
        team_a: str,
        team_b: str,
        neutral: bool = False,
        home_team: str = None,
    ) -> float:
        """
        Get win probability for team A vs team B.

        Args:
            team_a: First team abbreviation
            team_b: Second team abbreviation
            neutral: If True, no home field advantage applied
            home_team: Which team is home (if not neutral). If None, team_a is home.

        Returns:
            Probability that team A wins (0 to 1)
        """
        rating_a = self.ratings.get(team_a, self.initial_rating)
        rating_b = self.ratings.get(team_b, self.initial_rating)

        if not neutral:
            if home_team is None or home_team == team_a:
                rating_a += self.home_advantage
            elif home_team == team_b:
                rating_b += self.home_advantage

        return self.expected_score(rating_a, rating_b)

    def get_spread_prediction(self, home_team: str, away_team: str) -> float:
        """
        Convert Elo difference to point spread.

        Uses conversion: 25 Elo points ≈ 1 point spread
        (FiveThirtyEight uses this same conversion)

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Predicted spread (positive = home team favored)
        """
        home_elo = self.ratings.get(home_team, self.initial_rating) + self.home_advantage
        away_elo = self.ratings.get(away_team, self.initial_rating)
        return (home_elo - away_elo) / 25.0

    def get_total_prediction(
        self,
        home_team: str,
        away_team: str,
        league_avg_total: float = 45.5,
    ) -> float:
        """
        Predict game total based on team ratings.

        Higher-rated teams tend to be in higher-scoring games.
        This is a rough approximation - weather and other factors matter more.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            league_avg_total: League average total (default: 45.5)

        Returns:
            Predicted total points
        """
        home_elo = self.ratings.get(home_team, self.initial_rating)
        away_elo = self.ratings.get(away_team, self.initial_rating)

        # Teams above/below average adjust total slightly
        avg_elo = (home_elo + away_elo) / 2.0
        elo_diff_from_avg = avg_elo - self.initial_rating

        # Each 100 Elo above average adds ~1.5 points to total
        adjustment = elo_diff_from_avg / 100.0 * 1.5

        return league_avg_total + adjustment

    def _calculate_mov_multiplier(
        self,
        margin: float,
        elo_diff: float,
    ) -> float:
        """
        Calculate margin of victory multiplier.

        Larger margins lead to larger rating changes, but with diminishing returns.
        Based on FiveThirtyEight's formula.

        Args:
            margin: Point margin (positive for winner)
            elo_diff: Elo difference (winner - loser) BEFORE game

        Returns:
            Multiplier for K factor (typically 0.7 to 2.5)
        """
        if not self.mov_multiplier:
            return 1.0

        # Log-based formula with autocorrelation adjustment
        # Prevents winning by a lot against a bad team from inflating ratings
        abs_margin = abs(margin)
        if abs_margin == 0:
            return 1.0

        # Base multiplier increases with margin (log scale)
        base = math.log(abs_margin + 1) * 2.2

        # Autocorrelation adjustment: reduce multiplier if result was expected
        # If favorite won by a lot, they get less credit
        adjustment = 1.0
        if elo_diff > 0:  # Winner was favored
            adjustment = 2.2 / (2.2 + 0.001 * elo_diff)

        return base * adjustment

    def update_game(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        season: int = None,
        week: int = None,
        game_id: str = None,
    ) -> Tuple[float, float]:
        """
        Update ratings after a single game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_score: Home team points scored
            away_score: Away team points scored
            season: Season year (for history tracking)
            week: Week number (for history tracking)
            game_id: Game ID (for history tracking)

        Returns:
            Tuple of (home_rating_change, away_rating_change)
        """
        # Skip if scores are missing (future games)
        if pd.isna(home_score) or pd.isna(away_score):
            return (0.0, 0.0)

        # Get current ratings with HFA
        home_elo = self.ratings.get(home_team, self.initial_rating) + self.home_advantage
        away_elo = self.ratings.get(away_team, self.initial_rating)

        # Calculate expected scores
        home_expected = self.expected_score(home_elo, away_elo)
        away_expected = 1.0 - home_expected

        # Determine actual result (1 = win, 0.5 = tie, 0 = loss)
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
            margin = home_score - away_score
            elo_diff = home_elo - away_elo
        elif away_score > home_score:
            home_actual = 0.0
            away_actual = 1.0
            margin = away_score - home_score
            elo_diff = away_elo - home_elo
        else:
            home_actual = 0.5
            away_actual = 0.5
            margin = 0
            elo_diff = 0

        # Calculate MOV multiplier
        mov_mult = self._calculate_mov_multiplier(margin, elo_diff)

        # Calculate rating changes
        k = self.k_factor * mov_mult
        home_change = k * (home_actual - home_expected)
        away_change = k * (away_actual - away_expected)

        # Update ratings (without HFA)
        old_home = self.ratings.get(home_team, self.initial_rating)
        old_away = self.ratings.get(away_team, self.initial_rating)

        self.ratings[home_team] = old_home + home_change
        self.ratings[away_team] = old_away + away_change

        # Track history
        self.history.append({
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_elo_before': old_home,
            'away_elo_before': old_away,
            'home_elo_after': self.ratings[home_team],
            'away_elo_after': self.ratings[away_team],
            'home_change': home_change,
            'away_change': away_change,
            'mov_multiplier': mov_mult,
        })

        return (home_change, away_change)

    def apply_season_regression(self) -> None:
        """
        Apply between-season regression to the mean.

        Teams regress toward average (1500) between seasons to account for:
        - Roster turnover
        - Coaching changes
        - Draft picks
        - Free agency

        Typical regression is 1/3 toward the mean.
        """
        for team in self.ratings:
            current = self.ratings[team]
            regressed = current + self.season_regression * (self.initial_rating - current)
            self.ratings[team] = regressed
            logger.debug(f"{team}: {current:.0f} -> {regressed:.0f}")

    def initialize_from_schedule(
        self,
        schedule_df: pd.DataFrame,
        seasons: List[int] = None,
    ) -> None:
        """
        Initialize ratings by processing historical games.

        Args:
            schedule_df: NFLverse schedule DataFrame with columns:
                - game_id, season, week, game_type
                - home_team, away_team
                - home_score, away_score
            seasons: List of seasons to process (default: all in DataFrame)
        """
        # Filter to regular season and completed games
        df = schedule_df.copy()

        if seasons:
            df = df[df['season'].isin(seasons)]
        else:
            seasons = sorted(df['season'].unique())

        # Only process completed games (have scores)
        df = df[df['home_score'].notna() & df['away_score'].notna()]

        # Process only regular season and playoffs
        if 'game_type' in df.columns:
            df = df[df['game_type'].isin(['REG', 'POST', 'WC', 'DIV', 'CON', 'SB'])]

        # Sort by season, week
        df = df.sort_values(['season', 'week'])

        logger.info(f"Processing {len(df)} games from seasons {seasons}")

        current_season = None
        for _, game in df.iterrows():
            # Apply regression between seasons
            if current_season is not None and game['season'] != current_season:
                logger.info(f"Season {current_season} complete. Applying regression.")
                self.apply_season_regression()

            current_season = game['season']

            self.update_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                home_score=game['home_score'],
                away_score=game['away_score'],
                season=game['season'],
                week=game.get('week'),
                game_id=game.get('game_id'),
            )

        self.last_updated = datetime.now().isoformat()
        self.seasons_processed = list(seasons)

        logger.info(f"Elo ratings initialized. Processed seasons: {seasons}")

    def get_rankings(self) -> pd.DataFrame:
        """
        Get current team rankings sorted by Elo.

        Returns:
            DataFrame with columns: rank, team, elo, diff_from_avg
        """
        rankings = []
        for team, elo in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True):
            rankings.append({
                'team': team,
                'elo': elo,
                'diff_from_avg': elo - self.initial_rating,
            })

        df = pd.DataFrame(rankings)
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'team', 'elo', 'diff_from_avg']]

    def get_rating(self, team: str) -> float:
        """Get Elo rating for a team."""
        return self.ratings.get(team, self.initial_rating)

    def get_elo_diff(self, home_team: str, away_team: str) -> float:
        """Get Elo difference (home - away, including HFA)."""
        home_elo = self.ratings.get(home_team, self.initial_rating) + self.home_advantage
        away_elo = self.ratings.get(away_team, self.initial_rating)
        return home_elo - away_elo

    def save_ratings(self, filepath: str) -> None:
        """
        Save ratings to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            'ratings': self.ratings,
            'config': {
                'k_factor': self.k_factor,
                'home_advantage': self.home_advantage,
                'initial_rating': self.initial_rating,
                'season_regression': self.season_regression,
                'mov_multiplier': self.mov_multiplier,
            },
            'metadata': {
                'last_updated': self.last_updated,
                'seasons_processed': self.seasons_processed,
                'games_processed': len(self.history),
            }
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved Elo ratings to {filepath}")

    def load_ratings(self, filepath: str) -> bool:
        """
        Load ratings from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Ratings file not found: {filepath}")
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.ratings = data['ratings']

            if 'config' in data:
                self.k_factor = data['config'].get('k_factor', self.k_factor)
                self.home_advantage = data['config'].get('home_advantage', self.home_advantage)
                self.initial_rating = data['config'].get('initial_rating', self.initial_rating)
                self.season_regression = data['config'].get('season_regression', self.season_regression)
                self.mov_multiplier = data['config'].get('mov_multiplier', self.mov_multiplier)

            if 'metadata' in data:
                self.last_updated = data['metadata'].get('last_updated')
                self.seasons_processed = data['metadata'].get('seasons_processed', [])

            logger.info(f"Loaded Elo ratings from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            return False

    def update_weekly(self, schedule_df: pd.DataFrame, season: int, week: int) -> int:
        """
        Update ratings with a single week's games.

        Args:
            schedule_df: NFLverse schedule DataFrame
            season: Season year
            week: Week number

        Returns:
            Number of games processed
        """
        # Filter to specific week
        df = schedule_df[
            (schedule_df['season'] == season) &
            (schedule_df['week'] == week) &
            schedule_df['home_score'].notna() &
            schedule_df['away_score'].notna()
        ]

        games_processed = 0
        for _, game in df.iterrows():
            self.update_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                home_score=game['home_score'],
                away_score=game['away_score'],
                season=season,
                week=week,
                game_id=game.get('game_id'),
            )
            games_processed += 1

        self.last_updated = datetime.now().isoformat()
        logger.info(f"Updated Elo ratings with {games_processed} games from week {week}")

        return games_processed

    def get_history_df(self) -> pd.DataFrame:
        """Get rating history as DataFrame."""
        return pd.DataFrame(self.history)


def load_elo_ratings(filepath: str = None) -> EloRatingSystem:
    """
    Load or create Elo rating system.

    Args:
        filepath: Path to saved ratings (default: data/models/elo_ratings.json)

    Returns:
        EloRatingSystem instance
    """
    if filepath is None:
        from nfl_quant.config_paths import PROJECT_ROOT
        filepath = PROJECT_ROOT / 'data' / 'models' / 'elo_ratings.json'

    elo = EloRatingSystem()
    elo.load_ratings(str(filepath))
    return elo


def initialize_elo_from_nflverse(
    seasons: List[int] = None,
    save_path: str = None,
) -> EloRatingSystem:
    """
    Initialize Elo ratings from nflverse schedule data.

    Args:
        seasons: Seasons to process (default: [2023, 2024, 2025])
        save_path: Path to save ratings (default: data/models/elo_ratings.json)

    Returns:
        Initialized EloRatingSystem
    """
    from nfl_quant.config_paths import PROJECT_ROOT

    if seasons is None:
        seasons = [2023, 2024, 2025]

    if save_path is None:
        save_path = PROJECT_ROOT / 'data' / 'models' / 'elo_ratings.json'

    # Load schedule
    schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if not schedule_path.exists():
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.csv'

    if schedule_path.suffix == '.parquet':
        schedule_df = pd.read_parquet(schedule_path)
    else:
        schedule_df = pd.read_csv(schedule_path)

    # Initialize and save
    elo = EloRatingSystem()
    elo.initialize_from_schedule(schedule_df, seasons=seasons)
    elo.save_ratings(str(save_path))

    return elo


# Module-level cache for singleton pattern
_elo_instance: Optional[EloRatingSystem] = None


def get_elo_system() -> EloRatingSystem:
    """
    Get or create singleton Elo rating system.

    Returns:
        EloRatingSystem instance
    """
    global _elo_instance

    if _elo_instance is None:
        from nfl_quant.config_paths import PROJECT_ROOT
        filepath = PROJECT_ROOT / 'data' / 'models' / 'elo_ratings.json'

        _elo_instance = EloRatingSystem()
        if filepath.exists():
            _elo_instance.load_ratings(str(filepath))
        else:
            # Initialize from scratch if no saved ratings
            logger.warning("No saved Elo ratings found. Initializing from schedule...")
            _elo_instance = initialize_elo_from_nflverse()

    return _elo_instance
