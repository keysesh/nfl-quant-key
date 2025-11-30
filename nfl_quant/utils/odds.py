"""Odds conversion, EV calculation, and Kelly sizing."""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from nfl_quant.schemas import BetSizing, OddsRecord

logger = logging.getLogger(__name__)


class OddsEngine:
    """Handles odds conversion, EV calculation, and Kelly sizing."""

    @staticmethod
    def american_to_implied_prob(american_odds: int) -> float:
        """Convert American odds to implied probability.

        Args:
            american_odds: American odds (e.g., -110, +120)

        Returns:
            Implied probability [0, 1]
        """
        if american_odds < 0:
            # Negative odds: implied_prob = |odds| / (|odds| + 100)
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            # Positive odds: implied_prob = 100 / (odds + 100)
            return 100 / (american_odds + 100)

    @staticmethod
    def implied_prob_to_american(prob: float) -> int:
        """Convert implied probability to American odds.

        Args:
            prob: Probability [0, 1]

        Returns:
            American odds (rounded integer)
        """
        if prob >= 0.5:
            # Favorite (negative odds)
            odds = -100 * prob / (1 - prob)
            return int(round(odds))
        else:
            # Underdog (positive odds)
            odds = 100 * (1 - prob) / prob
            return int(round(odds))

    @staticmethod
    def calculate_ev(win_prob: float, implied_prob: float) -> float:
        """Calculate Expected Value as percentage.

        Args:
            win_prob: True win probability [0, 1]
            implied_prob: Implied probability from odds [0, 1]

        Returns:
            EV percentage (positive = profitable)
        """
        if implied_prob == 0:
            return 0.0
        ev = (win_prob / implied_prob) - 1.0
        return ev * 100

    @staticmethod
    def kelly_fraction(win_prob: float, implied_prob: float, kelly_fraction: float = 0.5) -> float:
        """Calculate Kelly Criterion bet fraction.

        Formula: f = (b*p - q) / b
        Where:
          b = odds (payoff ratio)
          p = win probability
          q = 1 - p (loss probability)
          f = fraction of bankroll

        Args:
            win_prob: True win probability [0, 1]
            implied_prob: Implied probability from odds [0, 1]
            kelly_fraction: Fraction of full Kelly (0.5 = half-Kelly)

        Returns:
            Fraction of bankroll to wager
        """
        if implied_prob == 0 or implied_prob >= 1.0:
            return 0.0

        # Calculate payoff ratio (simplified: assuming -110 style odds)
        # For -110: payoff = 100/110 ≈ 0.909
        payoff_ratio = (1 - implied_prob) / implied_prob

        # Full Kelly
        q = 1 - win_prob
        full_kelly = (payoff_ratio * win_prob - q) / payoff_ratio

        # Avoid overbetting
        full_kelly = max(0, min(full_kelly, 0.25))  # Cap at 25% Kelly

        # Apply Kelly fraction (typically 0.25-0.5)
        return full_kelly * kelly_fraction

    def size_bet(
        self,
        game_id: str,
        side: str,
        american_odds: int,
        win_prob: float,
        bankroll: float,
        kelly_fraction: float = 0.5,
        max_bet_pct: float = 5.0,
    ) -> BetSizing:
        """Calculate bet sizing recommendation.

        Args:
            game_id: Game ID
            side: Side string (e.g., "home_spread")
            american_odds: American odds
            win_prob: True win probability
            bankroll: Total bankroll
            kelly_fraction: Kelly fraction (0.5 for half-Kelly)
            max_bet_pct: Max bet as % of bankroll

        Returns:
            BetSizing recommendation
        """
        implied_prob = self.american_to_implied_prob(american_odds)
        ev_pct = self.calculate_ev(win_prob, implied_prob)
        kelly_pct = self.kelly_fraction(win_prob, implied_prob, kelly_fraction) * 100

        # Size bet
        max_bet_amount = bankroll * (max_bet_pct / 100)
        kelly_bet_amount = bankroll * (kelly_pct / 100)
        suggested_bet = min(kelly_bet_amount, max_bet_amount)

        # Don't bet if negative EV
        if ev_pct < 0:
            suggested_bet = 0.0
            kelly_pct = 0.0

        # Calculate potential outcomes
        if american_odds < 0:
            potential_win = suggested_bet * (100 / abs(american_odds))
            potential_loss = suggested_bet
        else:
            potential_win = suggested_bet * (american_odds / 100)
            potential_loss = suggested_bet

        return BetSizing(
            game_id=game_id,
            side=side,
            american_odds=american_odds,
            fair_odds=None,
            implied_prob=implied_prob,
            win_prob=win_prob,
            ev_pct=ev_pct,
            kelly_fraction=kelly_fraction,
            kelly_pct=kelly_pct,
            suggested_bet_amount=suggested_bet,
            max_suggested_bet=max_bet_amount,
            potential_win=potential_win,
            potential_loss=potential_loss,
        )

    @staticmethod
    def load_odds_csv(file_path: str) -> list[OddsRecord]:
        """Load odds from CSV file.

        Expected columns: game_id, side, american_odds

        Args:
            file_path: Path to CSV

        Returns:
            List of OddsRecord objects
        """
        records = []
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                record = OddsRecord(
                    game_id=str(row["game_id"]),
                    side=str(row["side"]),
                    american_odds=int(row["american_odds"]),
                )
                records.append(record)
            logger.info(f"Loaded {len(records)} odds records from {file_path}")
            return records
        except Exception as e:
            logger.error(f"Failed to load odds from {file_path}: {e}")
            raise

    @staticmethod
    def validate_odds_schema(file_path: str) -> bool:
        """Validate odds CSV schema.

        Args:
            file_path: Path to CSV

        Returns:
            True if valid
        """
        try:
            df = pd.read_csv(file_path)
            required_cols = {"game_id", "side", "american_odds"}
            if not required_cols.issubset(set(df.columns)):
                logger.error(f"Missing required columns. Expected: {required_cols}")
                return False

            # Check for malformed rows
            for _, row in df.iterrows():
                if pd.isna(row["game_id"]) or pd.isna(row["american_odds"]):
                    logger.error("Found NaN in required columns")
                    return False
                if row["american_odds"] == 0:
                    logger.error("Found invalid odds (zero)")
                    return False

            logger.info(f"Odds schema validation passed for {file_path}")
            return True
        except Exception as e:
            logger.error(f"Odds validation failed: {e}")
            return False


def load_game_status_map(week: int, season: int = 2025) -> Dict[str, str]:
    """Load NFLverse game status for given week.

    Args:
        week: NFL week number
        season: NFL season year

    Returns:
        Dictionary mapping game_id -> status
        Status values: "pre_game", "in_progress", "complete"
    """
    try:
        from pathlib import Path

        # Load schedule from R-fetched NFLverse data
        nflverse_dir = Path("data/nflverse")
        sched_file = nflverse_dir / "schedules.parquet"
        if not sched_file.exists():
            sched_file = nflverse_dir / "schedules.csv"
        if not sched_file.exists():
            sched_file = nflverse_dir / "games.parquet"

        if not sched_file.exists():
            logger.warning(f"No NFLverse schedule file found. Run: Rscript scripts/fetch/fetch_nflverse_data.R")
            return {}

        if sched_file.suffix == ".parquet":
            schedules = pd.read_parquet(sched_file)
        else:
            schedules = pd.read_csv(sched_file, low_memory=False)

        # Filter for the requested season and week
        if "season" in schedules.columns:
            schedules = schedules[schedules["season"] == season]
        week_games = schedules[schedules['week'] == week].copy()

        if week_games.empty:
            logger.warning(f"No games found for Week {week}, Season {season}")
            return {}

        # Build game status map from NFLverse schedule data
        # NFLverse game_id format: 2025_01_KC_BAL
        # Map NFLverse result column to status:
        # - result is NaN/None = "pre_game"
        # - result exists = "complete"
        game_status_map = {}
        for _, game in week_games.iterrows():
            game_id = game.get('game_id')
            if not game_id:
                continue

            # Determine status from result column
            result = game.get('result')
            if pd.isna(result) or result is None or result == '':
                status = "pre_game"
            else:
                status = "complete"

            game_status_map[str(game_id)] = status

        logger.info(f"Loaded {len(game_status_map)} game statuses from NFLverse for Week {week}")
        return game_status_map

    except Exception as e:
        logger.error(f"Failed to load game status map from NFLverse: {e}")
        return {}


def is_valid_pregame_odds(
    row: pd.Series,
    game_status_map: Dict[str, str],
    current_time: datetime,
    min_minutes_before_kickoff: int = 5,
    max_hours_stale: int = 24
) -> Tuple[bool, str]:
    """Multi-layered validation for pre-game odds.

    Validates odds using multiple data sources:
    1. Game Status Check (NFLverse schedules) - most reliable
    2. Temporal Validation (commence_time) - universal fallback
    3. Staleness Check (last_update) - optional data quality check

    Args:
        row: DataFrame row with odds data
        game_status_map: Dict mapping game_id -> status from NFLverse
        current_time: Current UTC datetime
        min_minutes_before_kickoff: Min minutes before game starts
        max_hours_stale: Max age of odds in hours

    Returns:
        Tuple of (is_valid: bool, rejection_reason: str)
    """
    # Layer 1: Game Status Check (NFLverse schedules)
    game_id = str(row.get('game_id', ''))

    # Try to match game_id in multiple formats
    game_status = None
    if game_id:
        # Direct match (NFLverse format: 2025_01_KC_BAL)
        game_status = game_status_map.get(game_id)

        # Try fuzzy matching if direct match fails
        if not game_status and '_' in game_id:
            # Try to find matching game by teams or week
            for nfl_game_id, status in game_status_map.items():
                # Simple heuristic: if game IDs share common components
                if any(part in nfl_game_id for part in game_id.split('_') if len(part) > 2):
                    game_status = status
                    break

    if game_status in ['in_progress', 'complete']:
        return False, f"game_status={game_status}"

    # Layer 2: Temporal Validation
    commence_time_str = row.get('commence_time', '')
    if not commence_time_str:
        # No commence_time - can't validate temporally
        if game_status == 'pre_game':
            return True, "valid_pregame"
        else:
            return False, "missing_commence_time"

    try:
        # Parse ISO 8601 format
        if isinstance(commence_time_str, str):
            # Handle various datetime formats
            if 'T' in commence_time_str:
                # ISO format: 2025-09-07T17:00:00Z
                commence_dt = datetime.fromisoformat(
                    commence_time_str.replace('Z', '+00:00')
                )
            else:
                # Fallback format
                commence_dt = datetime.strptime(
                    commence_time_str, '%Y-%m-%d %H:%M:%S'
                )
                commence_dt = commence_dt.replace(tzinfo=timezone.utc)
        else:
            # Already datetime
            commence_dt = commence_time_str
            if commence_dt.tzinfo is None:
                commence_dt = commence_dt.replace(tzinfo=timezone.utc)

        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        minutes_to_kickoff = (commence_dt - current_time).total_seconds() / 60

        # Game already started
        if minutes_to_kickoff < 0:
            return False, (
                f"game_already_started "
                f"({abs(minutes_to_kickoff):.1f}min_ago)"
            )

        # Too close to kickoff (odds may be stale or in-game)
        if minutes_to_kickoff < min_minutes_before_kickoff:
            return False, f"too_close_to_kickoff ({minutes_to_kickoff:.1f}min)"

    except Exception as e:
        logger.debug(f"Failed to parse commence_time: {e}")
        # Can't validate temporally - rely on NFLverse game status
        if game_status == 'pre_game':
            return True, "valid_pregame"
        else:
            return False, "invalid_commence_time_format"

    # Layer 3: Staleness Check (if last_update available)
    last_update_str = row.get('last_update', '')
    if last_update_str:
        try:
            if isinstance(last_update_str, str):
                if 'T' in last_update_str:
                    update_dt = datetime.fromisoformat(
                        last_update_str.replace('Z', '+00:00')
                    )
                else:
                    update_dt = datetime.strptime(
                        last_update_str, '%Y-%m-%d %H:%M:%S'
                    )
                    update_dt = update_dt.replace(tzinfo=timezone.utc)
            else:
                update_dt = last_update_str
                if update_dt.tzinfo is None:
                    update_dt = update_dt.replace(tzinfo=timezone.utc)

            hours_since_update = (
                (current_time - update_dt).total_seconds() / 3600
            )

            if hours_since_update > max_hours_stale:
                return False, f"stale_odds ({hours_since_update:.1f}h_old)"

        except Exception as e:
            logger.debug(f"Failed to parse last_update: {e}")
            # Don't reject if we can't parse - just skip staleness check
            pass

    # All checks passed
    return True, "valid_pregame"


def filter_pregame_odds(
    odds_df: pd.DataFrame,
    week: int,
    season: int = 2025,
    min_minutes_before_kickoff: int = 5,
    max_hours_stale: int = 24,
    current_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Filter odds to only valid pre-game lines.

    This is the main filtering function that should be called before
    matching odds to predictions. It removes:
    - In-game odds (game status = "in_progress")
    - Post-game odds (game status = "complete")
    - Odds too close to kickoff (< 5 minutes by default)
    - Stale odds (> 24 hours old by default)

    Args:
        odds_df: DataFrame with odds data
        week: NFL week number
        season: NFL season year
        min_minutes_before_kickoff: Minimum minutes before kickoff
        max_hours_stale: Maximum hours old for odds
        current_time: Current time (defaults to now in UTC)

    Returns:
        Filtered DataFrame with only valid pre-game odds

    Example:
        >>> odds_df = pd.read_csv('data/nfl_player_props_draftkings.csv')
        >>> valid_odds = filter_pregame_odds(odds_df, week=9)
        >>> # Use valid_odds for predictions
    """
    if odds_df.empty:
        logger.warning("Empty odds DataFrame provided")
        return odds_df

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    # Load game status map from NFLverse
    game_status_map = load_game_status_map(week, season)

    if not game_status_map:
        logger.warning(
            f"No game status data available for Week {week} - "
            f"filtering will rely only on temporal validation"
        )

    # Apply validation to each row
    validation_results = odds_df.apply(
        lambda row: is_valid_pregame_odds(
            row,
            game_status_map,
            current_time,
            min_minutes_before_kickoff,
            max_hours_stale
        ),
        axis=1
    )

    # Split results into is_valid and rejection_reason
    odds_df = odds_df.copy()
    odds_df['is_valid_pregame'] = validation_results.apply(lambda x: x[0])
    odds_df['rejection_reason'] = validation_results.apply(lambda x: x[1])

    # Filter to valid only
    valid_odds = odds_df[odds_df['is_valid_pregame']].copy()

    # Log filtering statistics
    total = len(odds_df)
    valid = len(valid_odds)
    rejected = total - valid

    logger.info(
        f"Odds filtering: {valid}/{total} valid "
        f"({valid/total*100:.1f}%)"
    )

    if rejected > 0:
        # Log rejection reason breakdown
        rejection_counts = (
            odds_df[~odds_df['is_valid_pregame']]['rejection_reason']
            .value_counts()
        )
        logger.info(f"Rejection reasons:")
        for reason, count in rejection_counts.items():
            logger.info(f"  - {reason}: {count} ({count/rejected*100:.1f}%)")

        # Warning if high rejection rate
        if rejected / total > 0.3:
            logger.warning(
                f"High rejection rate ({rejected/total*100:.1f}%) - "
                f"check if odds data is stale or games have started"
            )

    # Drop temporary columns before returning
    valid_odds = valid_odds.drop(
        columns=['is_valid_pregame', 'rejection_reason'],
        errors='ignore'
    )

    return valid_odds
