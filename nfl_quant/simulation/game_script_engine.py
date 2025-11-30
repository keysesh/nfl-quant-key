"""
Dynamic Game Script Engine for Monte Carlo simulations.

Models how game flow evolves based on score differential:
- Trailing teams pass more, leading teams rush more
- Clock management in 4th quarter
- Two-minute drill logic
- Garbage time adjustments (prevent defense)

Research-backed PROE (Pass Rate Over Expected) adjustments based on:
- Score differential
- Time remaining
- Field position
- Down and distance state
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GameState(Enum):
    """Game state categories."""
    CLOSE = "close"  # Within 1 score (≤8 points)
    MODERATE_LEAD = "moderate_lead"  # 9-16 points
    BLOWOUT = "blowout"  # >16 points
    GARBAGE_TIME = "garbage_time"  # >20 points in 4th quarter


@dataclass
class GameScriptConfig:
    """Configuration for game script adjustments."""

    # Pass rate adjustments by score differential (percentage points)
    # Positive = more passing, negative = less passing
    trailing_by_3_to_7: float = 0.05  # +5% pass rate
    trailing_by_8_to_14: float = 0.10  # +10% pass rate
    trailing_by_15_plus: float = 0.15  # +15% pass rate (desperate)

    leading_by_3_to_7: float = -0.02  # -2% pass rate (slight run tilt)
    leading_by_8_to_14: float = -0.08  # -8% pass rate (protect lead)
    leading_by_15_plus: float = -0.12  # -12% pass rate (run clock)

    # Time-based adjustments (4th quarter)
    fourth_quarter_trailing_multiplier: float = 1.5  # Amplify trailing effect
    fourth_quarter_leading_multiplier: float = 1.3  # Amplify leading effect

    # Two-minute drill (last 2 minutes of half)
    two_minute_drill_pass_boost: float = 0.20  # +20% pass rate when trailing
    two_minute_drill_threshold: int = 120  # 120 seconds = 2 minutes

    # Garbage time thresholds
    garbage_time_threshold: int = 20  # 20+ point lead in 4th = garbage time
    garbage_time_pass_reduction: float = -0.25  # -25% pass rate (run clock)
    garbage_time_prevent_defense: float = 0.15  # +15% completion%, -10% yards/att

    # Pace adjustments (seconds per play)
    hurry_up_pace: float = 20.0  # Seconds per play when trailing late
    normal_pace: float = 30.0
    slow_pace: float = 40.0  # Seconds per play when leading late

    # Base pass rate (situation-neutral)
    base_pass_rate: float = 0.60  # 60% passes, 40% runs


class GameScriptEngine:
    """
    Simulates dynamic game script evolution during Monte Carlo trials.

    Tracks score differential over time and adjusts:
    - Pass/rush distribution
    - Pace (seconds per play)
    - Player usage patterns
    - Efficiency expectations (garbage time, prevent defense)
    """

    def __init__(self, config: Optional[GameScriptConfig] = None):
        """
        Initialize game script engine.

        Args:
            config: Game script configuration (uses defaults if None)
        """
        self.config = config or GameScriptConfig()

    def simulate_game_flow(
        self,
        home_team_strength: float,
        away_team_strength: float,
        n_quarters: int = 4,
        possessions_per_quarter: int = 6,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate play-by-play game flow with evolving game script.

        Args:
            home_team_strength: Home team expected points per drive
            away_team_strength: Away team expected points per drive
            n_quarters: Number of quarters (default 4)
            possessions_per_quarter: Possessions per quarter (default 6)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with possession-level simulation:
                - quarter: int
                - possession_number: int
                - offensive_team: str ('home' or 'away')
                - score_differential: int (from offensive team perspective)
                - time_remaining: int (seconds)
                - pass_rate: float
                - pace: float (seconds per play)
                - game_state: str
                - points_scored: int
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize game state
        home_score = 0
        away_score = 0
        game_log = []

        total_possessions = n_quarters * possessions_per_quarter
        seconds_per_quarter = 900  # 15 minutes

        # Alternate possessions
        for poss_idx in range(total_possessions):
            # Determine quarter and time
            quarter = (poss_idx // possessions_per_quarter) + 1
            quarter_possession = poss_idx % possessions_per_quarter
            time_remaining = seconds_per_quarter * (n_quarters - quarter + 1) - (quarter_possession * 120)
            time_remaining = max(0, time_remaining)

            # Alternate between home and away
            offensive_team = 'home' if poss_idx % 2 == 0 else 'away'
            team_strength = home_team_strength if offensive_team == 'home' else away_team_strength

            # Calculate score differential from offensive team perspective
            if offensive_team == 'home':
                score_diff = home_score - away_score
            else:
                score_diff = away_score - home_score

            # Determine game state
            game_state = self._classify_game_state(
                score_differential=score_diff,
                quarter=quarter,
                time_remaining=time_remaining
            )

            # Calculate pass rate and pace based on game script
            pass_rate = self.calculate_pass_rate(
                score_differential=score_diff,
                quarter=quarter,
                time_remaining=time_remaining,
                base_pass_rate=self.config.base_pass_rate
            )

            pace = self.calculate_pace(
                score_differential=score_diff,
                quarter=quarter,
                time_remaining=time_remaining
            )

            # Simulate points scored on this drive
            points_scored = self._simulate_drive_outcome(
                team_strength=team_strength,
                pass_rate=pass_rate,
                game_state=game_state
            )

            # Update scores
            if offensive_team == 'home':
                home_score += points_scored
            else:
                away_score += points_scored

            # Log possession
            game_log.append({
                'quarter': quarter,
                'possession_number': poss_idx,
                'offensive_team': offensive_team,
                'score_differential': score_diff,
                'time_remaining': time_remaining,
                'pass_rate': pass_rate,
                'pace': pace,
                'game_state': game_state.value,
                'points_scored': points_scored,
                'home_score': home_score,
                'away_score': away_score
            })

        return pd.DataFrame(game_log)

    def calculate_pass_rate(
        self,
        score_differential: int,
        quarter: int,
        time_remaining: int,
        base_pass_rate: float = 0.60
    ) -> float:
        """
        Calculate pass rate based on game script.

        Args:
            score_differential: Score differential (positive = winning, negative = losing)
            quarter: Current quarter (1-4)
            time_remaining: Time remaining in game (seconds)
            base_pass_rate: Situation-neutral pass rate

        Returns:
            Adjusted pass rate (0.0 to 1.0)
        """
        adjustment = 0.0

        # Score-based adjustments
        if score_differential <= -15:
            adjustment = self.config.trailing_by_15_plus
        elif score_differential <= -8:
            adjustment = self.config.trailing_by_8_to_14
        elif score_differential <= -3:
            adjustment = self.config.trailing_by_3_to_7
        elif score_differential >= 15:
            adjustment = self.config.leading_by_15_plus
        elif score_differential >= 8:
            adjustment = self.config.leading_by_8_to_14
        elif score_differential >= 3:
            adjustment = self.config.leading_by_3_to_7

        # Amplify in 4th quarter
        if quarter == 4:
            if score_differential < 0:
                adjustment *= self.config.fourth_quarter_trailing_multiplier
            elif score_differential > 0:
                adjustment *= self.config.fourth_quarter_leading_multiplier

        # Two-minute drill logic
        if time_remaining <= self.config.two_minute_drill_threshold and score_differential < 0:
            adjustment += self.config.two_minute_drill_pass_boost

        # Garbage time
        if quarter == 4 and abs(score_differential) >= self.config.garbage_time_threshold:
            if score_differential > 0:  # Leading team
                adjustment = self.config.garbage_time_pass_reduction
            # Trailing team in garbage time still passes a lot (stay as-is)

        # Apply adjustment
        pass_rate = base_pass_rate + adjustment

        # Clip to valid range
        return np.clip(pass_rate, 0.30, 0.85)

    def calculate_pace(
        self,
        score_differential: int,
        quarter: int,
        time_remaining: int
    ) -> float:
        """
        Calculate pace (seconds per play) based on game script.

        Args:
            score_differential: Score differential (positive = winning, negative = losing)
            quarter: Current quarter (1-4)
            time_remaining: Time remaining in game (seconds)

        Returns:
            Seconds per play
        """
        # Default pace
        pace = self.config.normal_pace

        # 4th quarter adjustments
        if quarter == 4:
            if score_differential < -8:
                # Trailing: hurry up
                pace = self.config.hurry_up_pace
            elif score_differential > 8 and time_remaining < 600:  # <10 minutes
                # Leading: run clock
                pace = self.config.slow_pace

        # Two-minute drill
        if time_remaining <= self.config.two_minute_drill_threshold and score_differential < 0:
            pace = self.config.hurry_up_pace

        return pace

    def _classify_game_state(
        self,
        score_differential: int,
        quarter: int,
        time_remaining: int
    ) -> GameState:
        """
        Classify current game state.

        Args:
            score_differential: Score differential
            quarter: Current quarter
            time_remaining: Time remaining (seconds)

        Returns:
            GameState enum value
        """
        abs_diff = abs(score_differential)

        # Garbage time check (4th quarter only)
        if quarter == 4 and abs_diff >= self.config.garbage_time_threshold:
            return GameState.GARBAGE_TIME

        # Blowout
        if abs_diff > 16:
            return GameState.BLOWOUT

        # Moderate lead
        if abs_diff > 8:
            return GameState.MODERATE_LEAD

        # Close game
        return GameState.CLOSE

    def _simulate_drive_outcome(
        self,
        team_strength: float,
        pass_rate: float,
        game_state: GameState
    ) -> int:
        """
        Simulate points scored on a single drive.

        Args:
            team_strength: Expected points per drive
            pass_rate: Current pass rate (affects efficiency)
            game_state: Current game state

        Returns:
            Points scored (0, 3, 6, 7, 8)
        """
        # Adjust team strength based on game state
        adjusted_strength = team_strength

        # Garbage time: prevent defense gives up yards but not TDs
        if game_state == GameState.GARBAGE_TIME:
            adjusted_strength *= 0.85  # Reduce TD probability

        # Very pass-heavy or run-heavy affects efficiency
        if pass_rate > 0.75:
            # Overly predictable passing
            adjusted_strength *= 0.95
        elif pass_rate < 0.40:
            # Overly predictable running
            adjusted_strength *= 0.92

        # Sample drive outcome
        # Use exponential distribution for points (most drives score 0)
        base_rate = 1.0 / max(adjusted_strength, 0.5)  # Prevent division by zero
        raw_points = np.random.exponential(scale=1.0 / base_rate)

        # Discretize to NFL scoring outcomes
        if raw_points < 0.5:
            return 0  # No score (punt, turnover)
        elif raw_points < 2.0:
            return 3  # Field goal
        elif raw_points < 3.5:
            return 7  # Touchdown + XP
        else:
            # Small chance of missed XP or 2PT conversion
            return np.random.choice([6, 7, 8], p=[0.02, 0.96, 0.02])

    def get_player_usage_adjustments(
        self,
        position: str,
        score_differential: int,
        quarter: int,
        time_remaining: int
    ) -> Dict[str, float]:
        """
        Get position-specific usage adjustments based on game script.

        Args:
            position: Player position (QB, RB, WR, TE)
            score_differential: Score differential
            quarter: Current quarter
            time_remaining: Time remaining (seconds)

        Returns:
            Dictionary of usage multipliers:
                - targets_multiplier: float
                - carries_multiplier: float
                - snaps_multiplier: float
        """
        adjustments = {
            'targets_multiplier': 1.0,
            'carries_multiplier': 1.0,
            'snaps_multiplier': 1.0
        }

        # Calculate pass rate delta from base
        current_pass_rate = self.calculate_pass_rate(
            score_differential=score_differential,
            quarter=quarter,
            time_remaining=time_remaining
        )
        pass_rate_delta = current_pass_rate - self.config.base_pass_rate

        # Position-specific adjustments
        if position == 'QB':
            # QB attempts scale with pass rate
            adjustments['targets_multiplier'] = 1.0 + (pass_rate_delta / self.config.base_pass_rate)

        elif position == 'RB':
            # RB carries inverse of pass rate
            adjustments['carries_multiplier'] = 1.0 - (pass_rate_delta / (1.0 - self.config.base_pass_rate))

            # RB targets slightly increase when passing more
            adjustments['targets_multiplier'] = 1.0 + (pass_rate_delta * 0.5)

        elif position in ['WR', 'TE']:
            # Receiver targets scale with pass rate
            adjustments['targets_multiplier'] = 1.0 + (pass_rate_delta / self.config.base_pass_rate)

            # WR1 gets more targets when trailing (target concentration)
            if score_differential < -7:
                adjustments['wr1_boost'] = 1.15  # +15% to primary WR

        # Garbage time: backup players get more snaps
        if quarter == 4 and abs(score_differential) >= self.config.garbage_time_threshold:
            adjustments['backup_snaps_multiplier'] = 1.50  # Backups play more

        return adjustments


# Example usage and testing
if __name__ == '__main__':
    # Initialize game script engine
    engine = GameScriptEngine()

    # Simulate a close game
    print("=== Close Game Simulation ===")
    game_flow = engine.simulate_game_flow(
        home_team_strength=2.0,  # 2 points per drive
        away_team_strength=2.0,
        n_quarters=4,
        possessions_per_quarter=6,
        seed=42
    )

    print(game_flow[['quarter', 'offensive_team', 'score_differential', 'pass_rate', 'game_state', 'points_scored']].head(20))
    print(f"\nFinal Score - Home: {game_flow.iloc[-1]['home_score']}, Away: {game_flow.iloc[-1]['away_score']}")

    # Test pass rate calculation
    print("\n=== Pass Rate Examples ===")
    scenarios = [
        (0, 1, 3600, "Neutral, Q1"),
        (-10, 4, 300, "Down 10, Q4, 5 min left"),
        (14, 4, 600, "Up 14, Q4, 10 min left"),
        (-3, 4, 100, "Down 3, Q4, 2-min drill"),
        (21, 4, 180, "Up 21, Q4, garbage time")
    ]

    for score_diff, quarter, time_rem, desc in scenarios:
        pass_rate = engine.calculate_pass_rate(score_diff, quarter, time_rem)
        pace = engine.calculate_pace(score_diff, quarter, time_rem)
        print(f"{desc}: Pass Rate = {pass_rate:.2%}, Pace = {pace:.1f} sec/play")

    # Test player usage adjustments
    print("\n=== Player Usage Adjustments ===")
    adjustments_down_10 = engine.get_player_usage_adjustments('RB', -10, 4, 300)
    print(f"RB down 10 in Q4: {adjustments_down_10}")

    adjustments_up_14 = engine.get_player_usage_adjustments('RB', 14, 4, 600)
    print(f"RB up 14 in Q4: {adjustments_up_14}")
