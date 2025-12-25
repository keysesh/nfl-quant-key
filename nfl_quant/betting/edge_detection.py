"""
Edge Detection and CLV (Closing Line Value) Tracking

Implements:
1. Proper no-vig probability calculation
2. Edge calculation with market-specific thresholds
3. CLV tracking (key metric for long-term success)
4. Kelly criterion bet sizing (delegates to kelly_criterion.py)
5. Risk management controls
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from nfl_quant.betting.kelly_criterion import (
    calculate_fractional_kelly as _canonical_kelly,
    american_to_decimal,
)
from nfl_quant.core.unified_betting import (
    american_odds_to_implied_prob as _canonical_american_to_implied,
)

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """Container for a single bet recommendation."""
    player: str
    prop_type: str
    line: float
    direction: str  # 'over' or 'under'

    # Probabilities
    model_prob: float
    market_prob: float
    no_vig_prob: float

    # Edge metrics
    edge: float  # model_prob - no_vig_prob
    edge_vs_market: float  # model_prob - market_prob
    expected_value: float  # (model_prob * payout) - 1

    # Odds
    odds: int  # American odds

    # Sizing
    kelly_fraction: float
    recommended_units: float

    # Confidence
    confidence_tier: str  # 'high', 'medium', 'low'
    calibrator_coverage: float

    # CLV tracking
    opening_prob: Optional[float] = None
    closing_prob: Optional[float] = None
    clv: Optional[float] = None


class EdgeDetector:
    """
    Detect betting edges with proper vig removal and market-specific thresholds.
    """

    def __init__(self):
        # Market-specific minimum edge thresholds
        self.min_edge_thresholds = {
            'player_reception_yds': 0.03,  # 3% - most liquid
            'player_receptions': 0.03,
            'player_rush_yds': 0.04,  # 4%
            'player_pass_yds': 0.05,  # 5%
            'player_pass_tds': 0.08,  # 8% - binary events
            'player_anytime_td': 0.08,
            'player_rush_att': 0.04,
            'player_pass_completions': 0.05,
        }

        # Confidence tier thresholds
        self.confidence_tiers = {
            'high': 0.07,  # 7%+ edge
            'medium': 0.04,  # 4-7% edge
            'low': 0.02,  # 2-4% edge
        }

        # Typical vig on DraftKings props
        self.assumed_vig = 0.045  # 4.5% per side

    def american_to_implied(self, odds: int) -> float:
        """
        Convert American odds to implied probability.

        Delegates to canonical implementation in nfl_quant.core.unified_betting.

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Implied probability (0-1)
        """
        return _canonical_american_to_implied(odds)

    def american_to_decimal(self, odds: int) -> float:
        """Convert American odds to decimal odds.

        Delegates to canonical implementation in nfl_quant.betting.kelly_criterion.
        """
        return american_to_decimal(odds)

    def calculate_no_vig_probability(
        self,
        over_odds: int,
        under_odds: int
    ) -> Tuple[float, float]:
        """
        Remove vig to get fair probabilities.

        Args:
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            (fair_over_prob, fair_under_prob) with vig removed
        """
        # Get implied probabilities
        over_implied = self.american_to_implied(over_odds)
        under_implied = self.american_to_implied(under_odds)

        # Total is typically > 1 due to vig
        total_implied = over_implied + under_implied

        # Normalize to remove vig
        fair_over = over_implied / total_implied
        fair_under = under_implied / total_implied

        return fair_over, fair_under

    def calculate_edge(
        self,
        model_prob: float,
        over_odds: int,
        under_odds: int,
        direction: str = 'over'
    ) -> Dict:
        """
        Calculate edge for a bet.

        Args:
            model_prob: Our model's probability for the direction
            over_odds: American odds for over
            under_odds: American odds for under
            direction: 'over' or 'under'

        Returns:
            Dictionary with edge metrics
        """
        # Get fair probabilities
        fair_over, fair_under = self.calculate_no_vig_probability(over_odds, under_odds)

        if direction == 'over':
            fair_prob = fair_over
            market_implied = self.american_to_implied(over_odds)
            odds = over_odds
        else:  # under
            fair_prob = fair_under
            market_implied = self.american_to_implied(under_odds)
            odds = under_odds

        # Edge calculations
        edge = model_prob - fair_prob
        edge_vs_market = model_prob - market_implied

        # Expected value
        decimal_odds = self.american_to_decimal(odds)
        ev = (model_prob * decimal_odds) - 1

        return {
            'model_prob': model_prob,
            'fair_prob': fair_prob,
            'market_implied': market_implied,
            'edge': edge,
            'edge_vs_market': edge_vs_market,
            'expected_value': ev,
            'odds': odds,
            'decimal_odds': decimal_odds,
        }

    def kelly_fraction(
        self,
        win_prob: float,
        odds: int,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly criterion bet fraction.

        Delegates to canonical implementation in nfl_quant.betting.kelly_criterion.

        Args:
            win_prob: Probability of winning
            odds: American odds
            fraction: Kelly fraction (0.25 = quarter Kelly, recommended)

        Returns:
            Fraction of bankroll to bet
        """
        # Delegate to canonical implementation
        kelly = _canonical_kelly(win_prob, odds, fraction)

        # Apply edge_detection specific cap at 5%
        return min(kelly, 0.05)

    def get_confidence_tier(self, edge: float) -> str:
        """Determine confidence tier based on edge."""
        if edge >= self.confidence_tiers['high']:
            return 'high'
        elif edge >= self.confidence_tiers['medium']:
            return 'medium'
        elif edge >= self.confidence_tiers['low']:
            return 'low'
        else:
            return 'no_bet'

    def evaluate_bet(
        self,
        player: str,
        prop_type: str,
        line: float,
        model_over_prob: float,
        over_odds: int,
        under_odds: int,
        calibrator_coverage: float = 0.5
    ) -> Optional[BetRecommendation]:
        """
        Evaluate a potential bet and return recommendation.

        Args:
            player: Player name
            prop_type: Type of prop (e.g., 'player_reception_yds')
            line: Betting line
            model_over_prob: Model's probability of over hitting
            over_odds: American odds for over
            under_odds: American odds for under
            calibrator_coverage: Confidence in calibrator (0-1)

        Returns:
            BetRecommendation if edge found, None otherwise
        """
        # Evaluate both directions
        over_edge_info = self.calculate_edge(model_over_prob, over_odds, under_odds, 'over')
        under_edge_info = self.calculate_edge(1 - model_over_prob, over_odds, under_odds, 'under')

        # Choose direction with higher edge
        if over_edge_info['edge'] > under_edge_info['edge']:
            edge_info = over_edge_info
            direction = 'over'
            model_prob = model_over_prob
        else:
            edge_info = under_edge_info
            direction = 'under'
            model_prob = 1 - model_over_prob

        # Check minimum edge threshold
        min_edge = self.min_edge_thresholds.get(prop_type, 0.05)

        if edge_info['edge'] < min_edge:
            logger.debug(f"  {player} {prop_type}: Edge {edge_info['edge']:.2%} < min {min_edge:.2%}")
            return None

        # Get confidence tier
        confidence_tier = self.get_confidence_tier(edge_info['edge'])

        if confidence_tier == 'no_bet':
            return None

        # Calculate Kelly sizing
        kelly = self.kelly_fraction(model_prob, edge_info['odds'], fraction=0.25)

        # Adjust for calibrator confidence
        adjusted_kelly = kelly * calibrator_coverage

        # Convert to units (100 units = full bankroll)
        recommended_units = adjusted_kelly * 100

        # Apply minimum bet threshold
        if recommended_units < 0.5:
            return None

        return BetRecommendation(
            player=player,
            prop_type=prop_type,
            line=line,
            direction=direction,
            model_prob=model_prob,
            market_prob=edge_info['market_implied'],
            no_vig_prob=edge_info['fair_prob'],
            edge=edge_info['edge'],
            edge_vs_market=edge_info['edge_vs_market'],
            expected_value=edge_info['expected_value'],
            odds=edge_info['odds'],
            kelly_fraction=adjusted_kelly,
            recommended_units=recommended_units,
            confidence_tier=confidence_tier,
            calibrator_coverage=calibrator_coverage,
        )


class CLVTracker:
    """
    Track Closing Line Value (CLV) - the primary indicator of long-term success.

    CLV = Our bet probability - Closing line probability

    Positive CLV means we consistently beat closing lines, indicating real edge.
    """

    def __init__(self):
        self.bets = []
        self.clv_history = []

    def add_bet(
        self,
        bet: BetRecommendation,
        opening_odds: int = None,
        closing_odds: int = None
    ):
        """
        Add a bet with optional opening/closing line tracking.

        Args:
            bet: BetRecommendation object
            opening_odds: Odds when we placed the bet
            closing_odds: Odds at game time
        """
        bet_record = {
            'player': bet.player,
            'prop_type': bet.prop_type,
            'line': bet.line,
            'direction': bet.direction,
            'model_prob': bet.model_prob,
            'market_prob_at_bet': bet.market_prob,
            'edge_at_bet': bet.edge,
            'odds_at_bet': bet.odds,
            'opening_odds': opening_odds,
            'closing_odds': closing_odds,
            'clv': None,
        }

        # Calculate CLV if closing odds available
        if closing_odds is not None:
            closing_implied = EdgeDetector().american_to_implied(closing_odds)
            clv = bet.model_prob - closing_implied
            bet_record['clv'] = clv
            self.clv_history.append(clv)

        self.bets.append(bet_record)

    def get_clv_summary(self) -> Dict:
        """Get summary statistics for CLV."""
        if not self.clv_history:
            return {
                'total_bets': len(self.bets),
                'clv_tracked': 0,
                'avg_clv': 0,
                'positive_clv_pct': 0,
            }

        clv_values = np.array(self.clv_history)

        return {
            'total_bets': len(self.bets),
            'clv_tracked': len(self.clv_history),
            'avg_clv': float(np.mean(clv_values)),
            'median_clv': float(np.median(clv_values)),
            'std_clv': float(np.std(clv_values)),
            'positive_clv_pct': float((clv_values > 0).mean()),
            'clv_percentiles': {
                '25th': float(np.percentile(clv_values, 25)),
                '50th': float(np.percentile(clv_values, 50)),
                '75th': float(np.percentile(clv_values, 75)),
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bet history to DataFrame."""
        return pd.DataFrame(self.bets)


class RiskManager:
    """
    Risk management controls for betting.
    """

    def __init__(
        self,
        bankroll: float = 10000,
        max_bet_fraction: float = 0.03,  # 3% max per bet
        max_daily_loss: float = 0.05,  # 5% daily loss limit
        max_exposure_per_player: float = 0.10,  # 10% max per player
        max_exposure_per_game: float = 0.15,  # 15% max per game
    ):
        self.bankroll = bankroll
        self.max_bet_fraction = max_bet_fraction
        self.max_daily_loss = max_daily_loss
        self.max_exposure_per_player = max_exposure_per_player
        self.max_exposure_per_game = max_exposure_per_game

        self.daily_pnl = 0
        self.current_exposure = {}  # {player: amount, game: amount}

    def check_bet_allowed(
        self,
        bet: BetRecommendation,
        bet_amount: float = None
    ) -> Tuple[bool, str]:
        """
        Check if a bet passes risk management rules.

        Args:
            bet: BetRecommendation object
            bet_amount: Proposed bet amount (default: calculate from Kelly)

        Returns:
            (allowed, reason)
        """
        if bet_amount is None:
            bet_amount = bet.kelly_fraction * self.bankroll

        # Check max bet size
        if bet_amount > self.bankroll * self.max_bet_fraction:
            return False, f"Bet exceeds max fraction ({self.max_bet_fraction:.1%})"

        # Check daily loss limit
        if self.daily_pnl <= -self.bankroll * self.max_daily_loss:
            return False, f"Daily loss limit reached ({self.max_daily_loss:.1%})"

        # Check player exposure
        player_exposure = self.current_exposure.get(bet.player, 0) + bet_amount
        if player_exposure > self.bankroll * self.max_exposure_per_player:
            return False, f"Max player exposure reached ({self.max_exposure_per_player:.1%})"

        return True, "OK"

    def record_bet(self, bet: BetRecommendation, amount: float):
        """Record a bet for exposure tracking."""
        self.current_exposure[bet.player] = self.current_exposure.get(bet.player, 0) + amount

    def record_result(self, profit_loss: float):
        """Record P&L for daily tracking."""
        self.daily_pnl += profit_loss

    def reset_daily(self):
        """Reset daily limits."""
        self.daily_pnl = 0

    def get_status(self) -> Dict:
        """Get current risk status."""
        return {
            'bankroll': self.bankroll,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.bankroll,
            'current_exposure': self.current_exposure,
            'total_exposure': sum(self.current_exposure.values()),
        }


def generate_bet_recommendations(
    props_df: pd.DataFrame,
    model_probs: pd.Series,
    calibrator_coverage: float = 0.7
) -> List[BetRecommendation]:
    """
    Generate bet recommendations from model predictions.

    Args:
        props_df: DataFrame with prop info (player, line, odds, etc.)
        model_probs: Series of model probabilities for over
        calibrator_coverage: Confidence in model calibration

    Returns:
        List of BetRecommendation objects
    """
    detector = EdgeDetector()
    recommendations = []

    for idx, row in props_df.iterrows():
        model_prob = model_probs.loc[idx]

        rec = detector.evaluate_bet(
            player=row['player'],
            prop_type=row['market'],
            line=row['line'],
            model_over_prob=model_prob,
            over_odds=int(row['over_odds']),
            under_odds=int(row['under_odds']),
            calibrator_coverage=calibrator_coverage
        )

        if rec is not None:
            recommendations.append(rec)

    # Sort by edge (highest first)
    recommendations.sort(key=lambda x: x.edge, reverse=True)

    return recommendations
