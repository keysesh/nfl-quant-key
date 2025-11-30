"""Parlay recommendation engine - generate parlay combinations from single bets."""

from typing import List, Dict, Optional, Tuple
import itertools
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from .odds_calculator import ParlayOddsCalculator
    from .correlation import CorrelationChecker, ParlayLeg
    from .push_handler import PushHandler
except ImportError:
    from odds_calculator import ParlayOddsCalculator
    from correlation import CorrelationChecker, ParlayLeg
    from push_handler import PushHandler


def load_betting_config():
    """Load betting thresholds configuration from JSON file."""
    config_path = Path("configs/betting_thresholds.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback defaults if config doesn't exist
        return {
            "parlay_optimization": {
                "correlation_threshold": 0.70,
                "max_legs": 4,
                "min_confidence_per_leg": 0.52,
                "min_edge_per_leg": 0.02
            },
            "bet_sizing": {
                "default_bankroll": 50.0
            }
        }


# Load config at module level
BETTING_CONFIG = load_betting_config()


def format_market_name(market: Optional[str]) -> str:
    """Convert market code to readable stat name."""
    if not market:
        return ""

    market_map = {
        'player_pass_yds': 'Pass Yds',
        'player_rush_yds': 'Rush Yds',
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Rec Yds',
        'player_pass_tds': 'Pass TDs',
        'player_rush_tds': 'Rush TDs',
        'player_receiving_tds': 'Rec TDs',
        'player_anytime_td': 'Anytime TD',
        'game_total': 'Total',
        'game_moneyline': 'ML',
        'game_spread': 'Spread',
    }
    return market_map.get(market, market.replace('_', ' ').title())


def format_leg_name(leg: ParlayLeg) -> str:
    """Format parlay leg name with market context."""
    market_name = ""
    if leg.market:
        market_name = format_market_name(leg.market)
        if market_name:
            market_name = f"{market_name} "

    return f"{market_name}{leg.name}"


@dataclass
class SingleBet:
    """Represents a single bet recommendation."""
    name: str
    bet_type: str
    game: str
    team: Optional[str] = None
    player: Optional[str] = None
    market: Optional[str] = None
    odds: Optional[int] = None
    our_prob: Optional[float] = None
    market_prob: Optional[float] = None
    edge: Optional[float] = None
    bet_size: Optional[float] = None
    potential_profit: Optional[float] = None


@dataclass
class ParlayRecommendation:
    """Represents a parlay recommendation."""
    legs: List[SingleBet]
    true_odds: int
    model_odds: int
    true_prob: float
    model_prob: float
    edge: float
    correlation_valid: bool
    correlation_issues: List[str]
    recommended_stake: float
    potential_win: float
    expected_value: Optional[float] = None


class ParlayRecommender:
    """Generate parlay recommendations from single bets."""

    def __init__(
        self,
        correlation_threshold: float = None,
        max_legs: int = None,
        min_confidence: float = None,
        min_edge: float = None
    ):
        """Initialize parlay recommender.

        Args:
            correlation_threshold: Maximum allowed correlation (loads from config if None)
            max_legs: Maximum legs per parlay (loads from config if None)
            min_confidence: Minimum confidence for bet inclusion (loads from config if None)
            min_edge: Minimum edge required (loads from config if None)
        """
        parlay_config = BETTING_CONFIG.get("parlay_optimization", {})

        self.correlation_threshold = correlation_threshold if correlation_threshold is not None else parlay_config.get("correlation_threshold", 0.70)
        self.max_legs = max_legs if max_legs is not None else parlay_config.get("max_legs", 4)
        self.min_confidence = min_confidence if min_confidence is not None else parlay_config.get("min_confidence_per_leg", 0.52)
        self.min_edge = min_edge if min_edge is not None else parlay_config.get("min_edge_per_leg", 0.02)

        self.calculator = ParlayOddsCalculator()
        self.correlation_checker = CorrelationChecker()

    def generate_parlays(
        self,
        single_bets: List[SingleBet],
        num_parlays: int = 10
    ) -> List[ParlayRecommendation]:
        """Generate top parlay recommendations from single bets.

        Args:
            single_bets: List of single bet recommendations
            num_parlays: Number of parlays to return

        Returns:
            List of parlay recommendations
        """
        # Filter to high-confidence bets
        qualified_bets = [
            bet for bet in single_bets
            if bet.our_prob and bet.our_prob >= self.min_confidence
        ]

        if len(qualified_bets) < 2:
            return []

        # Generate combinations
        all_parlays = []

        for num_legs in range(2, min(self.max_legs + 1, len(qualified_bets) + 1)):
            for combo in itertools.combinations(qualified_bets, num_legs):
                parlay = self._create_parlay(list(combo))
                if parlay:
                    all_parlays.append(parlay)

        # Sort by edge
        all_parlays.sort(key=lambda p: p.edge, reverse=True)

        # Return top N
        return all_parlays[:num_parlays]

    def _calculate_correlation_coefficient(
        self,
        bet1: SingleBet,
        bet2: SingleBet
    ) -> float:
        """
        Calculate correlation coefficient between two bets.

        Uses heuristics based on:
        - Same player (strong positive)
        - Same team offense (moderate positive)
        - Same game (weak positive)
        - Competing volume (negative)

        Args:
            bet1: First bet
            bet2: Second bet

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Same player, different markets
        if bet1.player and bet2.player and bet1.player == bet2.player:
            if bet1.market and bet2.market:
                pair = (bet1.market, bet2.market)
                if pair in self.correlation_checker.SAME_PLAYER_CORRELATIONS:
                    return self.correlation_checker.SAME_PLAYER_CORRELATIONS[pair]
                # Reverse order
                rev_pair = (bet2.market, bet1.market)
                if rev_pair in self.correlation_checker.SAME_PLAYER_CORRELATIONS:
                    return self.correlation_checker.SAME_PLAYER_CORRELATIONS[rev_pair]
            return 0.60  # Default same-player correlation

        # Same team (moderate positive for offensive stats)
        if bet1.team and bet2.team and bet1.team == bet2.team:
            # Both offensive props (QB/WR/RB)
            if bet1.bet_type == 'Player Prop' and bet2.bet_type == 'Player Prop':
                return 0.30  # Moderate positive (same offense)

        # Same game (weak positive due to game script)
        if bet1.game and bet2.game and bet1.game == bet2.game:
            return 0.15  # Weak positive

        # No correlation
        return 0.0

    def _create_parlay(
        self,
        bets: List[SingleBet]
    ) -> Optional[ParlayRecommendation]:
        """Create a parlay from a list of bets.

        Args:
            bets: List of bets to combine

        Returns:
            ParlayRecommendation or None if invalid
        """
        if len(bets) < 2:
            return None

        # NEW: Check for multiple bets from same game (correlated!)
        games = [bet.game for bet in bets]
        if len(set(games)) < len(bets):
            # Multiple bets from same game
            return None

        # Check correlation
        legs = [
            ParlayLeg(
                name=bet.name,
                bet_type=bet.bet_type,
                game=bet.game,
                team=bet.team,
                player=bet.player,
                market=bet.market,
                odds=bet.odds
            )
            for bet in bets
        ]

        validation = self.correlation_checker.validate_parlay(legs)

        # Skip if blocked
        if validation['blocked']:
            return None

        # Calculate odds
        leg_odds = [bet.odds for bet in bets if bet.odds]
        leg_names = [bet.name for bet in bets]
        leg_probs = [bet.our_prob for bet in bets if bet.our_prob]

        if len(leg_odds) != len(bets) or len(leg_probs) != len(bets):
            return None

        breakdown = self.calculator.calculate_parlay_odds(
            leg_odds=leg_odds,
            leg_names=leg_names,
            leg_probs=leg_probs
        )

        # Calculate stake using config bankroll
        bet_sizing_config = BETTING_CONFIG.get("bet_sizing", {})
        bankroll = bet_sizing_config.get("default_bankroll", 50.0)
        stake_info = self.calculator.calculate_stake(
            breakdown=breakdown,
            bankroll=bankroll,
            staking_method='fixed_pct',
            risk_level='moderate',
            num_legs=len(bets)
        )

        # Calculate EV
        expected_value = None
        if breakdown['model_parlay_prob']:
            decimal_odds = self.calculator.american_to_decimal(breakdown['true_parlay_odds'])
            expected_payout = stake_info['recommended_stake'] * decimal_odds * breakdown['model_parlay_prob']
            expected_value = expected_payout - stake_info['recommended_stake']

        return ParlayRecommendation(
            legs=bets,
            true_odds=breakdown['true_parlay_odds'],
            model_odds=breakdown['model_parlay_odds'] or breakdown['true_parlay_odds'],
            true_prob=breakdown['true_parlay_prob'],
            model_prob=breakdown['model_parlay_prob'] or breakdown['true_parlay_prob'],
            edge=breakdown['edge'] or 0.0,
            correlation_valid=validation['valid'],
            correlation_issues=validation['correlations'],
            recommended_stake=stake_info['recommended_stake'],
            potential_win=stake_info['potential_win'],
            expected_value=expected_value
        )

    def format_parlay_recommendation(self, parlay: ParlayRecommendation) -> str:
        """Format parlay recommendation for display.

        Args:
            parlay: Parlay recommendation

        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            f"PARLAY RECOMMENDATION ({len(parlay.legs)} legs)",
            "=" * 60,
        ]

        # List legs
        for i, leg in enumerate(parlay.legs, 1):
            leg_display = format_leg_name(leg)
            lines.append(f"Leg {i}: {leg_display}")
            if leg.game:
                lines.append(f"   Game: {leg.game}")
            if leg.our_prob:
                lines.append(f"   Confidence: {leg.our_prob:.1%}")

        lines.append("")

        # Odds and edge
        lines.append(f"True Odds: {parlay.true_odds:+d} ({parlay.true_prob:.1%} chance)")
        lines.append(f"Model Odds: {parlay.model_odds:+d} ({parlay.model_prob:.1%} chance)")
        lines.append(f"Edge: {parlay.edge:.1%}")

        # Correlation
        if parlay.correlation_valid:
            lines.append("✅ No correlation issues")
        else:
            lines.append("⚠️  Correlation warnings:")
            for issue in parlay.correlation_issues:
                lines.append(f"   - {issue}")

        # Betting info
        lines.append("")
        lines.append(f"Recommended Stake: ${parlay.recommended_stake:.2f}")
        lines.append(f"Potential Win: ${parlay.potential_win:.2f}")

        if parlay.expected_value:
            ev_emoji = "✅" if parlay.expected_value > 0 else "❌"
            lines.append(f"{ev_emoji} Expected Value: ${parlay.expected_value:.2f}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict_list(self, parlays: List[ParlayRecommendation]) -> List[Dict]:
        """Convert parlay recommendations to dictionary list.

        Args:
            parlays: List of parlay recommendations

        Returns:
            List of dictionaries
        """
        result = []

        for i, parlay in enumerate(parlays, 1):
            legs_str = " | ".join([leg.name for leg in parlay.legs])

            result.append({
                'rank': i,
                'legs': legs_str,
                'num_legs': len(parlay.legs),
                'true_odds': parlay.true_odds,
                'model_odds': parlay.model_odds,
                'true_prob': f"{parlay.true_prob:.1%}",
                'model_prob': f"{parlay.model_prob:.1%}",
                'edge': f"{parlay.edge:.1%}",
                'correlation_valid': '✅' if parlay.correlation_valid else '⚠️',
                'recommended_stake': f"${parlay.recommended_stake:.2f}",
                'potential_win': f"${parlay.potential_win:.2f}",
                'expected_value': f"${parlay.expected_value:.2f}" if parlay.expected_value else "N/A",
            })

        return result

    def filter_by_game_script(
        self,
        single_bets: List[SingleBet],
        max_per_game: int = 2
    ) -> List[SingleBet]:
        """Filter bets to avoid too many from same game.

        Args:
            single_bets: List of single bets
            max_per_game: Maximum bets per game

        Returns:
            Filtered list
        """
        from collections import Counter

        game_counts = Counter(bet.game for bet in single_bets if bet.game)

        filtered = []
        game_added = Counter()

        # Sort by confidence (highest first)
        sorted_bets = sorted(
            single_bets,
            key=lambda b: b.our_prob or 0,
            reverse=True
        )

        for bet in sorted_bets:
            if not bet.game:
                filtered.append(bet)
            elif game_added[bet.game] < max_per_game:
                filtered.append(bet)
                game_added[bet.game] += 1

        return filtered

    def suggest_game_script(self, bets: List[SingleBet]) -> str:
        """Suggest a game script based on bet types.

        Args:
            bets: List of bets

        Returns:
            Suggested game script
        """
        # Count bet types
        num_overs = sum(1 for bet in bets if 'over' in bet.bet_type.lower() or 'Over' in bet.name)
        num_unders = sum(1 for bet in bets if 'under' in bet.bet_type.lower() or 'Under' in bet.name)
        num_favorites = sum(1 for bet in bets if bet.team and 'spread' in bet.bet_type.lower())

        # Determine script
        if num_overs > num_unders:
            if num_favorites > 0:
                return "High-scoring game with favorites covering"
            else:
                return "High-scoring game"
        elif num_unders > num_overs:
            return "Low-scoring defensive game"
        else:
            return "Balanced game script"
