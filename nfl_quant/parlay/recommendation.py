"""Parlay recommendation engine - generate parlay combinations from single bets.

Enhanced with:
- Empirical correlation matrix (from historical data)
- Correlation-adjusted probability calculator (Gaussian copula)
- Parlay Kelly criterion (variance-adjusted)
- Cross-game only enforcement (no SGP)
"""

from typing import List, Dict, Optional, Tuple
import itertools
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

try:
    from .odds_calculator import ParlayOddsCalculator
    from .correlation import CorrelationChecker, ParlayLeg
    from .push_handler import PushHandler
    from .correlation_adjusted import CorrelationAdjustedCalculator, ParlayLeg as CAParlayLeg
    from .odds_fetcher import ParlayOddsFetcher
except ImportError:
    from odds_calculator import ParlayOddsCalculator
    from correlation import CorrelationChecker, ParlayLeg
    from push_handler import PushHandler
    try:
        from correlation_adjusted import CorrelationAdjustedCalculator, ParlayLeg as CAParlayLeg
        from odds_fetcher import ParlayOddsFetcher
    except ImportError:
        CorrelationAdjustedCalculator = None
        CAParlayLeg = None
        ParlayOddsFetcher = None

try:
    from nfl_quant.betting.parlay_kelly import ParlayKelly, calculate_parlay_odds
except ImportError:
    ParlayKelly = None
    calculate_parlay_odds = None


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
    # New fields for enhanced parlay system
    naive_prob: Optional[float] = None  # Naive probability (product)
    adjusted_prob: Optional[float] = None  # Correlation-adjusted probability
    correlation_factor: Optional[float] = None  # adjusted/naive ratio
    naive_ev: Optional[float] = None  # EV using naive probability
    adjusted_ev: Optional[float] = None  # EV using adjusted probability
    recommended_units: Optional[float] = None  # Parlay Kelly units
    games: Optional[str] = None  # List of games involved
    sources: Optional[str] = None  # Edge sources (LVT, PLAYER_BIAS, BOTH)


class ParlayRecommender:
    """Generate parlay recommendations from single bets.

    Enhanced with:
    - Empirical correlation matrix
    - Correlation-adjusted probability (Gaussian copula)
    - Parlay Kelly criterion with variance penalty
    - Cross-game only enforcement
    """

    def __init__(
        self,
        correlation_threshold: float = None,
        max_legs: int = None,
        min_confidence: float = None,
        min_edge: float = None,
        use_empirical_correlations: bool = True,
        cross_game_only: bool = True,
        bankroll: float = 1000.0
    ):
        """Initialize parlay recommender.

        Args:
            correlation_threshold: Maximum allowed correlation (loads from config if None)
            max_legs: Maximum legs per parlay (loads from config if None)
            min_confidence: Minimum confidence for bet inclusion (loads from config if None)
            min_edge: Minimum edge required (loads from config if None)
            use_empirical_correlations: Use learned correlations from historical data
            cross_game_only: Only allow cross-game parlays (no SGP)
            bankroll: Bankroll for Kelly sizing
        """
        parlay_config = BETTING_CONFIG.get("parlay_optimization", {})

        self.correlation_threshold = correlation_threshold if correlation_threshold is not None else parlay_config.get("correlation_threshold", 0.70)
        self.max_legs = max_legs if max_legs is not None else parlay_config.get("max_legs", 4)
        self.min_confidence = min_confidence if min_confidence is not None else parlay_config.get("min_confidence_per_leg", 0.52)
        self.min_edge = min_edge if min_edge is not None else parlay_config.get("min_edge_per_leg", 0.02)
        self.use_empirical_correlations = use_empirical_correlations
        self.cross_game_only = cross_game_only
        self.bankroll = bankroll

        self.calculator = ParlayOddsCalculator()
        self.correlation_checker = CorrelationChecker()

        # Initialize enhanced components if available
        self.correlation_adjusted_calc = None
        self.parlay_kelly = None
        self.odds_fetcher = None

        if use_empirical_correlations and CorrelationAdjustedCalculator is not None:
            try:
                self.correlation_adjusted_calc = CorrelationAdjustedCalculator()
            except Exception as e:
                print(f"Warning: Could not load correlation calculator: {e}")

        if ParlayKelly is not None:
            self.parlay_kelly = ParlayKelly()

        if ParlayOddsFetcher is not None:
            self.odds_fetcher = ParlayOddsFetcher()

    def generate_parlays(
        self,
        single_bets: List[SingleBet],
        num_parlays: int = 10,
        max_leg_reuse: int = 3,
        diversify: bool = True
    ) -> List[ParlayRecommendation]:
        """Generate top parlay recommendations from single bets.

        Enhanced with:
        - EV-based ranking (not just probability)
        - Leg diversity constraint (limit reuse)
        - Variety tiers (best per leg count)
        - Market concentration penalty

        Args:
            single_bets: List of single bet recommendations
            num_parlays: Number of parlays to return
            max_leg_reuse: Maximum times a leg can appear across parlays
            diversify: If True, enforce diversity constraints

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

        # Generate ALL valid combinations first
        all_parlays = []

        for num_legs in range(2, min(self.max_legs + 1, len(qualified_bets) + 1)):
            for combo in itertools.combinations(qualified_bets, num_legs):
                parlay = self._create_parlay(list(combo))
                if parlay:
                    # Calculate diversity score (penalty for market concentration)
                    diversity_score = self._calculate_diversity_score(list(combo))
                    parlay._diversity_score = diversity_score
                    all_parlays.append(parlay)

        if not all_parlays:
            return []

        if not diversify:
            # Simple mode: just sort by EV and return top N
            all_parlays.sort(key=lambda p: p.adjusted_ev or 0, reverse=True)
            return all_parlays[:num_parlays]

        # SMART DIVERSIFICATION MODE
        return self._select_diverse_parlays(
            all_parlays,
            num_parlays=num_parlays,
            max_leg_reuse=max_leg_reuse
        )

    def _calculate_diversity_score(self, bets: List[SingleBet]) -> float:
        """Calculate diversity score for a parlay (higher = more diverse).

        Penalizes:
        - Multiple bets from same market type (all rush attempts)
        - Multiple bets from same position (all RBs)
        - Similar line types (all UNDERs on same stat)

        Args:
            bets: List of bets in the parlay

        Returns:
            Diversity score (0-1, higher is better)
        """
        if len(bets) <= 1:
            return 1.0

        # Count market types
        markets = [bet.market for bet in bets if bet.market]
        market_diversity = len(set(markets)) / len(markets) if markets else 1.0

        # Count teams (more teams = more diverse)
        teams = [bet.team for bet in bets if bet.team]
        team_diversity = len(set(teams)) / len(teams) if teams else 1.0

        # Check for all same direction (all UNDERs or all OVERs)
        directions = []
        for bet in bets:
            if 'UNDER' in bet.name.upper() or 'UNDER' in bet.bet_type.upper():
                directions.append('UNDER')
            elif 'OVER' in bet.name.upper() or 'OVER' in bet.bet_type.upper():
                directions.append('OVER')
        direction_diversity = len(set(directions)) / len(directions) if directions else 1.0

        # Weighted average
        return 0.4 * market_diversity + 0.4 * team_diversity + 0.2 * direction_diversity

    def _select_diverse_parlays(
        self,
        all_parlays: List[ParlayRecommendation],
        num_parlays: int,
        max_leg_reuse: int
    ) -> List[ParlayRecommendation]:
        """Select diverse parlays with leg reuse constraints.

        Strategy:
        1. Group by leg count (2-leg, 3-leg, 4-leg)
        2. For each group, select best by EV with diversity
        3. Enforce max_leg_reuse across ALL selected parlays
        4. Return balanced mix across leg counts

        Args:
            all_parlays: All valid parlays
            num_parlays: Total parlays to return
            max_leg_reuse: Max times any leg can appear

        Returns:
            Diverse list of parlays
        """
        from collections import Counter

        # Group by leg count
        by_legs = {}
        for parlay in all_parlays:
            n = len(parlay.legs)
            if n not in by_legs:
                by_legs[n] = []
            by_legs[n].append(parlay)

        # Sort each group by composite score (EV * diversity)
        for n in by_legs:
            by_legs[n].sort(
                key=lambda p: (p.adjusted_ev or 0) * getattr(p, '_diversity_score', 1.0),
                reverse=True
            )

        # Allocate slots per leg count (balanced)
        leg_counts = sorted(by_legs.keys())
        slots_per_group = max(1, num_parlays // len(leg_counts))
        extra_slots = num_parlays - (slots_per_group * len(leg_counts))

        # Track leg usage
        leg_usage = Counter()
        selected = []

        # Select from each group
        for i, n in enumerate(leg_counts):
            group = by_legs[n]
            slots = slots_per_group + (1 if i < extra_slots else 0)

            for parlay in group:
                if len(selected) >= num_parlays:
                    break

                # Check leg reuse constraint
                leg_names = [leg.name for leg in parlay.legs]
                can_use = all(leg_usage[name] < max_leg_reuse for name in leg_names)

                if can_use:
                    selected.append(parlay)
                    for name in leg_names:
                        leg_usage[name] += 1

                    slots -= 1
                    if slots <= 0:
                        break

        # If we didn't fill all slots, add more from best remaining
        if len(selected) < num_parlays:
            remaining = [p for p in all_parlays if p not in selected]
            remaining.sort(key=lambda p: p.adjusted_ev or 0, reverse=True)

            for parlay in remaining:
                if len(selected) >= num_parlays:
                    break

                leg_names = [leg.name for leg in parlay.legs]
                can_use = all(leg_usage[name] < max_leg_reuse for name in leg_names)

                if can_use:
                    selected.append(parlay)
                    for name in leg_names:
                        leg_usage[name] += 1

        # Final sort by EV for output
        selected.sort(key=lambda p: p.adjusted_ev or 0, reverse=True)

        return selected

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

        Enhanced with:
        - Correlation-adjusted probability
        - Parlay Kelly sizing
        - Cross-game enforcement

        Args:
            bets: List of bets to combine

        Returns:
            ParlayRecommendation or None if invalid
        """
        if len(bets) < 2:
            return None

        # Enforce cross-game only (no SGP)
        games = [bet.game for bet in bets]
        if self.cross_game_only and len(set(games)) < len(bets):
            # Multiple bets from same game - skip for cross-game only mode
            return None

        # Check correlation using static checker
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

        # Calculate parlay odds
        if self.odds_fetcher is not None:
            parlay_american, parlay_decimal, _ = self.odds_fetcher.calculate_parlay_odds(leg_odds)
        else:
            breakdown = self.calculator.calculate_parlay_odds(
                leg_odds=leg_odds,
                leg_names=leg_names,
                leg_probs=leg_probs
            )
            parlay_american = breakdown['true_parlay_odds']
            parlay_decimal = self.calculator.american_to_decimal(parlay_american)

        # Calculate naive probability (product)
        naive_prob = float(np.prod(leg_probs))

        # Calculate correlation-adjusted probability
        adjusted_prob = naive_prob
        correlation_factor = 1.0

        if self.correlation_adjusted_calc is not None and CAParlayLeg is not None:
            try:
                # Convert to CAParlayLeg format
                ca_legs = [
                    CAParlayLeg(
                        player=bet.player or "",
                        team=bet.team or "",
                        market=bet.market or "",
                        direction=bet.bet_type.split()[-1] if bet.bet_type else "UNDER",
                        line=0.0,  # Not used for correlation
                        confidence=bet.our_prob or 0.5,
                        game=bet.game or ""
                    )
                    for bet in bets
                ]
                naive_prob, adjusted_prob, correlation_factor = \
                    self.correlation_adjusted_calc.calculate_joint_probability(ca_legs)
            except Exception as e:
                print(f"Warning: Correlation adjustment failed: {e}")
                adjusted_prob = naive_prob
                correlation_factor = 1.0

        # Calculate edge using adjusted probability
        implied_prob = 1.0 / parlay_decimal if parlay_decimal > 0 else 0.0
        edge = adjusted_prob - implied_prob

        # Calculate EV
        naive_ev = (naive_prob * parlay_decimal - 1) * 10  # For $10 bet
        adjusted_ev = (adjusted_prob * parlay_decimal - 1) * 10

        # Calculate Kelly sizing
        recommended_units = 0.0
        recommended_stake = 0.0

        if self.parlay_kelly is not None and edge > 0.05:  # 5% min edge
            try:
                kelly_result = self.parlay_kelly.calculate_parlay_kelly(
                    adjusted_prob=adjusted_prob,
                    parlay_odds=parlay_american,
                    num_legs=len(bets),
                    bankroll=self.bankroll
                )
                if kelly_result.should_bet:
                    recommended_stake = kelly_result.recommended_stake
                    recommended_units = self.parlay_kelly.calculate_units(
                        adjusted_prob=adjusted_prob,
                        parlay_odds=parlay_american,
                        num_legs=len(bets)
                    )
            except Exception as e:
                print(f"Warning: Kelly calculation failed: {e}")

        # Fallback stake calculation
        if recommended_stake == 0:
            bet_sizing_config = BETTING_CONFIG.get("bet_sizing", {})
            bankroll = bet_sizing_config.get("default_bankroll", 50.0)
            stake_info = self.calculator.calculate_stake(
                breakdown={'model_parlay_prob': adjusted_prob, 'true_parlay_odds': parlay_american},
                bankroll=bankroll,
                staking_method='fixed_pct',
                risk_level='moderate',
                num_legs=len(bets)
            )
            recommended_stake = stake_info.get('recommended_stake', 0)

        # Calculate potential win
        potential_win = recommended_stake * (parlay_decimal - 1)

        # Collect games and sources
        unique_games = list(set(games))
        games_str = " | ".join(unique_games)

        # Get sources from bets (if available)
        sources = set()
        for bet in bets:
            if hasattr(bet, 'source') and bet.source:
                sources.add(bet.source)
        sources_str = ", ".join(sources) if sources else "EDGE"

        return ParlayRecommendation(
            legs=bets,
            true_odds=parlay_american,
            model_odds=parlay_american,
            true_prob=implied_prob,
            model_prob=adjusted_prob,
            edge=edge,
            correlation_valid=validation['valid'],
            correlation_issues=validation.get('correlations', []),
            recommended_stake=recommended_stake,
            potential_win=potential_win,
            expected_value=adjusted_ev,
            # New enhanced fields
            naive_prob=naive_prob,
            adjusted_prob=adjusted_prob,
            correlation_factor=correlation_factor,
            naive_ev=naive_ev,
            adjusted_ev=adjusted_ev,
            recommended_units=recommended_units,
            games=games_str,
            sources=sources_str
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

        Enhanced with correlation-adjusted fields.

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
                'correlation_valid': 'Y' if parlay.correlation_valid else 'N',
                'recommended_stake': f"${parlay.recommended_stake:.2f}",
                'potential_win': f"${parlay.potential_win:.2f}",
                'expected_value': f"${parlay.expected_value:.2f}" if parlay.expected_value else "N/A",
                # New enhanced fields
                'naive_prob': f"{parlay.naive_prob:.2%}" if parlay.naive_prob else "N/A",
                'adjusted_prob': f"{parlay.adjusted_prob:.2%}" if parlay.adjusted_prob else "N/A",
                'correlation_factor': f"{parlay.correlation_factor:.3f}" if parlay.correlation_factor else "1.000",
                'naive_ev': f"${parlay.naive_ev:.2f}" if parlay.naive_ev else "N/A",
                'adjusted_ev': f"${parlay.adjusted_ev:.2f}" if parlay.adjusted_ev else "N/A",
                'recommended_units': f"{parlay.recommended_units:.2f}u" if parlay.recommended_units else "0.00u",
                'games': parlay.games or "",
                'sources': parlay.sources or "",
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
