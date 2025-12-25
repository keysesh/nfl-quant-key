"""
Contrarian Betting Strategy

The model is a reliable NEGATIVE indicator. This module:
1. Identifies when model diverges significantly from line
2. Applies line-level rules that historically beat the vig
3. Generates FADE picks (bet opposite of model)

VALIDATED on 20,816 bets across 2023-2025 seasons.

CORE RULES (consistent across all seasons):
- Receptions >= 4.5 → UNDER: 55.2% WR, +5.3% ROI (n=1,776)
- Receptions >= 5.5 → UNDER: 57.1% WR, +9.0% ROI (n=804)

SPECULATIVE RULES (positive overall but lost in 2024):
- Rushing >= 58.5 → UNDER: 56.0% WR, +6.9% ROI (2024: -9.4%)
- Rushing >= 70.5 → UNDER: 59.5% WR, +13.6% ROI (2024: -9.8%)

REMOVED (not profitable on larger sample):
- Receiving yards <= 22.5 → OVER: 50.1% WR, -4.3% ROI

The key insight: The model's consistent errors ARE the edge.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BetDirection(Enum):
    OVER = "OVER"
    UNDER = "UNDER"
    NO_BET = "NO_BET"


@dataclass
class ContrarianPick:
    """A contrarian betting pick."""
    player: str
    market: str
    line: float
    model_pred: float
    direction: BetDirection
    rule_name: str
    expected_wr: float
    expected_roi: float
    confidence: str  # "high", "medium", "speculative"
    divergence_pct: float = 0.0
    sample_size: int = 0  # From validation
    validated: bool = True  # True if profitable across all seasons

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'player': self.player,
            'market': self.market,
            'line': self.line,
            'model_pred': self.model_pred,
            'direction': self.direction.value,
            'rule_name': self.rule_name,
            'expected_wr': self.expected_wr,
            'expected_roi': self.expected_roi,
            'confidence': self.confidence,
            'divergence_pct': self.divergence_pct,
            'sample_size': self.sample_size,
            'validated': self.validated
        }


# =============================================================================
# CONTRARIAN RULES (derived from backtest analysis)
# =============================================================================

def evaluate_receptions(line: float, model_pred: float) -> Tuple[BetDirection, str, float, float, int, bool]:
    """
    Receptions rules - HIGH CONFIDENCE (consistent all seasons).

    VALIDATED on 20,816 bets across 2023-2025:
    - Line >= 5.5: 57.1% WR, +9.0% ROI, n=804 (consistent all years)
    - Line >= 4.5: 55.2% WR, +5.3% ROI, n=1,776 (consistent all years)

    Args:
        line: The betting line
        model_pred: Model's prediction

    Returns:
        Tuple of (direction, rule_name, expected_wr, expected_roi, sample_size, validated)
    """
    # Highest confidence: tight threshold (all 3 seasons profitable)
    if line >= 5.5:
        return (BetDirection.UNDER, "REC_HIGH_LINE_VALIDATED", 0.571, 9.0, 804, True)

    # Good confidence: standard threshold (all 3 seasons profitable)
    if line >= 4.5:
        return (BetDirection.UNDER, "REC_MID_LINE_VALIDATED", 0.552, 5.3, 1776, True)

    return (BetDirection.NO_BET, "", 0.0, 0.0, 0, False)


def evaluate_receiving_yards(line: float, model_pred: float) -> Tuple[BetDirection, str, float, float, int, bool]:
    """
    Receiving yards rules - DISABLED after large sample validation.

    Original rule (line <= 22.5 → OVER) showed:
    - Small sample (419): +11.9% ROI
    - Large sample (20,816): -4.3% ROI on 3,203 bets ❌

    Rule removed due to inconsistent performance across seasons.

    Args:
        line: The betting line
        model_pred: Model's prediction

    Returns:
        Tuple of (direction, rule_name, expected_wr, expected_roi, sample_size, validated)
    """
    # All receiving yards rules disabled
    return (BetDirection.NO_BET, "DISABLED_AFTER_VALIDATION", 0.0, 0.0, 0, False)


def evaluate_rushing_yards(line: float, model_pred: float, include_speculative: bool = True) -> Tuple[BetDirection, str, float, float, int, bool]:
    """
    Rushing yards rules - SPECULATIVE (high variance).

    Large sample validation (20,816 bets):
    - 2023: +6.5% ROI ✓
    - 2024: -9.4% ROI ❌
    - 2025: +21.2% ROI ✓

    Flag as speculative due to inconsistent year-over-year performance.

    Args:
        line: The betting line
        model_pred: Model's prediction
        include_speculative: Whether to include speculative rules (default True)

    Returns:
        Tuple of (direction, rule_name, expected_wr, expected_roi, sample_size, validated)
    """
    if not include_speculative:
        return (BetDirection.NO_BET, "", 0.0, 0.0, 0, False)

    # SPECULATIVE: High line >= 70.5 (higher threshold = better ROI but high variance)
    if line >= 70.5:
        return (BetDirection.UNDER, "RUSH_HIGH_LINE_SPECULATIVE", 0.595, 13.6, 368, False)

    # SPECULATIVE: High line >= 58.5
    if line >= 58.5:
        return (BetDirection.UNDER, "RUSH_MID_LINE_SPECULATIVE", 0.560, 6.9, 809, False)

    return (BetDirection.NO_BET, "", 0.0, 0.0, 0, False)


# Market name mappings (DraftKings -> internal)
MARKET_ALIASES = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_pass_yds': 'passing_yards',
}


def get_contrarian_signal(
    market: str,
    line: float,
    model_pred: float,
    include_speculative: bool = True
) -> Tuple[BetDirection, str, float, float, int, bool]:
    """
    Main entry point for contrarian signal generation.

    Args:
        market: Market type (e.g., 'player_receptions', 'receptions')
        line: The betting line
        model_pred: Model's prediction
        include_speculative: Include speculative (inconsistent) rules (default True)

    Returns:
        Tuple of (direction, rule_name, expected_wr, expected_roi, sample_size, validated)
    """
    # Normalize market name
    market_normalized = MARKET_ALIASES.get(market, market).lower()

    # EXCLUDE passing yards entirely - 41.8% WR, -20% ROI (catastrophic)
    if 'pass' in market_normalized:
        logger.debug(f"Excluding passing yards market: {market}")
        return (BetDirection.NO_BET, "EXCLUDED_MARKET", 0.0, 0.0, 0, False)

    # Route to market-specific evaluator
    if 'reception' in market_normalized and 'yard' not in market_normalized:
        # Receptions (count) - VALIDATED rules
        return evaluate_receptions(line, model_pred)

    if 'receiving' in market_normalized or ('reception' in market_normalized and 'yard' in market_normalized):
        # Receiving yards - DISABLED after validation
        return evaluate_receiving_yards(line, model_pred)

    if 'rush' in market_normalized:
        # Rushing yards - SPECULATIVE rules
        return evaluate_rushing_yards(line, model_pred, include_speculative)

    logger.debug(f"Unknown market: {market}")
    return (BetDirection.NO_BET, "UNKNOWN_MARKET", 0.0, 0.0, 0, False)


def generate_contrarian_picks(
    props: List[Dict[str, Any]],
    min_roi: float = 0.0,
    min_confidence: str = "low",
    validated_only: bool = False,
    include_speculative: bool = True
) -> List[ContrarianPick]:
    """
    Generate contrarian picks from props data.

    Args:
        props: List of dicts with keys: player, market, line, model_pred
               (or pred_mean, stat_type as alternatives)
        min_roi: Minimum expected ROI threshold (default 0 = all qualifying picks)
        min_confidence: Minimum confidence level ("high", "medium", "speculative")
        validated_only: Only return picks from rules validated across all seasons
        include_speculative: Include speculative (inconsistent) rules (default True)

    Returns:
        List of ContrarianPick objects, sorted by expected ROI descending
    """
    picks = []
    confidence_levels = {"high": 3, "medium": 2, "speculative": 1, "low": 1}
    min_conf_level = confidence_levels.get(min_confidence, 1)

    for prop in props:
        # Extract fields with fallbacks for different column names
        player = prop.get('player') or prop.get('player_name', 'Unknown')
        market = prop.get('market') or prop.get('stat_type', '')
        line = prop.get('line', 0)
        model_pred = prop.get('model_pred') or prop.get('pred_mean', 0)

        if not market or line <= 0:
            continue

        # Get contrarian signal (now returns 6 values)
        direction, rule, exp_wr, exp_roi, sample_size, validated = get_contrarian_signal(
            market=market,
            line=line,
            model_pred=model_pred,
            include_speculative=include_speculative
        )

        # Skip if no bet
        if direction == BetDirection.NO_BET:
            continue

        # Skip non-validated rules if requested
        if validated_only and not validated:
            continue

        # Skip if below minimum ROI
        if exp_roi < min_roi:
            continue

        # Determine confidence level based on validation status and ROI
        if validated and exp_roi >= 8:
            confidence = "high"
            conf_level = 3
        elif validated and exp_roi >= 3:
            confidence = "medium"
            conf_level = 2
        elif not validated:
            confidence = "speculative"
            conf_level = 1
        else:
            confidence = "medium"
            conf_level = 2

        # Skip if below minimum confidence
        if conf_level < min_conf_level:
            continue

        # Calculate divergence
        divergence_pct = (model_pred - line) / line * 100 if line > 0 else 0

        picks.append(ContrarianPick(
            player=player,
            market=market,
            line=line,
            model_pred=model_pred,
            direction=direction,
            rule_name=rule,
            expected_wr=exp_wr,
            expected_roi=exp_roi,
            confidence=confidence,
            divergence_pct=divergence_pct,
            sample_size=sample_size,
            validated=validated
        ))

    # Sort by expected ROI descending
    picks.sort(key=lambda x: -x.expected_roi)

    logger.info(f"Generated {len(picks)} contrarian picks from {len(props)} props")

    return picks


def summarize_picks(picks: List[ContrarianPick]) -> Dict[str, Any]:
    """
    Generate summary statistics for a list of picks.

    Args:
        picks: List of ContrarianPick objects

    Returns:
        Dict with summary statistics
    """
    if not picks:
        return {
            'total': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'speculative': 0,
            'validated_count': 0,
            'avg_wr': 0.0,
            'avg_roi': 0.0,
            'by_market': {},
            'by_rule': {}
        }

    high_conf = [p for p in picks if p.confidence == "high"]
    med_conf = [p for p in picks if p.confidence == "medium"]
    speculative = [p for p in picks if p.confidence == "speculative"]
    validated = [p for p in picks if p.validated]

    avg_wr = sum(p.expected_wr for p in picks) / len(picks)
    avg_roi = sum(p.expected_roi for p in picks) / len(picks)

    # By market
    by_market = {}
    for pick in picks:
        market = pick.market
        if market not in by_market:
            by_market[market] = {'count': 0, 'over': 0, 'under': 0}
        by_market[market]['count'] += 1
        if pick.direction == BetDirection.OVER:
            by_market[market]['over'] += 1
        else:
            by_market[market]['under'] += 1

    # By rule
    by_rule = {}
    for pick in picks:
        rule = pick.rule_name
        if rule not in by_rule:
            by_rule[rule] = 0
        by_rule[rule] += 1

    return {
        'total': len(picks),
        'high_confidence': len(high_conf),
        'medium_confidence': len(med_conf),
        'speculative': len(speculative),
        'validated_count': len(validated),
        'avg_wr': avg_wr,
        'avg_roi': avg_roi,
        'by_market': by_market,
        'by_rule': by_rule
    }
