"""
SNR-Based Bet Filtering

Filters bets based on Signal-to-Noise Ratio analysis:
- HIGH SNR markets (receptions): Bet with moderate confidence
- MEDIUM/LOW SNR markets (yards): Require larger edges

Key insight: 0.5 unit edge is 25% of std for receptions but only 1.5% for yards.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from configs.model_config import (
    MARKET_SNR_CONFIG,
    MarketSNRConfig,
    get_market_snr_config,
    get_market_tier,
    get_confidence_threshold,
    get_min_edge,
    should_bet_market,
)

logger = logging.getLogger(__name__)


@dataclass
class BetDecision:
    """Result of SNR-based bet filtering."""
    should_bet: bool
    market: str
    tier: str
    model_confidence: float
    line_vs_trailing: float
    min_confidence_required: float
    min_line_deviation_required: float
    rejection_reason: Optional[str] = None


class SNRFilter:
    """
    Signal-to-Noise Ratio based bet filter.
    
    Uses market-specific thresholds based on empirical analysis:
    - Receptions: High SNR, 0.5 catches detectable
    - Rush yards: Medium SNR, need larger deviation
    - Reception yards: Low SNR, need significant edge
    - Pass yards: Very low SNR, rarely profitable
    """
    
    def __init__(self):
        self.stats = {
            'total_evaluated': 0,
            'total_passed': 0,
            'by_market': {},
            'by_tier': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'rejection_reasons': {},
        }
    
    def evaluate_bet(
        self,
        market: str,
        model_confidence: float,
        line: float,
        trailing_stat: float,
        predicted_mean: Optional[float] = None,
    ) -> BetDecision:
        """
        Evaluate whether a bet passes SNR thresholds.
        
        Args:
            market: Market type (player_receptions, player_rush_yds, etc.)
            model_confidence: Model's predicted probability (0-1)
            line: Betting line
            trailing_stat: Player's trailing average for this stat
            predicted_mean: Optional simulation mean (if available)
            
        Returns:
            BetDecision with should_bet and reasoning
        """
        self.stats['total_evaluated'] += 1
        
        config = get_market_snr_config(market)
        if not config:
            return BetDecision(
                should_bet=False,
                market=market,
                tier='UNKNOWN',
                model_confidence=model_confidence,
                line_vs_trailing=line - trailing_stat,
                min_confidence_required=0.70,
                min_line_deviation_required=5.0,
                rejection_reason=f"Unknown market: {market}",
            )
        
        tier = config.tier
        line_vs_trailing = line - trailing_stat
        
        # Track market stats
        if market not in self.stats['by_market']:
            self.stats['by_market'][market] = {'evaluated': 0, 'passed': 0}
        self.stats['by_market'][market]['evaluated'] += 1
        
        # Check 1: Confidence threshold
        if model_confidence < config.confidence_threshold:
            reason = f"Confidence {model_confidence:.1%} < {config.confidence_threshold:.1%}"
            self._track_rejection(reason)
            return BetDecision(
                should_bet=False,
                market=market,
                tier=tier,
                model_confidence=model_confidence,
                line_vs_trailing=line_vs_trailing,
                min_confidence_required=config.confidence_threshold,
                min_line_deviation_required=config.min_line_deviation,
                rejection_reason=reason,
            )
        
        # Check 2: Line deviation (how far line is from trailing)
        if abs(line_vs_trailing) < config.min_line_deviation:
            reason = f"Line deviation {abs(line_vs_trailing):.1f} < {config.min_line_deviation:.1f}"
            self._track_rejection(reason)
            return BetDecision(
                should_bet=False,
                market=market,
                tier=tier,
                model_confidence=model_confidence,
                line_vs_trailing=line_vs_trailing,
                min_confidence_required=config.confidence_threshold,
                min_line_deviation_required=config.min_line_deviation,
                rejection_reason=reason,
            )
        
        # Passed all checks
        self.stats['total_passed'] += 1
        self.stats['by_market'][market]['passed'] += 1
        self.stats['by_tier'][tier] += 1
        
        return BetDecision(
            should_bet=True,
            market=market,
            tier=tier,
            model_confidence=model_confidence,
            line_vs_trailing=line_vs_trailing,
            min_confidence_required=config.confidence_threshold,
            min_line_deviation_required=config.min_line_deviation,
            rejection_reason=None,
        )
    
    def _track_rejection(self, reason: str):
        """Track rejection reason for stats."""
        # Normalize reason
        key = reason.split(':')[0] if ':' in reason else reason[:30]
        self.stats['rejection_reasons'][key] = self.stats['rejection_reasons'].get(key, 0) + 1
    
    def get_summary(self) -> str:
        """Get summary of filtering statistics."""
        total = self.stats['total_evaluated']
        passed = self.stats['total_passed']
        pass_rate = passed / total * 100 if total > 0 else 0
        
        lines = [
            "=" * 60,
            "SNR FILTER SUMMARY",
            "=" * 60,
            f"Total evaluated: {total}",
            f"Total passed: {passed} ({pass_rate:.1f}%)",
            "",
            "By Market:",
        ]
        
        for market, stats in sorted(self.stats['by_market'].items()):
            ev = stats['evaluated']
            pa = stats['passed']
            rate = pa / ev * 100 if ev > 0 else 0
            config = get_market_snr_config(market)
            tier = config.tier if config else 'UNK'
            lines.append(f"  {market}: {pa}/{ev} ({rate:.1f}%) [{tier}]")
        
        lines.extend([
            "",
            "By Tier:",
            f"  HIGH: {self.stats['by_tier']['HIGH']}",
            f"  MEDIUM: {self.stats['by_tier']['MEDIUM']}",
            f"  LOW: {self.stats['by_tier']['LOW']}",
        ])
        
        if self.stats['rejection_reasons']:
            lines.extend(["", "Top Rejection Reasons:"])
            sorted_reasons = sorted(
                self.stats['rejection_reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for reason, count in sorted_reasons:
                lines.append(f"  {reason}: {count}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = {
            'total_evaluated': 0,
            'total_passed': 0,
            'by_market': {},
            'by_tier': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'rejection_reasons': {},
        }


def filter_bets_by_snr(
    bets: List[Dict[str, Any]],
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter a list of bets by SNR criteria.
    
    Args:
        bets: List of bet dictionaries with keys:
            - market: str
            - model_confidence or confidence: float
            - line: float
            - trailing_stat: float
        verbose: Whether to print summary
        
    Returns:
        (approved_bets, rejected_bets)
    """
    snr_filter = SNRFilter()
    approved = []
    rejected = []
    
    for bet in bets:
        market = bet.get('market', '')
        confidence = bet.get('model_confidence', bet.get('confidence', 0.5))
        line = bet.get('line', 0)
        trailing = bet.get('trailing_stat', line)  # Default to line if no trailing
        
        decision = snr_filter.evaluate_bet(
            market=market,
            model_confidence=confidence,
            line=line,
            trailing_stat=trailing,
        )
        
        bet['snr_decision'] = decision
        bet['snr_tier'] = decision.tier
        bet['snr_passed'] = decision.should_bet
        
        if decision.should_bet:
            approved.append(bet)
        else:
            bet['snr_rejection'] = decision.rejection_reason
            rejected.append(bet)
    
    if verbose:
        print(snr_filter.get_summary())
    
    return approved, rejected


# Convenience function for single bet evaluation
def should_place_bet(
    market: str,
    model_confidence: float,
    line: float,
    trailing_stat: float,
) -> Tuple[bool, str]:
    """
    Quick check if a single bet should be placed.
    
    Returns:
        (should_bet: bool, reason: str)
    """
    snr_filter = SNRFilter()
    decision = snr_filter.evaluate_bet(
        market=market,
        model_confidence=model_confidence,
        line=line,
        trailing_stat=trailing_stat,
    )
    
    if decision.should_bet:
        return True, f"PASS: {decision.tier} tier, conf={decision.model_confidence:.1%}"
    else:
        return False, decision.rejection_reason or "Unknown reason"
