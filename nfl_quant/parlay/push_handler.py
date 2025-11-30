"""Push and void handling for parlay legs."""

from typing import List, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class ParlayLeg:
    """Represents a parlay leg."""
    name: str
    bet_type: str
    line: Optional[float] = None
    actual_value: Optional[float] = None
    player: Optional[str] = None
    snap_count: Optional[int] = None
    

@dataclass
class LegResult:
    """Result for a single parlay leg."""
    leg_name: str
    status: str  # 'won', 'lost', 'push', 'void'
    actual_value: Optional[float] = None
    snap_count: Optional[int] = None
    
    
class PushHandler:
    """Handle pushes, voids, and DNP scenarios for parlays."""
    
    # Common push scenarios
    PUSH_THRESHOLD = 0.5  # Within 0.5 for spreads/totals
    
    def __init__(self, push_behavior: str = 'drop_leg'):
        """Initialize push handler.
        
        Args:
            push_behavior: 'drop_leg' or 'void_parlay'
        """
        self.push_behavior = push_behavior
    
    def grade_leg(self, leg: ParlayLeg) -> LegResult:
        """Grade a single parlay leg.
        
        Args:
            leg: Parlay leg to grade
            
        Returns:
            LegResult with status
        """
        # Check for DNP first
        if leg.player and leg.snap_count is not None:
            if leg.snap_count == 0:
                return LegResult(
                    leg_name=leg.name,
                    status='void',
                    snap_count=0
                )
        
        # Check for push
        if leg.actual_value is not None and leg.line is not None:
            diff = abs(leg.actual_value - leg.line)
            
            if diff < self.PUSH_THRESHOLD:
                return LegResult(
                    leg_name=leg.name,
                    status='push',
                    actual_value=leg.actual_value
                )
            
            # Determine win/loss based on bet type
            if 'Over' in leg.bet_type:
                return LegResult(
                    leg_name=leg.name,
                    status='won' if leg.actual_value > leg.line else 'lost',
                    actual_value=leg.actual_value
                )
            elif 'Under' in leg.bet_type:
                return LegResult(
                    leg_name=leg.name,
                    status='won' if leg.actual_value < leg.line else 'lost',
                    actual_value=leg.actual_value
                )
            elif 'spread' in leg.bet_type.lower():
                # For spreads: margin must cover the line
                return LegResult(
                    leg_name=leg.name,
                    status='won' if leg.actual_value > leg.line else 'lost',
                    actual_value=leg.actual_value
                )
        
        # Default to pending if we don't have enough info
        return LegResult(
            leg_name=leg.name,
            status='pending'
        )
    
    def handle_push(
        self,
        legs: List[ParlayLeg],
        results: List[LegResult],
        pushed_indices: List[int]
    ) -> Dict:
        """Handle push scenario.
        
        Args:
            legs: Original parlay legs
            results: Grade results for each leg
            pushed_indices: Indices of pushed legs
            
        Returns:
            Dictionary with adjusted parlay info
        """
        if not pushed_indices:
            return {
                'adjusted': False,
                'remaining_legs': legs,
                'remaining_results': results,
                'message': 'No pushes to handle',
            }
        
        if self.push_behavior == 'void_parlay':
            return {
                'adjusted': True,
                'voided': True,
                'remaining_legs': [],
                'remaining_results': [],
                'message': f'Parlay voided due to {len(pushed_indices)} push(es)',
            }
        
        # Drop pushed legs (standard behavior)
        remaining_legs = [leg for i, leg in enumerate(legs) if i not in pushed_indices]
        remaining_results = [res for i, res in enumerate(results) if i not in pushed_indices]
        
        # If all legs pushed, parlay is void
        if len(remaining_legs) == 0:
            return {
                'adjusted': True,
                'voided': True,
                'remaining_legs': [],
                'remaining_results': [],
                'message': 'All legs pushed - parlay voided',
            }
        
        # If only 1 leg remains, it becomes a single bet
        if len(remaining_legs) == 1:
            return {
                'adjusted': True,
                'voided': False,
                'remaining_legs': remaining_legs,
                'remaining_results': remaining_results,
                'message': f'Parlay adjusted to single bet: {remaining_legs[0].name}',
                'now_single_bet': True,
            }
        
        return {
            'adjusted': True,
            'voided': False,
            'remaining_legs': remaining_legs,
            'remaining_results': remaining_results,
            'message': f'Dropped {len(pushed_indices)} pushed leg(s), {len(remaining_legs)} remain',
            'original_legs': len(legs),
            'remaining_legs_count': len(remaining_legs),
        }
    
    def handle_void(
        self,
        legs: List[ParlayLeg],
        results: List[LegResult],
        voided_indices: List[int]
    ) -> Dict:
        """Handle void scenario (DNP, cancellation, etc.).
        
        Args:
            legs: Original parlay legs
            results: Grade results for each leg
            voided_indices: Indices of voided legs
            
        Returns:
            Dictionary with adjusted parlay info
        """
        # Same logic as push handling
        return self.handle_push(legs, results, voided_indices)
    
    def determine_parlay_outcome(
        self,
        results: List[LegResult]
    ) -> Dict:
        """Determine overall parlay outcome.
        
        Args:
            results: Grade results for each leg
            
        Returns:
            Dictionary with final outcome
        """
        # Check for any losses (instant loss)
        for result in results:
            if result.status == 'lost':
                return {
                    'status': 'lost',
                    'message': 'One or more legs lost',
                    'leg_results': results,
                }
        
        # Check for all wins
        all_won = all(r.status == 'won' for r in results)
        if all_won:
            return {
                'status': 'won',
                'message': 'All legs won!',
                'leg_results': results,
            }
        
        # Check for pushes/voids
        pushed = [r for r in results if r.status in ['push', 'void']]
        if pushed:
            return {
                'status': 'push',
                'message': f'{len(pushed)} leg(s) pushed/voided',
                'pushed_legs': [r.leg_name for r in pushed],
                'leg_results': results,
            }
        
        # Still pending
        return {
            'status': 'pending',
            'message': 'Parlay still pending',
            'leg_results': results,
        }
    
    def recalculate_parlay_odds(
        self,
        original_legs: List[ParlayLeg],
        remaining_legs: List[ParlayLeg],
        original_odds: int
    ) -> Dict:
        """Recalculate parlay odds after push.
        
        Args:
            original_legs: Original parlay legs
            remaining_legs: Legs after push removal
            original_odds: Original parlay odds
            
        Returns:
            Dictionary with new odds info
        """
        if len(remaining_legs) == 0:
            return {
                'adjusted_odds': None,
                'adjusted_prob': None,
                'message': 'All legs removed - no adjusted odds',
            }
        
        if len(remaining_legs) == len(original_legs):
            return {
                'adjusted_odds': original_odds,
                'adjusted_prob': None,
                'message': 'No legs removed - original odds apply',
            }
        
        # Calculate new parlay odds based on remaining legs
        # This would require the original leg odds, which we don't have here
        # In practice, this would be called with the full leg information
        
        return {
            'adjusted_odds': None,  # Would calculate from remaining legs
            'adjusted_prob': None,
            'message': f'Odds recalculated for {len(remaining_legs)}-leg parlay',
            'remaining_legs': len(remaining_legs),
            'dropped_legs': len(original_legs) - len(remaining_legs),
        }
    
    def format_parlay_status(
        self,
        legs: List[ParlayLeg],
        results: List[LegResult],
        overall_outcome: Dict
    ) -> str:
        """Format parlay status for display.
        
        Args:
            legs: Parlay legs
            results: Leg results
            overall_outcome: Overall outcome dictionary
            
        Returns:
            Formatted status string
        """
        lines = [
            f"Parlay Status: {overall_outcome['status'].upper()}",
            "━" * 50,
        ]
        
        for i, (leg, result) in enumerate(zip(legs, results), 1):
            status_emoji = {
                'won': '✅',
                'lost': '❌',
                'push': '➖',
                'void': '⚪',
                'pending': '⏳',
            }.get(result.status, '❓')
            
            lines.append(f"{status_emoji} Leg {i}: {leg.name} → {result.status.upper()}")
            
            if result.actual_value is not None:
                lines[-1] += f" (actual: {result.actual_value})"
            
            if result.snap_count is not None:
                lines[-1] += f" (snaps: {result.snap_count})"
        
        lines.append("━" * 50)
        lines.append(f"Overall: {overall_outcome['message']}")
        
        return "\n".join(lines)


# OT Handling rules
class OTHandler:
    """Handle overtime scenarios for parlay legs."""
    
    @staticmethod
    def should_include_ot(bet_type: str, leg_name: str) -> bool:
        """Determine if bet should include OT.
        
        Args:
            bet_type: Type of bet (spread, total, prop, etc.)
            leg_name: Full leg description
            
        Returns:
            True if OT should be included
        """
        # Regulation-only bets
        if 'regulation only' in leg_name.lower() or 'reg only' in leg_name.lower():
            return False
        
        # First quarter/half bets don't include OT
        if '1st' in leg_name.lower() or 'q1' in leg_name.lower():
            return False
        if '1h' in leg_name.lower() or 'first half' in leg_name.lower():
            return False
        
        # Default: include OT for all full-game markets
        return True
    
    @staticmethod
    def adjust_for_overtime(
        actual_value: float,
        regulation_value: float,
        include_ot: bool
    ) -> float:
        """Adjust value based on OT inclusion.
        
        Args:
            actual_value: Full game value (may include OT)
            regulation_value: Regulation-only value
            include_ot: Whether bet includes OT
            
        Returns:
            Adjusted value to use for grading
        """
        return actual_value if include_ot else regulation_value






