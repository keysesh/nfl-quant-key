"""Correlation detection for parlay legs."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParlayLeg:
    """Represents a parlay leg."""
    name: str
    bet_type: str  # 'spread', 'total', 'player_prop', 'team_total', etc.
    game: str  # e.g., "CHI @ BAL"
    team: Optional[str] = None
    player: Optional[str] = None
    market: Optional[str] = None  # e.g., 'player_pass_yds', 'player_rush_yds'
    odds: Optional[int] = None
    
    def __post_init__(self):
        """Extract team if from game string."""
        if self.game and not self.team:
            # Try to extract team from game string
            if ' @ ' in self.game:
                self.team = self.game.split(' @ ')[-1]  # Home team


class CorrelationChecker:
    """Detect correlated combinations in parlay legs."""

    # Same-player correlations (very high)
    SAME_PLAYER_CORRELATIONS = {
        ('player_rush_yds', 'player_anytime_td'): 0.75,
        ('player_receptions', 'player_reception_yds'): 0.85,
        ('player_pass_yds', 'player_pass_tds'): 0.70,
        ('player_rush_attempts', 'player_rush_yds'): 0.90,
        ('player_targets', 'player_receptions'): 0.95,
        ('player_reception_yds', 'player_reception_tds'): 0.65,
        ('player_rush_yds', 'player_rush_tds'): 0.70,
        ('player_pass_yds', 'player_passing_tds'): 0.68,
    }

    # Team correlations (moderate - same team offensive stats)
    TEAM_CORRELATIONS = {
        ('team_total_over', 'qb_pass_yds'): 0.65,
        ('team_total_over', 'qb_pass_tds'): 0.60,
        ('home_total_over', 'home_pass_yds'): 0.70,
        ('away_total_over', 'away_pass_yds'): 0.70,
        # Teammates on same offense (moderate positive)
        ('wr1_rec_yds', 'wr2_rec_yds'): 0.40,  # Same passing volume
        ('qb_pass_yds', 'wr_rec_yds'): 0.55,   # QB feeds WR
        ('rb_rush_yds', 'positive_game_script'): 0.50,
    }

    # Game correlations
    GAME_CORRELATIONS = {
        ('first_half_over', 'full_game_over'): 0.70,
        ('first_half_spread', 'full_game_spread'): 0.55,
        ('first_quarter_over', 'full_game_over'): 0.60,
    }

    # Anti-correlations (negative correlation)
    ANTI_CORRELATIONS = {
        ('team_total_over', 'opponent_total_under'): -0.30,
        ('team_total_under', 'opponent_total_over'): -0.30,
        # Competing volume on same team
        ('rb1_rush_attempts', 'rb2_rush_attempts'): -0.45,
        ('wr1_targets', 'wr2_targets'): -0.25,
    }
    
    def __init__(self):
        """Initialize correlation checker."""
        self.all_pairs = {
            **self.SAME_PLAYER_CORRELATIONS,
            **self.TEAM_CORRELATIONS,
            **self.GAME_CORRELATIONS,
            **self.ANTI_CORRELATIONS,
        }
    
    def check_correlation(
        self,
        leg1: ParlayLeg,
        leg2: ParlayLeg
    ) -> Tuple[bool, str]:
        """Check if two legs are correlated.
        
        Args:
            leg1: First parlay leg
            leg2: Second parlay leg
            
        Returns:
            Tuple of (is_correlated, reason_string)
        """
        # Same player, different stats
        if leg1.player and leg2.player and leg1.player == leg2.player:
            if leg1.market and leg2.market:
                pair = (leg1.market, leg2.market)
                if pair in self.SAME_PLAYER_CORRELATIONS:
                    correlation = self.SAME_PLAYER_CORRELATIONS[pair]
                    return True, f"Same player ({leg1.player}) correlations: {leg1.market} + {leg2.market} ({correlation:.0%})"
                
                # Check reverse order
                reverse_pair = (leg2.market, leg1.market)
                if reverse_pair in self.SAME_PLAYER_CORRELATIONS:
                    correlation = self.SAME_PLAYER_CORRELATIONS[reverse_pair]
                    return True, f"Same player ({leg1.player}) correlations: {leg2.market} + {leg1.market} ({correlation:.0%})"
        
        # Same team, scoring correlations
        if leg1.team and leg2.team and leg1.team == leg2.team:
            # Team total + player prop (same team)
            if leg1.bet_type in ['team_total', 'home_total', 'away_total']:
                if leg2.bet_type == 'player_prop' and leg2.market:
                    # QB passing drives team scoring
                    if 'pass_yds' in leg2.market or 'pass_tds' in leg2.market:
                        return True, f"Team {leg1.team} total correlates with QB passing stats (QB drives team scoring)"
            
            # Player over + team over
            if leg1.bet_type == 'player_prop' and leg2.bet_type in ['team_total', 'home_total', 'away_total']:
                if leg1.market and ('pass_yds' in leg1.market or 'pass_tds' in leg1.market or 'reception_yds' in leg1.market or 'rush_yds' in leg1.market):
                    return True, f"{leg1.player} props correlate with {leg2.team} team total"
        
        # Same game, game-script correlations
        if leg1.game == leg2.game:
            # Game total + specific player props
            if leg1.bet_type == 'total' and leg2.bet_type == 'player_prop':
                if leg2.market and ('pass_yds' in leg2.market or 'reception_yds' in leg2.market):
                    return True, f"Game total correlates with {leg2.player} passing/receiving props"
        
        # First half vs full game
        if 'first_half' in leg1.name.lower() and leg1.game == leg2.game:
            if leg2.bet_type in ['total', 'spread']:
                return True, f"First half vs full game are correlated"
        
        # Not correlated
        return False, ""
    
    def validate_parlay(self, legs: List[ParlayLeg]) -> Dict:
        """Validate a complete parlay for correlations.
        
        Args:
            legs: List of parlay legs
            
        Returns:
            Dictionary with validation results
        """
        correlations = []
        blocked = False
        
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                is_correlated, reason = self.check_correlation(legs[i], legs[j])
                
                if is_correlated:
                    correlations.append({
                        'leg1_idx': i,
                        'leg2_idx': j,
                        'leg1_name': legs[i].name,
                        'leg2_name': legs[j].name,
                        'reason': reason,
                    })
                    
                    # Block if high correlation (>0.70)
                    if any(keyword in reason.lower() for keyword in ['85%', '90%', '95%', 'very high']):
                        blocked = True
        
        return {
            'valid': len(correlations) == 0 and not blocked,
            'correlations': correlations,
            'blocked': blocked,
            'message': self._format_message(correlations, blocked),
        }
    
    def _format_message(self, correlations: List[Dict], blocked: bool) -> str:
        """Format validation message.
        
        Args:
            correlations: List of correlation issues
            blocked: Whether parlay is blocked
            
        Returns:
            Formatted message
        """
        if not correlations:
            return "✅ No correlated legs detected"
        
        if blocked:
            msg = "❌ PARLAY BLOCKED - High correlation detected:\n"
        else:
            msg = "⚠️  CORRELATION WARNING:\n"
        
        for i, corr in enumerate(correlations, 1):
            msg += f"{i}. {corr['leg1_name']} + {corr['leg2_name']}\n"
            msg += f"   → {corr['reason']}\n"
        
        if not blocked:
            msg += "\nConsider removing one leg to reduce correlation risk."
        
        return msg
    
    def suggest_alternatives(
        self,
        legs: List[ParlayLeg],
        correlation_issue: Dict
    ) -> List[str]:
        """Suggest alternative legs to reduce correlation.
        
        Args:
            legs: Original parlay legs
            correlation_issue: Specific correlation issue
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        leg1_idx = correlation_issue['leg1_idx']
        leg2_idx = correlation_issue['leg2_idx']
        
        reason = correlation_issue['reason'].lower()
        
        if 'same player' in reason:
            suggestions.append(f"Remove one of the {legs[leg1_idx].player} props")
        
        if 'team total' in reason and 'qb' in reason:
            suggestions.append("Remove team total OR QB props (pick one direction)")
        
        if 'first half' in reason:
            suggestions.append("Use either first half OR full game markets, not both")
        
        if not suggestions:
            suggestions.append("Consider removing one of the correlated legs")
        
        return suggestions






