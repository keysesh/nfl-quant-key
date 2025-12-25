"""Parlay odds calculation with transparent math."""

from typing import List, Dict, Optional
import numpy as np

from nfl_quant.core.unified_betting import (
    american_odds_to_implied_prob as _canonical_american_to_implied,
)


class ParlayOddsCalculator:
    """Calculate true parlay odds and compare to book offerings."""

    @staticmethod
    def american_to_implied_prob(american_odds: int) -> float:
        """Convert American odds to implied probability.

        Delegates to canonical implementation in nfl_quant.core.unified_betting.

        Args:
            american_odds: American odds (e.g., -118, +145)

        Returns:
            Implied probability [0, 1]
        """
        return _canonical_american_to_implied(american_odds)
    
    @staticmethod
    def implied_prob_to_american(prob: float) -> int:
        """Convert implied probability to American odds.
        
        Args:
            prob: Probability [0, 1]
            
        Returns:
            American odds (rounded integer)
        """
        if prob >= 0.5:
            odds = -100 * prob / (1 - prob)
            return int(round(odds))
        else:
            odds = 100 * (1 - prob) / prob
            return int(round(odds))
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds.
        
        Args:
            american_odds: American odds
            
        Returns:
            Decimal odds
        """
        if american_odds < 0:
            return 100 / abs(american_odds) + 1
        else:
            return american_odds / 100 + 1
    
    def calculate_parlay_odds(
        self,
        leg_odds: List[int],
        leg_names: List[str],
        leg_probs: Optional[List[float]] = None,
        book_odds: Optional[int] = None
    ) -> Dict:
        """Calculate true parlay odds and compare to book offering.
        
        Args:
            leg_odds: List of American odds for each leg
            leg_names: List of leg descriptions (e.g., "BAL -2.5")
            leg_probs: Optional list of true probabilities (from model)
            book_odds: Optional book parlay odds to compare against
            
        Returns:
            Dictionary with breakdown of odds calculation
        """
        # Step 1: Convert each leg to implied probability
        leg_probs_implied = [self.american_to_implied_prob(odds) for odds in leg_odds]
        
        # Step 2: Calculate true parlay probability
        true_parlay_prob = np.prod(leg_probs_implied)
        
        # Step 3: Convert back to American odds
        true_parlay_odds = self.implied_prob_to_american(true_parlay_prob)
        
        # Calculate extra juice if book odds provided
        if book_odds:
            book_implied_prob = self.american_to_implied_prob(book_odds)
            extra_vig = book_implied_prob - true_parlay_prob
        else:
            book_implied_prob = None
            extra_vig = None
        
        # If custom probabilities provided (from model), calculate edge
        if leg_probs:
            model_parlay_prob = np.prod(leg_probs)
            model_parlay_odds = self.implied_prob_to_american(model_parlay_prob)
        else:
            model_parlay_prob = None
            model_parlay_odds = None
        
        return {
            'legs': [
                {
                    'name': name,
                    'odds': odds,
                    'implied_prob': prob,
                    'model_prob': leg_probs[i] if leg_probs else None
                }
                for i, (name, odds, prob) in enumerate(zip(leg_names, leg_odds, leg_probs_implied))
            ],
            'true_parlay_prob': true_parlay_prob,
            'true_parlay_odds': true_parlay_odds,
            'book_parlay_odds': book_odds,
            'book_implied_prob': book_implied_prob,
            'extra_vig': extra_vig,
            'model_parlay_prob': model_parlay_prob,
            'model_parlay_odds': model_parlay_odds,
            'edge': (model_parlay_prob - true_parlay_prob) if model_parlay_prob else None,
        }
    
    def format_odds_breakdown(self, breakdown: Dict) -> str:
        """Format odds breakdown for display.
        
        Args:
            breakdown: Dictionary from calculate_parlay_odds()
            
        Returns:
            Formatted string
        """
        lines = [
            "Parlay Odds Breakdown:",
            "━" * 50,
        ]
        
        for i, leg in enumerate(breakdown['legs'], 1):
            name = leg['name']
            odds = leg['odds']
            prob = leg['implied_prob']
            model_prob = leg['model_prob']
            
            if model_prob:
                lines.append(f"Leg {i}: {name} ({odds:+d}) → {prob:.1%} implied, {model_prob:.1%} model")
            else:
                lines.append(f"Leg {i}: {name} ({odds:+d}) → {prob:.1%} implied")
        
        lines.append("━" * 50)
        
        lines.append(f"True Parlay Odds: {breakdown['true_parlay_odds']:+d} ({breakdown['true_parlay_prob']:.1%} chance)")
        
        if breakdown['book_parlay_odds']:
            lines.append(f"Book Offering: {breakdown['book_parlay_odds']:+d}")
            if breakdown['extra_vig']:
                lines.append(f"Extra Juice: {breakdown['extra_vig']:.1%} (book adds 7-10% on parlays)")
        
        if breakdown['model_parlay_odds']:
            lines.append(f"Model Odds: {breakdown['model_parlay_odds']:+d} ({breakdown['model_parlay_prob']:.1%} chance)")
            if breakdown['edge']:
                lines.append(f"Edge: {breakdown['edge']:.1%}")
        
        return "\n".join(lines)
    
    def calculate_stake(
        self,
        breakdown: Dict,
        bankroll: float,
        staking_method: str = 'fixed_pct',
        risk_level: str = 'moderate',
        num_legs: int = 3
    ) -> Dict:
        """Calculate recommended stake for parlay.
        
        Args:
            breakdown: Dictionary from calculate_parlay_odds()
            bankroll: Total bankroll
            staking_method: 'fixed_pct' or 'kelly'
            risk_level: 'conservative', 'moderate', or 'aggressive'
            num_legs: Number of legs in parlay
            
        Returns:
            Dictionary with stake recommendation
        """
        # Fixed percentage presets
        PRESETS = {
            'conservative': {
                2: 0.0025,  # 0.25%
                3: 0.0015,  # 0.15%
                4: 0.001,   # 0.10%
                5: 0.0005,  # 0.05%
            },
            'moderate': {
                2: 0.005,   # 0.5%
                3: 0.003,   # 0.3%
                4: 0.002,   # 0.2%
                5: 0.001,   # 0.1%
            },
            'aggressive': {
                2: 0.01,    # 1.0%
                3: 0.0075,  # 0.75%
                4: 0.005,   # 0.5%
                5: 0.0025,  # 0.25%
            },
        }
        
        if staking_method == 'fixed_pct':
            preset_pct = PRESETS[risk_level].get(num_legs, 0.001)
            recommended_stake = bankroll * preset_pct
        else:  # kelly
            # Fractional Kelly for parlays (0.25x recommended)
            if not breakdown['model_parlay_prob']:
                # No model edge, use preset
                preset_pct = PRESETS[risk_level].get(num_legs, 0.001)
                recommended_stake = bankroll * preset_pct
            else:
                # Calculate fractional Kelly
                decimal_odds = self.american_to_decimal(breakdown['true_parlay_odds'])
                net_payoff = decimal_odds - 1
                true_prob = breakdown['model_parlay_prob']
                
                # Full Kelly
                full_kelly = (net_payoff * true_prob - (1 - true_prob)) / net_payoff
                full_kelly = max(0, min(full_kelly, 0.10))  # Cap at 10%
                
                # Fractional Kelly (0.25x)
                fractional_kelly = full_kelly * 0.25
                recommended_stake = bankroll * fractional_kelly
        
        # Cap at 2% of bankroll
        max_stake = bankroll * 0.02
        recommended_stake = min(recommended_stake, max_stake)
        
        # Calculate potential payout
        decimal_odds = self.american_to_decimal(breakdown['true_parlay_odds'])
        potential_win = recommended_stake * (decimal_odds - 1)
        
        # Calculate EV if model probability available
        if breakdown['model_parlay_prob']:
            expected_payout = recommended_stake * decimal_odds * breakdown['model_parlay_prob']
            expected_value = expected_payout - recommended_stake
        else:
            expected_value = None
        
        return {
            'recommended_stake': round(recommended_stake, 2),
            'potential_win': round(potential_win, 2),
            'potential_total': round(recommended_stake + potential_win, 2),
            'expected_value': round(expected_value, 2) if expected_value else None,
            'staking_method': staking_method,
            'risk_level': risk_level,
        }






