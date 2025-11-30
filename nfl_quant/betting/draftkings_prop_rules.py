"""
DraftKings Prop Betting Rules Engine

This module implements DraftKings-specific rules for NFL player props,
with special attention to touchdown scorer markets.

Official DraftKings Rules:
- Anytime TD scorer: Player must physically cross the goal line with the ball
  OR catch the ball in the end zone
- Passing TDs DO NOT count for QB anytime TD bets
- Only rushing or receiving TDs count for QB anytime TD markets
- Player must be active and play at least one snap (including special teams)

Source: DraftKings Sportsbook House Rules (2024)
https://sportsbook.draftkings.com/help/sport-rules/football
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DraftKingsAnytimeTDRules:
    """
    Implements DraftKings-specific rules for Anytime TD Scorer markets.

    Key Rule: For a player to win an anytime TD bet, they must:
    1. Physically carry the ball across the goal line (rushing TD), OR
    2. Catch the ball in the end zone (receiving TD)

    For Quarterbacks:
    - Passing TDs DO NOT count
    - Only rushing TDs (QB sneak, scramble, designed run) count
    - Receiving TDs on trick plays count (rare but possible)
    """

    @staticmethod
    def calculate_qb_anytime_td_probability(
        rush_td_mean: float,
        rec_td_mean: float = 0.0,
        pass_td_mean: float = 0.0  # Included for clarity but NOT used
    ) -> float:
        """
        Calculate probability of QB scoring an anytime TD.

        CRITICAL: Per DraftKings rules, passing TDs DO NOT count.
        Only rushing and receiving TDs qualify.

        Args:
            rush_td_mean: Expected rushing TDs (from model)
            rec_td_mean: Expected receiving TDs (rare, for trick plays)
            pass_td_mean: NOT USED - passing TDs don't count for anytime TD

        Returns:
            Probability (0.0 to 1.0) that QB scores anytime TD

        Examples:
            >>> # Josh Allen: 0.35 rushing TDs/game expected
            >>> calc_qb_anytime_td_probability(0.35, 0.0, 2.1)
            0.2952  # ~29.5% chance of anytime TD (passing TDs ignored)

            >>> # Mahomes on trick play week
            >>> calc_qb_anytime_td_probability(0.15, 0.05, 2.5)
            0.1813  # Includes both rush + rec chance
        """
        import numpy as np

        # IMPORTANT: Passing TDs are explicitly ignored
        # Only rush and receiving TDs count

        if rush_td_mean == 0.0 and rec_td_mean == 0.0:
            return 0.0

        # P(any TD) = 1 - P(no rushing TD) * P(no receiving TD)
        # Using Poisson distribution: P(0) = exp(-lambda)
        prob_no_rush_td = np.exp(-rush_td_mean) if rush_td_mean > 0 else 1.0
        prob_no_rec_td = np.exp(-rec_td_mean) if rec_td_mean > 0 else 1.0

        # P(any TD) = 1 - P(both fail)
        prob_any_td = 1.0 - (prob_no_rush_td * prob_no_rec_td)

        logger.debug(
            f"QB Anytime TD calculation: "
            f"rush_mean={rush_td_mean:.3f}, rec_mean={rec_td_mean:.3f}, "
            f"pass_mean={pass_td_mean:.3f} (IGNORED), "
            f"prob={prob_any_td:.3f}"
        )

        return prob_any_td

    @staticmethod
    def calculate_rb_anytime_td_probability(
        rush_td_mean: float,
        rec_td_mean: float
    ) -> float:
        """
        Calculate probability of RB scoring an anytime TD.

        RBs can score via:
        - Rushing TDs (most common)
        - Receiving TDs (less common but significant)

        Args:
            rush_td_mean: Expected rushing TDs
            rec_td_mean: Expected receiving TDs

        Returns:
            Probability (0.0 to 1.0) that RB scores anytime TD
        """
        import numpy as np

        if rush_td_mean == 0.0 and rec_td_mean == 0.0:
            return 0.0

        prob_no_rush_td = np.exp(-rush_td_mean) if rush_td_mean > 0 else 1.0
        prob_no_rec_td = np.exp(-rec_td_mean) if rec_td_mean > 0 else 1.0

        prob_any_td = 1.0 - (prob_no_rush_td * prob_no_rec_td)

        return prob_any_td

    @staticmethod
    def calculate_wr_te_anytime_td_probability(
        rec_td_mean: float
    ) -> float:
        """
        Calculate probability of WR/TE scoring an anytime TD.

        WR/TE typically only score via:
        - Receiving TDs (primary method)
        - Rushing TDs (rare, usually on trick plays/end-arounds)

        For simplicity and accuracy, we focus on receiving TDs only
        since rushing TDs for WR/TE are extremely rare.

        Args:
            rec_td_mean: Expected receiving TDs

        Returns:
            Probability (0.0 to 1.0) that WR/TE scores anytime TD
        """
        import numpy as np

        if rec_td_mean == 0.0:
            return 0.0

        # P(any receiving TD) = 1 - P(no receiving TDs)
        prob_no_rec_td = np.exp(-rec_td_mean)
        prob_any_td = 1.0 - prob_no_rec_td

        return prob_any_td


class DraftKingsPropActionRules:
    """
    Implements DraftKings rules for when props are "action" vs "no action".

    Key Rules:
    - Player must be active (on 53-man roster)
    - Player must play at least one snap (offense, defense, or special teams)
    - If player is inactive, all props are voided (refunded)
    - Overtime counts for all player props
    - Stat corrections: Books grade on official stats at end of game
    """

    @staticmethod
    def is_prop_action(
        player_active: bool,
        played_snap: bool
    ) -> bool:
        """
        Determine if a prop bet is "action" per DraftKings rules.

        Args:
            player_active: Is player on active roster?
            played_snap: Did player play at least one snap?

        Returns:
            True if prop is action, False if should be voided
        """
        return player_active and played_snap

    @staticmethod
    def overtime_counts() -> bool:
        """
        Check if overtime counts for player props.

        Per DraftKings rules: YES, overtime always counts.

        Returns:
            True (overtime always counts for player props)
        """
        return True


class DraftKingsQBTDMarkets:
    """
    Explains different QB TD markets on DraftKings.

    Markets:
    1. Anytime TD Scorer: Only rushing + receiving TDs count
    2. Passing TD props (O/U 1.5, 2.5, etc.): Only passing TDs count
    3. Pass+Rush+Rec TDs: All TDs count (less common market)
    """

    MARKET_DEFINITIONS = {
        'player_anytime_td': {
            'name': 'Anytime TD Scorer',
            'qb_scoring_types': ['rushing', 'receiving'],
            'excludes': ['passing'],
            'description': 'QB must physically score (rush/catch), passing TDs do NOT count'
        },
        'player_pass_tds': {
            'name': 'Passing TDs',
            'qb_scoring_types': ['passing'],
            'excludes': ['rushing', 'receiving'],
            'description': 'Only passing TDs count, rushing TDs do NOT count'
        },
        'player_rush_tds': {
            'name': 'Rushing TDs',
            'qb_scoring_types': ['rushing'],
            'excludes': ['passing', 'receiving'],
            'description': 'Only rushing TDs count'
        },
        'player_total_tds': {
            'name': 'Total TDs (Pass+Rush+Rec)',
            'qb_scoring_types': ['passing', 'rushing', 'receiving'],
            'excludes': [],
            'description': 'All touchdown types count (rare market)'
        }
    }

    @classmethod
    def get_market_definition(cls, market: str) -> Optional[Dict]:
        """
        Get the official DraftKings definition for a TD market.

        Args:
            market: Market identifier (e.g., 'player_anytime_td')

        Returns:
            Dictionary with market rules, or None if unknown market
        """
        return cls.MARKET_DEFINITIONS.get(market)

    @classmethod
    def explain_qb_anytime_td_rules(cls) -> str:
        """
        Return a plain-English explanation of QB anytime TD rules.

        Returns:
            Human-readable explanation string
        """
        return """
DraftKings QB Anytime TD Scorer Rules:

✅ COUNTS AS TD:
  - QB rushing touchdown (QB sneak, scramble, designed run)
  - QB receiving touchdown (trick play where QB catches pass)

❌ DOES NOT COUNT:
  - QB throwing a touchdown pass
  - Passing TDs only count for "Passing TD" props, not "Anytime TD"

Example:
  - If Patrick Mahomes throws 3 TD passes but doesn't rush for a TD:
    ❌ Anytime TD bet LOSES
    ✅ "Over 1.5 Passing TDs" bet WINS

  - If Josh Allen rushes for 1 TD and throws 2 TD passes:
    ✅ Anytime TD bet WINS (rushing TD counts)
    ✅ "Over 1.5 Passing TDs" bet WINS (passing TDs count separately)
"""


# Convenience function for NFL QUANT integration
def calculate_anytime_td_probability(
    position: str,
    rush_td_mean: float = 0.0,
    rec_td_mean: float = 0.0,
    pass_td_mean: float = 0.0
) -> float:
    """
    Calculate DraftKings-compliant anytime TD probability for any position.

    This is the main entry point for the NFL QUANT recommendation system.

    Args:
        position: Player position ('QB', 'RB', 'WR', 'TE')
        rush_td_mean: Expected rushing TDs
        rec_td_mean: Expected receiving TDs
        pass_td_mean: Expected passing TDs (IGNORED for anytime TD)

    Returns:
        Probability (0.0 to 1.0) of player scoring anytime TD

    Raises:
        ValueError: If position is not recognized
    """
    rules = DraftKingsAnytimeTDRules()

    if position == 'QB':
        # CRITICAL: Pass TDs don't count for QB anytime TD
        return rules.calculate_qb_anytime_td_probability(
            rush_td_mean=rush_td_mean,
            rec_td_mean=rec_td_mean,
            pass_td_mean=pass_td_mean  # Included for logging but not used
        )

    elif position in ['RB', 'FB']:
        # Fullbacks score like running backs (rushing + receiving TDs)
        return rules.calculate_rb_anytime_td_probability(
            rush_td_mean=rush_td_mean,
            rec_td_mean=rec_td_mean
        )

    elif position in ['WR', 'TE']:
        return rules.calculate_wr_te_anytime_td_probability(
            rec_td_mean=rec_td_mean
        )

    elif position in ['K', 'P']:
        # Kickers/Punters never score TDs in anytime TD scorer markets
        logger.warning(f"Position '{position}' does not score TDs - returning 0.0")
        return 0.0

    elif position in ['DB', 'CB', 'S', 'LB', 'DE', 'DT', 'DL', 'OL', 'OG', 'OT', 'C']:
        # Defensive/Offensive line players may score on fumble returns, but:
        # - This is extremely rare
        # - "Anytime TD Scorer" typically refers to offensive TDs only
        # - Return TDs are usually a separate market
        logger.warning(
            f"Position '{position}' (defensive/line) does not typically score "
            f"offensive TDs - returning 0.0. Return TDs are a separate market."
        )
        return 0.0

    else:
        raise ValueError(
            f"Unknown position '{position}'. "
            f"Supported positions: QB, RB, FB, WR, TE (K, P, defensive return 0.0)"
        )


if __name__ == "__main__":
    # Example usage and validation
    print("DraftKings Anytime TD Rules - Examples\n")
    print("=" * 60)

    # Example 1: Josh Allen (dual-threat QB)
    print("\n1. Josh Allen (dual-threat QB)")
    print("   Expected: 2.1 passing TDs, 0.35 rushing TDs")
    prob = calculate_anytime_td_probability(
        position='QB',
        pass_td_mean=2.1,
        rush_td_mean=0.35,
        rec_td_mean=0.0
    )
    print(f"   Anytime TD Probability: {prob:.1%}")
    print(f"   Note: Passing TDs ignored per DraftKings rules")

    # Example 2: Patrick Mahomes (pocket passer)
    print("\n2. Patrick Mahomes (pocket passer)")
    print("   Expected: 2.5 passing TDs, 0.10 rushing TDs")
    prob = calculate_anytime_td_probability(
        position='QB',
        pass_td_mean=2.5,
        rush_td_mean=0.10,
        rec_td_mean=0.0
    )
    print(f"   Anytime TD Probability: {prob:.1%}")
    print(f"   Note: Very low despite high passing TDs")

    # Example 3: Christian McCaffrey (elite RB)
    print("\n3. Christian McCaffrey (elite RB)")
    print("   Expected: 0.65 rushing TDs, 0.25 receiving TDs")
    prob = calculate_anytime_td_probability(
        position='RB',
        rush_td_mean=0.65,
        rec_td_mean=0.25
    )
    print(f"   Anytime TD Probability: {prob:.1%}")

    # Example 4: Tyreek Hill (WR)
    print("\n4. Tyreek Hill (WR)")
    print("   Expected: 0.55 receiving TDs")
    prob = calculate_anytime_td_probability(
        position='WR',
        rec_td_mean=0.55
    )
    print(f"   Anytime TD Probability: {prob:.1%}")

    print("\n" + "=" * 60)
    print("\nRules Summary:")
    print(DraftKingsQBTDMarkets.explain_qb_anytime_td_rules())
