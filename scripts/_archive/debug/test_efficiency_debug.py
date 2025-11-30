#!/usr/bin/env python3
"""
Test Efficiency Predictor Debug Logging

Generates predictions for Derrick Henry to trace efficiency model behavior.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

logger = logging.getLogger(__name__)

def test_derrick_henry():
    """Test efficiency prediction for Derrick Henry."""
    logger.info("=" * 80)
    logger.info("DERRICK HENRY EFFICIENCY DEBUG TEST")
    logger.info("=" * 80)

    # Load predictors
    logger.info("\nLoading models...")
    usage_predictor, efficiency_predictor = load_predictors()

    # Create simulator
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=100  # Just 100 trials for testing
    )

    # Create PlayerPropInput for Derrick Henry Week 12 vs NYJ
    henry_input = PlayerPropInput(
        player_id="henry_derrick",
        player_name="Derrick Henry",
        position="RB",
        team="BAL",
        opponent="NYJ",
        week=12,
        # Trailing stats (from actual Week 12 predictions)
        trailing_snap_share=0.587,  # Required field
        trailing_carry_share=0.45,  # RB carry share of team rushes
        trailing_yards_per_opportunity=2.61,  # This is what we're testing
        trailing_yards_per_carry=4.72,  # ACTUAL trailing YPC
        trailing_td_rate=0.05,
        trailing_td_rate_rush=0.08,
        # Opponent defense
        opponent_def_epa_vs_position=0.0520,  # Jets weak vs run
        # Game context
        projected_team_total=24.0,  # Required
        projected_opponent_total=20.0,  # Required
        projected_game_script=0.3,  # Required (positive = leading)
        projected_pace=60.0,
        # Team usage projections (required for V3 simulator)
        projected_team_pass_attempts=32.0,
        projected_team_rush_attempts=28.0,
        projected_team_targets=32.0,
    )

    logger.info("\n" + "=" * 80)
    logger.info("INPUT FEATURES")
    logger.info("=" * 80)
    logger.info(f"Player: {henry_input.player_name}")
    logger.info(f"Position: {henry_input.position}")
    logger.info(f"Opponent: {henry_input.opponent}")
    logger.info("\nTrailing Stats:")
    logger.info(f"  - Trailing YPC: {henry_input.trailing_yards_per_carry:.3f}")
    logger.info(f"  - Trailing Carry Share: {henry_input.trailing_carry_share:.3f}")
    logger.info("\nOpponent Defense:")
    logger.info(f"  - Opp Rush DEF EPA: {henry_input.opponent_def_epa_vs_position:+.4f}")
    logger.info("  - Interpretation: POSITIVE = WEAK defense (should INCREASE YPC)")

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING SIMULATION (WITH DEBUG LOGGING)")
    logger.info("=" * 80)
    logger.info("\nWatch for 🔍 EFFICIENCY DEBUG logs below:\n")

    # Run simulation (this will trigger debug logging)
    result = simulator.simulate_player(henry_input)

    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 80)

    if 'rushing_yards' in result:
        rush_yards = result['rushing_yards']
        logger.info(f"\nRushing Yards:")
        logger.info(f"  - Mean: {rush_yards.mean():.2f}")
        logger.info(f"  - Std: {rush_yards.std():.2f}")
        logger.info(f"  - Median: {median(rush_yards):.2f}")
        logger.info(f"  - 25th percentile: {percentile(rush_yards, 25):.2f}")
        logger.info(f"  - 75th percentile: {percentile(rush_yards, 75):.2f}")

        # Calculate implied YPC
        if 'rushing_attempts' in result:
            rush_att = result['rushing_attempts']
            implied_ypc = rush_yards.mean() / rush_att.mean() if rush_att.mean() > 0 else 0
            logger.info(f"\nImplied YPC:")
            logger.info(f"  - Projected Yards: {rush_yards.mean():.2f}")
            logger.info(f"  - Projected Attempts: {rush_att.mean():.2f}")
            logger.info(f"  - Implied YPC: {implied_ypc:.2f}")
            logger.info(f"  - Input Trailing YPC: {henry_input.trailing_yards_per_carry:.2f}")
            logger.info(f"  - Difference: {implied_ypc - henry_input.trailing_yards_per_carry:+.2f}")

    if 'receiving_yards' in result:
        rec_yards = result['receiving_yards']
        logger.info(f"\nReceiving Yards:")
        logger.info(f"  - Mean: {rec_yards.mean():.2f}")

    logger.info("\n" + "=" * 80)


def median(arr):
    import numpy as np
    return np.median(arr)


def percentile(arr, p):
    import numpy as np
    return np.percentile(arr, p)


if __name__ == '__main__':
    test_derrick_henry()
