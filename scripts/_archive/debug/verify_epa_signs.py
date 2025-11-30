#!/usr/bin/env python3
"""
EPA Sign Verification Script

Verifies that opponent defensive EPA signs are correct:
- Positive EPA = Defense allows MORE yards/points (WEAK defense)
- Negative EPA = Defense allows FEWER yards/points (STRONG defense)

This is critical for efficiency models - if signs are flipped, model will make
opposite adjustments (predicting LOWER efficiency against weak defenses).
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def verify_epa_sign_convention():
    """
    Verify EPA sign convention using known good/bad defenses.

    Returns:
        bool: True if signs are correct, False if flipped
    """
    logger.info("=" * 80)
    logger.info("EPA SIGN VERIFICATION")
    logger.info("=" * 80)

    # Load PBP data
    pbp_path = Path(__file__).parent.parent.parent / 'data' / 'nflverse' / 'pbp_2025.parquet'

    if not pbp_path.exists():
        logger.error(f"PBP file not found: {pbp_path}")
        return False

    pbp = pd.read_parquet(pbp_path)
    logger.info(f"\nLoaded {len(pbp):,} plays from 2025 season (weeks 1-11)")

    # Calculate defensive EPA for all teams
    teams_rush_epa = {}
    teams_pass_epa = {}

    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING DEFENSIVE EPA FOR ALL TEAMS")
    logger.info("=" * 80)

    for team in pbp['defteam'].dropna().unique():
        # Run defense EPA
        team_run_plays = pbp[(pbp['defteam'] == team) & (pbp['play_type'] == 'run') & (pbp['week'] < 12)]
        if len(team_run_plays) > 0:
            teams_rush_epa[team] = team_run_plays['epa'].mean()

        # Pass defense EPA
        team_pass_plays = pbp[(pbp['defteam'] == team) & (pbp['play_type'] == 'pass') & (pbp['week'] < 12)]
        if len(team_pass_plays) > 0:
            teams_pass_epa[team] = team_pass_plays['epa'].mean()

    # Calculate league averages
    league_run_epa = pbp[(pbp['play_type'] == 'run') & (pbp['week'] < 12)]['epa'].mean()
    league_pass_epa = pbp[(pbp['play_type'] == 'pass') & (pbp['week'] < 12)]['epa'].mean()

    logger.info(f"\nLeague Average Run EPA: {league_run_epa:+.4f}")
    logger.info(f"League Average Pass EPA: {league_pass_epa:+.4f}")

    # Sort teams by EPA
    rush_epa_sorted = sorted(teams_rush_epa.items(), key=lambda x: x[1], reverse=True)
    pass_epa_sorted = sorted(teams_pass_epa.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 80)
    logger.info("WEAKEST RUN DEFENSES (Allow most rushing yards)")
    logger.info("=" * 80)
    logger.info("Expected: POSITIVE EPA (allowing more yards than average)")
    logger.info("\n{:<6} {:>12} {:>12}".format("Team", "Run DEF EPA", "vs League"))
    logger.info("-" * 30)

    for team, epa in rush_epa_sorted[:5]:
        diff = epa - league_run_epa
        logger.info("{:<6} {:>+12.4f} {:>+12.4f}".format(team, epa, diff))

    logger.info("\n" + "=" * 80)
    logger.info("STRONGEST RUN DEFENSES (Allow fewest rushing yards)")
    logger.info("=" * 80)
    logger.info("Expected: NEGATIVE EPA (allowing fewer yards than average)")
    logger.info("\n{:<6} {:>12} {:>12}".format("Team", "Run DEF EPA", "vs League"))
    logger.info("-" * 30)

    for team, epa in rush_epa_sorted[-5:]:
        diff = epa - league_run_epa
        logger.info("{:<6} {:>+12.4f} {:>+12.4f}".format(team, epa, diff))

    logger.info("\n" + "=" * 80)
    logger.info("WEAKEST PASS DEFENSES (Allow most passing yards)")
    logger.info("=" * 80)
    logger.info("Expected: POSITIVE EPA (allowing more yards than average)")
    logger.info("\n{:<6} {:>12} {:>12}".format("Team", "Pass DEF EPA", "vs League"))
    logger.info("-" * 30)

    for team, epa in pass_epa_sorted[:5]:
        diff = epa - league_pass_epa
        logger.info("{:<6} {:>+12.4f} {:>+12.4f}".format(team, epa, diff))

    logger.info("\n" + "=" * 80)
    logger.info("STRONGEST PASS DEFENSES (Allow fewest passing yards)")
    logger.info("=" * 80)
    logger.info("Expected: NEGATIVE EPA (allowing fewer yards than average)")
    logger.info("\n{:<6} {:>12} {:>12}".format("Team", "Pass DEF EPA", "vs League"))
    logger.info("-" * 30)

    for team, epa in pass_epa_sorted[-5:]:
        diff = epa - league_pass_epa
        logger.info("{:<6} {:>+12.4f} {:>+12.4f}".format(team, epa, diff))

    # Verify specific teams from Week 12 analysis
    logger.info("\n" + "=" * 80)
    logger.info("JETS DEFENSE ANALYSIS (Week 12 Opponent)")
    logger.info("=" * 80)

    jets_run_epa = teams_rush_epa.get('NYJ', 0)
    jets_pass_epa = teams_pass_epa.get('NYJ', 0)

    logger.info(f"\nJets Run Defense EPA: {jets_run_epa:+.4f}")
    logger.info(f"League Average: {league_run_epa:+.4f}")
    logger.info(f"Difference: {jets_run_epa - league_run_epa:+.4f}")

    if jets_run_epa > league_run_epa:
        logger.info("✅ CORRECT: Jets WEAK vs run (positive EPA = allow MORE yards)")
        logger.info("   → Efficiency model should INCREASE RB YPC predictions vs Jets")
    else:
        logger.error("❌ WRONG: Jets strong vs run (negative EPA) contradicts reality")
        logger.error("   → EPA signs may be FLIPPED")

    logger.info(f"\nJets Pass Defense EPA: {jets_pass_epa:+.4f}")
    logger.info(f"League Average: {league_pass_epa:+.4f}")
    logger.info(f"Difference: {jets_pass_epa - league_pass_epa:+.4f}")

    if jets_pass_epa > league_pass_epa:
        logger.info("✅ CORRECT: Jets WEAK vs pass (positive EPA = allow MORE yards)")
        logger.info("   → Efficiency model should INCREASE WR/TE Y/R predictions vs Jets")
    else:
        logger.error("❌ WRONG: Jets strong vs pass (negative EPA) contradicts reality")
        logger.error("   → EPA signs may be FLIPPED")

    # Final verification
    logger.info("\n" + "=" * 80)
    logger.info("SIGN CONVENTION VERIFICATION")
    logger.info("=" * 80)

    # Check if weakest defenses have positive EPA
    weak_run_defenses_positive = all(epa > 0 for _, epa in rush_epa_sorted[:3])
    weak_pass_defenses_positive = all(epa > 0 for _, epa in pass_epa_sorted[:3])

    # Check if strongest defenses have negative EPA
    strong_run_defenses_negative = all(epa < 0 for _, epa in rush_epa_sorted[-3:])
    strong_pass_defenses_negative = all(epa < 0 for _, epa in pass_epa_sorted[-3:])

    logger.info(f"\nWeakest 3 run defenses have POSITIVE EPA: {weak_run_defenses_positive}")
    logger.info(f"Strongest 3 run defenses have NEGATIVE EPA: {strong_run_defenses_negative}")
    logger.info(f"Weakest 3 pass defenses have POSITIVE EPA: {weak_pass_defenses_positive}")
    logger.info(f"Strongest 3 pass defenses have NEGATIVE EPA: {strong_pass_defenses_negative}")

    all_correct = (
        weak_run_defenses_positive and
        strong_run_defenses_negative and
        weak_pass_defenses_positive and
        strong_pass_defenses_negative
    )

    if all_correct:
        logger.info("\n✅ EPA SIGNS ARE CORRECT")
        logger.info("   Positive EPA = Weak defense (allows more yards)")
        logger.info("   Negative EPA = Strong defense (allows fewer yards)")
        logger.info("\n   ✓ Efficiency models should INCREASE predictions vs weak defenses")
        logger.info("   ✓ Efficiency models should DECREASE predictions vs strong defenses")
        return True
    else:
        logger.error("\n❌ EPA SIGNS MAY BE FLIPPED")
        logger.error("   Models may be making OPPOSITE adjustments")
        logger.error("   → This would explain conservative efficiency projections")
        return False


def check_model_training_data():
    """
    Check if training data has correct EPA signs.
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL TRAINING DATA VERIFICATION")
    logger.info("=" * 80)

    # Try to load training data if available
    training_path = Path(__file__).parent.parent.parent / 'data' / 'training' / 'efficiency_training.parquet'

    if training_path.exists():
        logger.info(f"\nFound training data: {training_path}")
        training = pd.read_parquet(training_path)

        if 'opp_rush_def_epa' in training.columns:
            logger.info("\n📊 Training Data EPA Statistics:")
            logger.info(f"   opp_rush_def_epa range: [{training['opp_rush_def_epa'].min():.4f}, {training['opp_rush_def_epa'].max():.4f}]")
            logger.info(f"   opp_rush_def_epa mean: {training['opp_rush_def_epa'].mean():+.4f}")

            # Check correlation
            if 'yards_per_carry' in training.columns:
                corr = training[['opp_rush_def_epa', 'yards_per_carry']].corr().iloc[0, 1]
                logger.info(f"\n   Correlation (opp_rush_def_epa vs yards_per_carry): {corr:+.4f}")

                if corr > 0:
                    logger.info("   ✅ POSITIVE correlation: Higher EPA (weak defense) → Higher YPC")
                else:
                    logger.error("   ❌ NEGATIVE correlation: Higher EPA → Lower YPC (WRONG)")
        else:
            logger.warning("   No opp_rush_def_epa column in training data")
    else:
        logger.info("\nNo training data found - skipping correlation check")


if __name__ == '__main__':
    logger.info("Starting EPA sign verification...\n")

    # Run verification
    signs_correct = verify_epa_sign_convention()

    # Check training data
    check_model_training_data()

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    if signs_correct:
        logger.info("\n✅ EPA signs are CORRECT in production data")
        logger.info("   → Efficiency projection bug is likely NOT due to EPA sign flip")
        logger.info("   → Check for other issues:")
        logger.info("      - Model overfitting to training data")
        logger.info("      - Incorrect trailing feature calculations")
        logger.info("      - Extreme regression to mean")
    else:
        logger.error("\n❌ EPA signs appear FLIPPED")
        logger.error("   → This WOULD cause conservative efficiency projections")
        logger.error("   → Models would decrease YPC vs weak defenses (wrong)")
        logger.error("\n   ACTION REQUIRED:")
        logger.error("   1. Check nfl_quant/utils/epa_utils.py")
        logger.error("   2. Verify EPA calculation: opponent_epa = pbp[pbp['defteam'] == team]['epa'].mean()")
        logger.error("   3. May need to INVERT sign: opponent_epa = -opponent_epa")

    logger.info("\n" + "=" * 80)
