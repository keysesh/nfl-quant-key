#!/usr/bin/env python3
"""
Buccaneers Week 15 Team Health Synergy Analysis

This script demonstrates the Team Health Synergy module using the
Tampa Bay Buccaneers as a case study for Week 15, 2025.

Key Players Returning/Healthy:
- Mike Evans (WR1) - Returning from injury
- Chris Godwin (WR2) - Near full health (95%)
- Jalen McMillan/Egbuka (WR3) - Full participation
- O-Line returning to health

This example shows:
1. How individual snap ramp adjustments are calculated
2. How synergy effects compound beyond individual adjustments
3. Cascade effects on teammates (Egbuka benefits from Evans return)
4. Team total adjustments for betting

Usage:
    python scripts/examples/buccaneers_week15_synergy.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from nfl_quant.features.team_synergy import (
    calculate_team_synergy_adjustment,
    generate_synergy_report,
    calculate_team_total_adjustment,
    get_synergy_betting_implications,
    PlayerStatus,
    SynergyCondition,
    CascadeEffect,
    SYNERGY_CONDITIONS,
    CASCADE_DEFINITIONS,
)


def create_buccaneers_roster_week15():
    """
    Create simulated Buccaneers roster status for Week 15.

    Based on typical injury patterns, this shows:
    - Mike Evans returning from multi-week absence
    - Chris Godwin healthy but managed
    - O-line near full strength
    """
    statuses = []

    # =========================================================================
    # QUARTERBACKS
    # =========================================================================
    statuses.append(PlayerStatus(
        player_name="Baker Mayfield",
        player_id="00-0034857",
        position="QB",
        position_rank=1,
        team="TB",
        game_status="Active",
        snap_expectation=1.0,
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    # =========================================================================
    # WIDE RECEIVERS - Key synergy group
    # =========================================================================
    # Mike Evans - RETURNING from injury (key synergy trigger)
    statuses.append(PlayerStatus(
        player_name="Mike Evans",
        player_id="00-0031345",
        position="WR",
        position_rank=1,  # WR1
        team="TB",
        game_status="Active",
        snap_expectation=0.50,  # Pitch count - 50% snaps first game back
        weeks_missed=4,  # Missed 4 weeks
        games_since_return=0,  # First game back
        is_returning=True  # KEY: Triggers synergy calculations
    ))

    # Chris Godwin - Near full health
    statuses.append(PlayerStatus(
        player_name="Chris Godwin",
        player_id="00-0033536",
        position="WR",
        position_rank=2,  # WR2
        team="TB",
        game_status="Active",
        snap_expectation=0.95,  # Near full
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    # Emeka Egbuka / Jalen McMillan - WR3
    statuses.append(PlayerStatus(
        player_name="Jalen McMillan",
        player_id="00-0039123",
        position="WR",
        position_rank=3,  # WR3/Slot
        team="TB",
        game_status="Active",
        snap_expectation=1.0,
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    # =========================================================================
    # OFFENSIVE LINE - Cohesion group
    # =========================================================================
    oline_players = [
        ("Tristan Wirfs", "LT", 1, 1.0, False),
        ("Ben Bredeson", "LG", 1, 0.95, False),  # Minor injury
        ("Robert Hainsey", "C", 1, 1.0, False),
        ("Cody Mauch", "RG", 1, 1.0, False),
        ("Luke Goedeke", "RT", 1, 0.90, True),  # Returning from 2-week absence
    ]

    for name, pos, rank, snap_exp, returning in oline_players:
        statuses.append(PlayerStatus(
            player_name=name,
            player_id=f"oline-{name.lower().replace(' ', '')}",
            position=pos,
            position_rank=rank,
            team="TB",
            game_status="Active" if snap_exp > 0.5 else "Questionable",
            snap_expectation=snap_exp,
            weeks_missed=2 if returning else 0,
            games_since_return=0 if returning else 10,
            is_returning=returning
        ))

    # =========================================================================
    # RUNNING BACKS
    # =========================================================================
    statuses.append(PlayerStatus(
        player_name="Rachaad White",
        player_id="00-0038574",
        position="RB",
        position_rank=1,
        team="TB",
        game_status="Active",
        snap_expectation=1.0,
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    statuses.append(PlayerStatus(
        player_name="Bucky Irving",
        player_id="00-0039456",
        position="RB",
        position_rank=2,
        team="TB",
        game_status="Active",
        snap_expectation=1.0,
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    # =========================================================================
    # TIGHT ENDS
    # =========================================================================
    statuses.append(PlayerStatus(
        player_name="Cade Otton",
        player_id="00-0038012",
        position="TE",
        position_rank=1,
        team="TB",
        game_status="Active",
        snap_expectation=1.0,
        weeks_missed=0,
        games_since_return=0,
        is_returning=False
    ))

    return statuses


def analyze_individual_vs_synergy():
    """
    Compare individual adjustments vs compound synergy effects.

    This demonstrates the core problem the synergy module solves:
    Evans alone + Godwin alone ≠ Evans + Godwin together
    """
    print("\n" + "=" * 70)
    print("INDIVIDUAL VS SYNERGY COMPARISON")
    print("=" * 70)

    # Individual adjustments (current system)
    evans_snap_factor = 0.50  # 50% snaps
    godwin_snap_factor = 0.95  # 95% snaps
    oline_avg_factor = 0.97  # 97% average

    # Simple addition (current approach)
    simple_team_boost = 1.0
    print("\nCURRENT APPROACH (Individual Snap Ramps):")
    print("-" * 50)
    print(f"  Mike Evans: {evans_snap_factor:.0%} snaps → {evans_snap_factor:.2f}x volume")
    print(f"  Chris Godwin: {godwin_snap_factor:.0%} snaps → {godwin_snap_factor:.2f}x volume")
    print(f"  O-Line avg: {oline_avg_factor:.0%} snaps → {oline_avg_factor:.2f}x blocking")
    print(f"  Simple team adjustment: {simple_team_boost:.2f}x (no synergy)")

    # Synergy approach (new system)
    print("\nNEW APPROACH (Synergy Module):")
    print("-" * 50)

    # Create roster
    roster = create_buccaneers_roster_week15()
    returning = [p for p in roster if p.is_returning]

    # Calculate synergy
    synergy = calculate_team_synergy_adjustment(roster, "TB", returning)

    print(f"  Active synergies: {len(synergy.active_synergies)}")
    for syn in synergy.active_synergies:
        print(f"    ✓ {syn['name']}: +{(syn['multiplier']-1)*100:.1f}%")

    print(f"\n  Team multiplier: {synergy.team_multiplier:.3f}x")
    print(f"  Offense multiplier: {synergy.offense_multiplier:.3f}x")

    # The difference
    diff = synergy.offense_multiplier - simple_team_boost
    print(f"\n  SYNERGY DELTA: +{diff*100:.1f}% above simple approach")

    # Player cascades
    if synergy.player_cascades:
        print("\n  PLAYER CASCADE EFFECTS:")
        for player, effects in synergy.player_cascades.items():
            print(f"    {player}:")
            for effect, value in effects.items():
                if 'boost' in effect or 'reduction' in effect:
                    print(f"      {effect}: {value:.2f}x")


def demonstrate_cascade_effects():
    """
    Show how Evans returning helps Godwin and McMillan.
    """
    print("\n" + "=" * 70)
    print("CASCADE EFFECTS DEMONSTRATION")
    print("=" * 70)

    print("\nWhen Mike Evans (WR1) returns:")
    print("-" * 50)

    # Show cascade definitions that apply
    for cascade in CASCADE_DEFINITIONS:
        if cascade.returning_player_position == 'WR1':
            print(f"\n  {cascade.description}")
            print(f"  Affected positions: {', '.join(cascade.affected_positions)}")
            print("  Effects:")
            for effect, value in cascade.effects.items():
                if value > 1:
                    print(f"    • {effect}: +{(value-1)*100:.0f}%")
                elif value < 1:
                    print(f"    • {effect}: {(value-1)*100:.0f}%")
                else:
                    print(f"    • {effect}: {value:.0%}")

    # Specific examples
    print("\n" + "-" * 50)
    print("CONCRETE EXAMPLES:")
    print("-" * 50)

    print("\n  Chris Godwin (WR2):")
    print("    Before Evans return: Bracketed by CB1 + safety help")
    print("    After Evans return:  Single coverage (CB2)")
    print("    Effect: +10% coverage reduction → easier catches")

    print("\n  Jalen McMillan (WR3/Slot):")
    print("    Before Evans return: Underneath defender shades toward him")
    print("    After Evans return:  LBs respect Evans deep threat")
    print("    Effect: +15% coverage reduction, +8% efficiency boost")

    print("\n  Baker Mayfield (QB):")
    print("    Before Evans return: Limited deep options, holds ball")
    print("    After Evans return:  Deep threat opens, faster decisions")
    print("    Effect: +20% deep ball viability, -5% sack rate")


def betting_implications_example():
    """
    Show concrete betting implications from synergy analysis.
    """
    print("\n" + "=" * 70)
    print("BETTING IMPLICATIONS")
    print("=" * 70)

    # Create roster and calculate
    roster = create_buccaneers_roster_week15()
    returning = [p for p in roster if p.is_returning]
    synergy = calculate_team_synergy_adjustment(roster, "TB", returning)

    # Vegas total assumption
    vegas_team_total = 24.5

    print(f"\nVegas implied team total: {vegas_team_total}")

    total_adj = calculate_team_total_adjustment(synergy, vegas_team_total)

    print(f"\nTeam Total Adjustment:")
    print(f"  Raw total:      {total_adj['raw_total']:.1f}")
    print(f"  Synergy mult:   {total_adj['multiplier']:.3f}x")
    print(f"  Adjusted total: {total_adj['adjusted_total']:.1f}")
    print(f"  Delta:          {total_adj['delta']:+.1f} points")

    # Player prop implications
    print("\n" + "-" * 50)
    print("PLAYER PROP IMPLICATIONS:")
    print("-" * 50)

    print("\n  Mike Evans:")
    print("    • Volume: REDUCE projections (50% snaps)")
    print("    • Efficiency: HIGH TD equity (red zone specialist)")
    print("    • Recommendation: Under on yards, consider anytime TD")

    print("\n  Chris Godwin:")
    print("    • Volume: Slight reduction (slot snaps → Evans takes X)")
    print("    • Efficiency: BOOST (less coverage, easier targets)")
    print("    • Recommendation: Lean Over on receptions if line is soft")

    print("\n  Jalen McMillan / Other WR3:")
    print("    • Volume: Reduction (Evans takes targets)")
    print("    • Efficiency: BOOST (+15% coverage reduction)")
    print("    • Net effect: Mixed - efficiency up, volume down")
    print("    • Recommendation: Watch for line overreaction")

    print("\n  Baker Mayfield:")
    print("    • Pass attempts: Similar (game script dependent)")
    print("    • Pass yards: BOOST (deep ball opens up)")
    print("    • Efficiency: Higher yards per attempt")
    print("    • Recommendation: Lean Over on passing yards")


def full_synergy_report():
    """
    Generate the complete synergy report.
    """
    print("\n" + "#" * 70)
    print("#  FULL SYNERGY REPORT: TAMPA BAY BUCCANEERS")
    print("#  Week 15, 2025 Season")
    print("#" * 70)

    # Create roster and calculate
    roster = create_buccaneers_roster_week15()
    returning = [p for p in roster if p.is_returning]
    synergy = calculate_team_synergy_adjustment(roster, "TB", returning)

    # Print full report
    report = generate_synergy_report(synergy)
    print(report)

    # Betting implications
    vegas_total = 24.5
    implications = get_synergy_betting_implications(synergy, vegas_total)

    print("\nBETTING IMPLICATIONS:")
    print("-" * 60)
    for imp in implications:
        print(f"  • {imp}")


def main():
    print("\n" + "=" * 70)
    print("BUCCANEERS WEEK 15 TEAM HEALTH SYNERGY ANALYSIS")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all demonstrations
    analyze_individual_vs_synergy()
    demonstrate_cascade_effects()
    betting_implications_example()
    full_synergy_report()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
