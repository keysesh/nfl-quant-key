#!/usr/bin/env python3
"""
Test script for the unified NFL QUANT system.

Validates:
1. Unified edge calculation works correctly
2. Kelly criterion is calculated (not empty strings)
3. Confidence tiers are standardized
4. Output schema is consistent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from nfl_quant.core.unified_betting import (
    calculate_kelly_fraction,
    assign_confidence_tier,
    calculate_edge_percentage,
    remove_vig_two_way,
    american_odds_to_implied_prob,
    select_best_side,
    calculate_expected_roi,
)
from nfl_quant.schemas_pkg.unified_output import (
    create_player_prop_recommendation,
    create_game_line_recommendation,
    UnifiedPipelineOutput,
)


def test_edge_calculation():
    """Test that edge calculation is consistent."""
    print("=" * 70)
    print("TEST 1: EDGE CALCULATION CONSISTENCY")
    print("=" * 70)

    # Test case: Model says 65% chance, market says 52.4% (after vig removal)
    model_prob = 0.65
    market_odds_over = -110
    market_odds_under = -110

    # Remove vig
    over_implied = american_odds_to_implied_prob(market_odds_over)  # 0.5238
    under_implied = american_odds_to_implied_prob(market_odds_under)  # 0.5238
    fair_over, fair_under = remove_vig_two_way(over_implied, under_implied)

    print(f"Market odds: Over {market_odds_over}, Under {market_odds_under}")
    print(f"Implied prob (with vig): {over_implied:.4f}")
    print(f"Fair market prob (vig removed): {fair_over:.4f}")
    print(f"Model prob: {model_prob:.4f}")

    edge_pct = calculate_edge_percentage(model_prob, fair_over)
    print(f"Edge = {model_prob:.4f} - {fair_over:.4f} = {edge_pct:.2f}%")

    # Verify: edge should be model_prob - market_prob
    expected_edge = (model_prob - fair_over) * 100
    assert abs(edge_pct - expected_edge) < 0.01, f"Edge calculation mismatch: {edge_pct} vs {expected_edge}"

    print("✅ Edge calculation is correct!")
    print()


def test_kelly_calculation():
    """Test that Kelly criterion is actually calculated."""
    print("=" * 70)
    print("TEST 2: KELLY CRITERION CALCULATION")
    print("=" * 70)

    test_cases = [
        (0.65, -110, "Player prop at -110"),
        (0.58, -110, "Spread bet at -110"),
        (0.70, +150, "Underdog moneyline"),
        (0.80, -200, "Heavy favorite moneyline"),
    ]

    for prob, odds, desc in test_cases:
        kelly = calculate_kelly_fraction(prob, odds, fractional=0.25)
        units = round(kelly * 100, 1)

        print(f"{desc}:")
        print(f"  Prob: {prob:.1%}, Odds: {odds}")
        print(f"  Quarter Kelly: {kelly:.4f} ({units} units)")

        # Kelly should be positive for edge > 0
        expected_edge = prob - american_odds_to_implied_prob(odds)
        if expected_edge > 0:
            assert kelly > 0, f"Kelly should be positive for positive edge"
        else:
            assert kelly == 0, f"Kelly should be 0 for negative edge"

        # Kelly should never be empty string (the old bug)
        assert isinstance(kelly, float), "Kelly must be a float, not a string"

    print("✅ Kelly criterion is properly calculated (not empty strings)!")
    print()


def test_confidence_tiers():
    """Test that confidence tiers are unified."""
    print("=" * 70)
    print("TEST 3: UNIFIED CONFIDENCE TIERS")
    print("=" * 70)

    test_cases = [
        (25.0, 0.75, 'player_prop', 'ELITE'),
        (20.0, 0.72, 'player_prop', 'ELITE'),
        (15.0, 0.80, 'player_prop', 'ELITE'),
        (12.0, 0.68, 'player_prop', 'HIGH'),
        (10.0, 0.65, 'spread', 'HIGH'),
        (6.0, 0.60, 'total', 'STANDARD'),
        (3.0, 0.55, 'spread', 'STANDARD'),
        (2.0, 0.52, 'moneyline', 'LOW'),
    ]

    for edge, prob, bet_type, expected_tier in test_cases:
        tier = assign_confidence_tier(edge, prob, bet_type)
        tier_str = tier.value if hasattr(tier, 'value') else str(tier)

        print(f"Edge: {edge:.1f}%, Prob: {prob:.1%}, Type: {bet_type}")
        print(f"  Expected: {expected_tier}, Got: {tier_str}")

        assert tier_str == expected_tier, f"Tier mismatch: expected {expected_tier}, got {tier_str}"

    # Verify all tiers are in standardized format
    valid_tiers = {'ELITE', 'HIGH', 'STANDARD', 'LOW'}
    for edge, prob, bet_type, _ in test_cases:
        tier = assign_confidence_tier(edge, prob, bet_type)
        tier_str = tier.value if hasattr(tier, 'value') else str(tier)
        assert tier_str in valid_tiers, f"Invalid tier: {tier_str}"

    print("✅ Confidence tiers are unified (ELITE/HIGH/STANDARD/LOW)!")
    print()


def test_unified_output_schema():
    """Test that output schema is consistent for both bet types."""
    print("=" * 70)
    print("TEST 4: UNIFIED OUTPUT SCHEMA")
    print("=" * 70)

    # Create a player prop recommendation
    prop_rec = create_player_prop_recommendation(
        player='Travis Kelce',
        nflverse_name='T.Kelce',
        team='KC',
        position='TE',
        game_id='2025_11_BUF_KC',
        game='BUF @ KC',
        week=11,
        season=2025,
        pick='Over',
        market='player_receptions',
        line=5.5,
        projection=6.8,
        projection_std=2.1,
        model_prob=0.72,
        market_prob=0.50,
        edge_pct=22.0,
        expected_roi=20.0,
        american_odds=-110,
        kelly_fraction=0.0935,
        recommended_units=9.4,
        confidence_tier='ELITE',
        raw_prob=0.70
    )

    # Create a game line recommendation
    gl_rec = create_game_line_recommendation(
        game_id='2025_11_BUF_KC',
        game='BUF @ KC',
        week=11,
        season=2025,
        bet_type='spread',
        team='KC',
        pick='KC -3.5',
        market_line=-3.5,
        model_fair_line=-4.2,
        model_prob=0.58,
        market_prob=0.50,
        edge_pct=8.0,
        expected_roi=7.3,
        american_odds=-110,
        kelly_fraction=0.0375,
        recommended_units=3.8,
        confidence_tier='HIGH'
    )

    # Check both have same core fields
    common_fields = [
        'bet_id', 'bet_type', 'game', 'team', 'pick',
        'model_prob', 'market_prob', 'edge_pct', 'expected_roi',
        'kelly_fraction', 'recommended_units', 'confidence_tier'
    ]

    for field in common_fields:
        assert hasattr(prop_rec, field), f"Player prop missing {field}"
        assert hasattr(gl_rec, field), f"Game line missing {field}"

    print(f"Player Prop: {prop_rec.pick}")
    print(f"  Edge: {prop_rec.edge_pct:.2f}%, Kelly: {prop_rec.kelly_fraction:.4f}")
    print(f"  Tier: {prop_rec.confidence_tier}")

    print(f"\nGame Line: {gl_rec.pick}")
    print(f"  Edge: {gl_rec.edge_pct:.2f}%, Kelly: {gl_rec.kelly_fraction:.4f}")
    print(f"  Tier: {gl_rec.confidence_tier}")

    # Test pipeline output aggregation
    output = UnifiedPipelineOutput(
        season=2025,
        week=11,
        player_props=[prop_rec],
        game_lines=[gl_rec]
    )
    output.compute_summaries()

    print(f"\nPipeline Summary:")
    print(f"  Total Recommendations: {output.total_recommendations}")
    print(f"  ELITE: {output.elite_picks}")
    print(f"  HIGH: {output.high_picks}")
    print(f"  Average Edge: {output.avg_edge_pct:.2f}%")
    print(f"  Total Kelly: {output.total_kelly_allocation:.2%}")

    print("✅ Unified output schema is consistent for all bet types!")
    print()


def test_side_selection():
    """Test that side selection uses market odds correctly."""
    print("=" * 70)
    print("TEST 5: SIDE SELECTION WITH MARKET ODDS")
    print("=" * 70)

    # Test spread selection
    home_cover_prob = 0.58
    home_odds = -110
    away_odds = -110

    result = select_best_side(
        home_cover_prob, 1 - home_cover_prob,
        home_odds, away_odds,
        'KC -3.5', 'BUF +3.5'
    )

    print("Spread Selection:")
    print(f"  Model Home Cover: {home_cover_prob:.1%}")
    print(f"  Selected: {result['pick']}")
    print(f"  Market Prob (vig removed): {result['market_prob']:.4f}")
    print(f"  Edge: {result['edge_pct']:.2f}%")

    # Verify market prob is NOT 0.5 (the old bug assumed 50/50)
    # With -110/-110, fair prob should be exactly 0.5
    assert abs(result['market_prob'] - 0.5) < 0.01, "Fair market prob should be ~0.5 for -110/-110"

    # Test with uneven odds
    over_prob = 0.60
    over_odds = -115
    under_odds = -105

    result2 = select_best_side(
        over_prob, 1 - over_prob,
        over_odds, under_odds,
        'OVER 47.5', 'UNDER 47.5'
    )

    print("\nTotal Selection (uneven odds):")
    print(f"  Model Over Prob: {over_prob:.1%}")
    print(f"  Over odds: {over_odds}, Under odds: {under_odds}")
    print(f"  Selected: {result2['pick']}")
    print(f"  Market Prob: {result2['market_prob']:.4f}")
    print(f"  Edge: {result2['edge_pct']:.2f}%")

    # Verify edge is calculated from actual market odds
    over_implied = american_odds_to_implied_prob(over_odds)
    under_implied = american_odds_to_implied_prob(under_odds)
    fair_over, _ = remove_vig_two_way(over_implied, under_implied)
    expected_edge = (over_prob - fair_over) * 100

    if 'OVER' in result2['pick']:
        assert abs(result2['edge_pct'] - expected_edge) < 0.1, f"Edge mismatch"

    print("✅ Side selection uses actual market odds (not 0.5 baseline)!")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("NFL QUANT UNIFIED SYSTEM VALIDATION")
    print("=" * 70)
    print()

    test_edge_calculation()
    test_kelly_calculation()
    test_confidence_tiers()
    test_unified_output_schema()
    test_side_selection()

    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe unified system is working correctly:")
    print("  ✅ Edge calculation: model_prob - market_prob")
    print("  ✅ Kelly criterion: Properly calculated (not empty)")
    print("  ✅ Confidence tiers: ELITE/HIGH/STANDARD/LOW")
    print("  ✅ Output schema: Unified for all bet types")
    print("  ✅ Side selection: Uses actual market odds")
    print()


if __name__ == "__main__":
    main()
