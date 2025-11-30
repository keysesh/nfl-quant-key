#!/usr/bin/env python3
"""
Regime Detection Integration - Example Usage

This script demonstrates how to use the regime-aware trailing stats extractor.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def example_basic_usage():
    """Example 1: Basic usage - compare standard vs regime extractors."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    from nfl_quant.features.trailing_stats import TrailingStatsExtractor
    from nfl_quant.regime.integration import RegimeAwareTrailingStats

    # Initialize both extractors
    standard_extractor = TrailingStatsExtractor()
    regime_extractor = RegimeAwareTrailingStats(enable_regime_detection=True)

    # Example player
    player = "J.Allen"
    position = "QB"
    week = 9

    print(f"\nGetting trailing stats for {player} ({position}) in Week {week}")
    print("-" * 80)

    # Standard 4-week stats
    print("\n1. Standard 4-week extractor:")
    try:
        standard_stats = standard_extractor.get_trailing_stats(player, position, week)
        print(f"   Trailing attempts: {standard_stats['trailing_attempts']:.1f}")
        print(f"   Window: Last 4 weeks (fixed)")
    except Exception as e:
        print(f"   Error: {e}")

    # Regime-aware stats
    print("\n2. Regime-aware extractor:")
    try:
        regime_stats = regime_extractor.get_trailing_stats(player, position, week)
        print(f"   Trailing attempts: {regime_stats['trailing_attempts']:.1f}")
        print(f"   Window: {regime_stats.get('window_weeks', 4)} weeks (dynamic)")

        if regime_stats.get('regime_detected'):
            print(f"   ✨ Regime detected: {regime_stats['regime_type']}")
            print(f"   Confidence: {regime_stats['regime_confidence']:.1%}")
            print(f"   Start week: {regime_stats['regime_start_week']}")
        else:
            print(f"   No regime detected (using standard 4-week window)")
    except Exception as e:
        print(f"   Error: {e}")


def example_feature_flag():
    """Example 2: Using the feature flag."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Feature Flag Usage")
    print("=" * 80)

    import os
    from nfl_quant.features.trailing_stats import get_trailing_stats_extractor

    # Disable regime detection
    os.environ['ENABLE_REGIME_DETECTION'] = '0'
    print("\n1. With ENABLE_REGIME_DETECTION=0:")
    extractor = get_trailing_stats_extractor()
    print(f"   Extractor type: {type(extractor).__name__}")

    # Enable regime detection
    os.environ['ENABLE_REGIME_DETECTION'] = '1'
    print("\n2. With ENABLE_REGIME_DETECTION=1:")
    # Note: Need to reload or create new instance
    from nfl_quant.regime.integration import get_regime_aware_extractor
    extractor = get_regime_aware_extractor(enable_regime=True)
    print(f"   Extractor type: {type(extractor).__name__}")

    # Reset
    os.environ['ENABLE_REGIME_DETECTION'] = '0'


def example_batch_processing():
    """Example 3: Processing multiple players."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 80)

    from nfl_quant.regime.integration import RegimeAwareTrailingStats

    # Initialize extractor
    extractor = RegimeAwareTrailingStats(enable_regime_detection=True)

    # Example players
    players = [
        ("J.Allen", "QB"),
        ("D.Henry", "RB"),
        ("S.Diggs", "WR"),
    ]

    week = 9

    print(f"\nProcessing {len(players)} players for Week {week}:")
    print("-" * 80)

    for player_name, position in players:
        try:
            stats = extractor.get_trailing_stats(player_name, position, week)

            regime_flag = "🔥" if stats.get('regime_detected') else "  "
            print(f"\n{regime_flag} {player_name} ({position})")
            print(f"   Attempts: {stats['trailing_attempts']:.1f}")
            print(f"   Window: {stats.get('window_weeks', 4)} weeks")

            if stats.get('regime_detected'):
                print(f"   Regime: {stats['regime_type']}")

        except Exception as e:
            print(f"\n   {player_name}: Error - {e}")


def example_cli_usage():
    """Example 4: CLI usage examples."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: CLI Usage")
    print("=" * 80)

    print("""
1. Standard mode (no regime detection):

   .venv/bin/python scripts/predict/generate_model_predictions.py 9

2. Enable regime detection:

   .venv/bin/python scripts/predict/generate_model_predictions.py 9 --enable-regime

3. Run validation:

   .venv/bin/python scripts/regime/validate_regime_integration.py --week 9

4. Compare predictions:

   # Generate standard predictions
   .venv/bin/python scripts/predict/generate_model_predictions.py 9
   mv data/model_predictions_week9.csv data/predictions_standard.csv

   # Generate regime predictions
   .venv/bin/python scripts/predict/generate_model_predictions.py 9 --enable-regime
   mv data/model_predictions_week9.csv data/predictions_regime.csv

   # Compare
   python scripts/regime/compare_predictions.py predictions_standard.csv predictions_regime.csv
    """)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "REGIME DETECTION - EXAMPLE USAGE" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")

    examples = [
        example_basic_usage,
        example_feature_flag,
        example_batch_processing,
        example_cli_usage,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ Examples complete!")
    print("=" * 80)
    print("\nFor more information:")
    print("  - README: nfl_quant/regime/README.md")
    print("  - Integration Guide: REGIME_INTEGRATION_COMPLETE.md")
    print("  - Master Plan: INTEGRATION_MASTER_PLAN.md")
    print()


if __name__ == '__main__':
    main()
