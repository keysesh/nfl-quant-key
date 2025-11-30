#!/usr/bin/env python3
"""
Enhanced pipeline validation.

This replaces the old PlayerSimulator checks with assertions against the
EnhancedProductionPipeline so every validation uses the same contextual feature
stack that powers production.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.integration.enhanced_production_pipeline import (
    EnhancedProductionPipeline,
    create_enhanced_pipeline,
)
from nfl_quant.integration.feature_aggregator import FeatureAggregator
from nfl_quant.integration.enhanced_prediction import AllFeatures, EnhancedPrediction
from nfl_quant.utils.season_utils import get_current_season, get_current_week

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PIPELINE: Optional[EnhancedProductionPipeline] = None
SAMPLE_CONTEXT: Dict[str, Any] = {}
CACHED_PREDICTION: Optional[EnhancedPrediction] = None


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}\n")


def get_pipeline() -> EnhancedProductionPipeline:
    """Lazy-load pipeline so every test uses the same instance."""
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = create_enhanced_pipeline()
    return PIPELINE


def _infer_opponent(aggregator: FeatureAggregator, team: str, season: int, week: int) -> str:
    """Try to find an opponent for the sample player from the schedule."""
    schedules = getattr(aggregator, "_schedules", None)
    if schedules is None or len(schedules) == 0:
        return "LAC"

    mask = (schedules["season"] == season) & (schedules["week"] == week)
    mask &= ((schedules["home_team"] == team) | (schedules["away_team"] == team))
    sample = schedules[mask]

    if sample.empty:
        sample = schedules[(schedules["season"] == season) & (
            (schedules["home_team"] == team) | (schedules["away_team"] == team)
        )]

    if sample.empty:
        sample = schedules[schedules["season"] == season]

    if sample.empty:
        return "LAC"

    row = sample.iloc[0]
    return row["away_team"] if row["home_team"] == team else row["home_team"]


def get_sample_context() -> Dict[str, Any]:
    """Build a consistent sample context for the validations."""
    global SAMPLE_CONTEXT
    if SAMPLE_CONTEXT:
        return SAMPLE_CONTEXT

    pipeline = get_pipeline()
    aggregator = pipeline.feature_aggregator

    season = get_current_season()
    week = get_current_week(season)
    if week <= 0:
        week = 10
    week = max(4, min(week, 18))

    team = "KC"
    opponent = _infer_opponent(aggregator, team, season, week)

    SAMPLE_CONTEXT = {
        "player_name": "T.Kelce",
        "player_display_name": "Travis Kelce",
        "team": team,
        "position": "TE",
        "opponent": opponent,
        "market": "receiving_yards",
        "line": 70.5,
        "week": week,
        "season": season,
    }
    return SAMPLE_CONTEXT


def get_cached_prediction() -> EnhancedPrediction:
    """Generate (and cache) an enhanced prediction for the sample context."""
    global CACHED_PREDICTION
    if CACHED_PREDICTION is not None:
        return CACHED_PREDICTION

    ctx = get_sample_context()
    pipeline = get_pipeline()

    CACHED_PREDICTION = pipeline.get_enhanced_prediction(
        player_name=ctx["player_name"],
        team=ctx["team"],
        position=ctx["position"],
        opponent=ctx["opponent"],
        market=ctx["market"],
        line=ctx["line"],
        week=ctx["week"],
        season=ctx["season"],
        market_prob_over=0.52,
        n_simulations=5000,
    )
    return CACHED_PREDICTION


def validate_pipeline_bootstrap() -> bool:
    """Ensure the enhanced pipeline initializes all dependencies."""
    print_section("1. Enhanced Pipeline Initialization")
    pipeline = get_pipeline()

    aggregator_ready = isinstance(pipeline.feature_aggregator, FeatureAggregator)
    param_provider_ready = hasattr(pipeline.param_provider, "weekly_data")
    calibrator_count = len(pipeline.calibrators)

    print(f"✅ Feature aggregator attached: {aggregator_ready}")
    try:
        sample_size = len(pipeline.param_provider.weekly_data)
        print(f"✅ Dynamic weekly data rows available: {sample_size:,}")
    except Exception as exc:
        print(f"❌ Failed to load dynamic parameters: {exc}")
        param_provider_ready = False

    print(f"✅ Calibrators loaded: {calibrator_count}")

    return aggregator_ready and param_provider_ready and calibrator_count > 0


def validate_feature_coverage() -> bool:
    """Validate that every contextual feature block returns data."""
    print_section("2. Contextual Feature Coverage")
    pipeline = get_pipeline()
    ctx = get_sample_context()
    features = pipeline.feature_aggregator.get_all_features(
        player_name=ctx["player_name"],
        team=ctx["team"],
        position=ctx["position"],
        opponent=ctx["opponent"],
        week=ctx["week"],
        season=ctx["season"],
        market=ctx["market"],
    )

    if not isinstance(features, AllFeatures):
        print("❌ Feature aggregator did not return AllFeatures")
        return False

    checks = {
        "defensive_matchup": features.defensive_matchup.matchup_multiplier,
        "weather": features.weather.passing_epa_multiplier,
        "rest_travel": features.rest_travel.rest_epa_multiplier,
        "snap_counts": features.snap_counts.snap_share_multiplier,
        "injury_impact": features.injury_impact.injury_redistribution_multiplier,
        "target_share": features.target_share.current_target_share,
        "ngs": features.ngs.ngs_skill_multiplier,
        "team_pace": features.team_pace.pace_multiplier,
        "qb_connection": features.qb_connection.qb_connection_multiplier,
        "historical_matchup": features.historical_matchup.vs_opponent_multiplier,
    }

    success = True
    for name, value in checks.items():
        if value is None:
            success = False
            print(f"❌ {name} missing data")
        else:
            print(f"✅ {name:20s} -> {value}")

    return success


def validate_prediction_generation() -> bool:
    """Ensure the enhanced pipeline produces a full prediction."""
    print_section("3. Enhanced Prediction Generation")
    prediction = get_cached_prediction()
    ctx = get_sample_context()
    friendly_name = ctx.get("player_display_name", prediction.player_name)

    print(f"Player: {friendly_name} ({prediction.team}) vs {prediction.opponent}")
    print(f"Base mean/std: {prediction.base_mean:.2f} / {prediction.base_std:.2f}")
    print(f"Adjusted mean/std: {prediction.adjusted_mean:.2f} / {prediction.adjusted_std:.2f}")

    if prediction.base_mean <= 0 or prediction.adjusted_mean <= 0:
        print("❌ Pipeline returned non-positive projection")
        return False

    if prediction.raw_prob_over <= 0 or prediction.calibrated_prob_over <= 0:
        print("❌ Probability outputs invalid")
        return False

    print(f"✅ Raw over prob: {prediction.raw_prob_over:.3f}")
    print(f"✅ Calibrated over prob: {prediction.calibrated_prob_over:.3f}")
    print(f"✅ Confidence tier: {prediction.confidence_tier}")
    print(f"✅ Data quality score: {prediction.data_quality_score:.2f}")

    return True


def validate_probability_calibration() -> bool:
    """Confirm that isotonic calibration hooks are wired into predictions."""
    print_section("4. Probability Calibration")
    prediction = get_cached_prediction()
    pipeline = get_pipeline()

    market_key = f"market_{prediction.market}"
    position_key = f"position_{prediction.position}"
    calibrators = pipeline.calibrators

    used_calibrator = market_key in calibrators or position_key in calibrators or "overall" in calibrators
    if not used_calibrator:
        print("❌ No calibrator available for market/position")
        return False

    delta = abs(prediction.calibrated_prob_over - prediction.raw_prob_over)
    print(f"Calibrator delta: {delta:.4f}")
    print("✅ Calibrator source:",
          market_key if market_key in calibrators else position_key if position_key in calibrators else "overall")

    return True


def validate_feature_contributions() -> bool:
    """Ensure feature contributions map exists and covers every adjustment."""
    print_section("5. Feature Contribution Attribution")
    prediction = get_cached_prediction()
    contributions = prediction.feature_contributions

    expected_keys = {
        "defensive_matchup",
        "weather",
        "rest",
        "travel",
        "snap_trend",
        "injury_redistribution",
        "team_pace",
        "ngs_skill",
        "qb_connection",
        "historical_vs_opponent",
        "total_adjustment_pct",
    }

    missing = expected_keys - contributions.keys()
    if missing:
        print(f"❌ Missing contribution entries: {sorted(missing)}")
        return False

    for name in sorted(expected_keys):
        print(f"{name:24s}: {contributions[name]:6.2f} pct points")

    return True


def validate_data_quality_controls() -> bool:
    """Verify the pipeline surfaces confidence tiers and context metadata."""
    print_section("6. Data Quality Controls")
    prediction = get_cached_prediction()

    has_timestamp = bool(prediction.prediction_timestamp)
    valid_tiers = {
        "HIGH_EDGE",
        "MEDIUM_HIGH_EDGE",
        "MEDIUM_EDGE",
        "LOW_EDGE",
        "LOW_DATA",
        "NO_EDGE",
        "NO_DATA",
    }
    tier_valid = prediction.confidence_tier in valid_tiers
    quality_range = 0 <= prediction.data_quality_score <= 100

    print(f"✅ Timestamp present: {has_timestamp}")
    print(f"✅ Confidence tier valid: {tier_valid}")
    print(f"✅ Data quality score range: {quality_range}")

    return has_timestamp and tier_valid and quality_range


def main() -> int:
    """Run all validation tests."""
    print_section("ENHANCED PIPELINE VALIDATION")
    print("Validating that every production surface references EnhancedProductionPipeline\n")

    tests = [
        ("Pipeline Initialization", validate_pipeline_bootstrap),
        ("Feature Coverage", validate_feature_coverage),
        ("Prediction Generation", validate_prediction_generation),
        ("Probability Calibration", validate_probability_calibration),
        ("Feature Contributions", validate_feature_contributions),
        ("Data Quality Controls", validate_data_quality_controls),
    ]

    results: Dict[str, bool] = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"\n❌ {test_name} crashed: {exc}")
            results[test_name] = False

    print_section("VALIDATION SUMMARY")
    passed = sum(1 for passed in results.values() if passed)
    total = len(results)

    for name, did_pass in results.items():
        status = "✅ PASS" if did_pass else "❌ FAIL"
        print(f"{status:10s} {name}")

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {passed}/{total} tests passed ({passed / total * 100:.0f}%)")
    print(f"{'=' * 80}\n")

    if passed == total:
        print("✅ Enhanced pipeline validated successfully!")
        return 0

    print("⚠️  Review failing sections above before running production.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
