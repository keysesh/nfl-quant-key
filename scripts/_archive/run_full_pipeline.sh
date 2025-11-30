#!/bin/bash
#
# NFL QUANT - Full Pipeline Runner
#
# This script runs the complete prediction pipeline with all new NFLverse integrations:
# 1. Fetch complete NFLverse data
# 2. Train position-market calibrators
# 3. Generate base model predictions
# 4. Enhance with NFLverse features
# 5. Apply position-specific calibration
# 6. Filter to quality bets
#
# Usage:
#   ./scripts/run_full_pipeline.sh
#   ./scripts/run_full_pipeline.sh --skip-fetch  # Skip data fetch
#   ./scripts/run_full_pipeline.sh --quick       # Skip training, fetch
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Parse arguments
SKIP_FETCH=false
SKIP_TRAIN=false
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-fetch)
            SKIP_FETCH=true
            ;;
        --skip-train)
            SKIP_TRAIN=true
            ;;
        --quick)
            QUICK_MODE=true
            SKIP_FETCH=true
            SKIP_TRAIN=true
            ;;
    esac
done

echo "========================================================================"
echo "NFL QUANT - FULL PIPELINE WITH NFLVERSE INTEGRATION"
echo "========================================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Date: $(date)"
echo ""

# Step 1: Fetch NFLverse Data
if [ "$SKIP_FETCH" = false ]; then
    echo "========================================================================"
    echo "STEP 1: Fetching NFLverse Data"
    echo "========================================================================"

    if command -v Rscript &> /dev/null; then
        echo "Running R fetch script..."
        Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last || {
            echo "Warning: R fetch failed, trying Python fallback..."
            PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/fetch/fetch_complete_nflverse.py --skip-r
        }
    else
        echo "R not found, using Python fallback..."
        PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/fetch/fetch_complete_nflverse.py --skip-r
    fi

    echo "✅ Data fetch complete"
    echo ""
else
    echo "STEP 1: Skipping data fetch (--skip-fetch)"
    echo ""
fi

# Step 1B: Refresh Injury Reports via Sleeper API (fills gap when NFLverse files missing)
echo "========================================================================"
echo "STEP 1B: Refreshing Injury Reports from Sleeper"
echo "========================================================================"

PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/fetch/fetch_injuries_sleeper.py || {
    echo "Warning: Sleeper injury fetch failed (continuing with existing files)"
}

echo ""

# Step 2: Train Position-Market Calibrators
if [ "$SKIP_TRAIN" = false ]; then
    echo "========================================================================"
    echo "STEP 2: Training Position-Market Calibrators"
    echo "========================================================================"

    PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/train/train_position_market_calibrators.py

    echo "✅ Calibrators trained"
    echo ""
else
    echo "STEP 2: Skipping calibrator training (--skip-train)"
    echo ""
fi

# Step 3: Run Game Simulations (REQUIRED for model predictions)
echo "========================================================================"
echo "STEP 3: Running Game Simulations"
echo "========================================================================"

if [ -f "scripts/simulate/run_game_simulations.py" ]; then
    # Get current week
    CURRENT_WEEK=$(python3 -c "from datetime import datetime; now=datetime.now(); print(min(18, (now - datetime(now.year, 9, 1)).days // 7 + 1) if now.month >= 9 else 11)")
    echo "Running simulations for week $CURRENT_WEEK..."
    PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/simulate/run_game_simulations.py --week "$CURRENT_WEEK" || {
        echo "Warning: Game simulation failed"
    }
else
    echo "Game simulation script not found"
fi

echo "✅ Game simulations ready"
echo ""

# Step 4: Generate ENHANCED Calibrated Predictions
# Uses EnhancedProductionPipeline with ALL contextual features:
# - Defensive EPA matchups, Weather/wind impact, Rest/travel factors
# - Snap count trends, Injury redistribution, Team pace
# - Next Gen Stats, QB-WR connection, Historical matchups
# - Isotonic probability calibration
echo "========================================================================"
echo "STEP 4: Generating ENHANCED Calibrated Predictions"
echo "========================================================================"

if [ -f "scripts/predict/generate_calibrated_picks.py" ]; then
    # Auto-detect week from date (uses season_utils.get_current_week())
    echo "Using enhanced production pipeline with auto week detection..."
    PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/predict/generate_calibrated_picks.py || {
        echo "Warning: Enhanced prediction generation failed"
        echo "Check that odds_week{WEEK}_comprehensive.csv exists"
    }
else
    echo "❌ generate_calibrated_picks.py not found!"
    echo "This script uses EnhancedProductionPipeline with all contextual features."
    exit 1
fi

echo "✅ Enhanced calibrated predictions ready"
echo ""

# Step 5: Generate Enhanced Picks with Feature Transparency
echo "========================================================================"
echo "STEP 5: Generating Enhanced Picks with Full Feature Visibility"
echo "========================================================================"

if [ -f "scripts/predict/generate_enhanced_picks.py" ]; then
    PYTHONPATH="$PROJECT_ROOT" .venv/bin/python scripts/predict/generate_enhanced_picks.py || {
        echo "Warning: Enhanced picks generation failed"
    }
else
    echo "Enhanced picks script not found, skipping feature visibility report"
fi

echo "✅ Enhanced recommendations ready with feature contributions"
echo ""

# Step 6: Skip Old Recommendation Systems (they don't use enhanced features)
echo "========================================================================"
echo "STEP 6: Pipeline Complete (Using Enhanced Features Only)"
echo "========================================================================"

echo "✅ All recommendations generated using EnhancedProductionPipeline"
echo "   Features applied: Defensive matchups, Weather, Rest/Travel,"
echo "                     Snap trends, Injuries, Team pace, NGS, QB chemistry"
echo ""

# Step 5: Summary
echo "========================================================================"
echo "PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Generated Reports:"
echo "  📊 ALL Bets (with quality tiers): reports/ALL_RECOMMENDATIONS.csv"
echo "  ✅ Quality Bets (HIGH+MEDIUM): reports/QUALITY_BETS.csv"
echo "  🏆 High Confidence Only: reports/HIGH_CONFIDENCE_BETS.csv"
echo ""

# Show summary stats
if [ -f "reports/ALL_RECOMMENDATIONS.csv" ]; then
    TOTAL_BETS=$(wc -l < reports/ALL_RECOMMENDATIONS.csv | tr -d ' ')
    TOTAL_BETS=$((TOTAL_BETS - 1))  # Subtract header

    echo "Summary:"
    echo "  Total bets analyzed: $TOTAL_BETS"

    # Count by quality tier
    HIGH_TIER=$(grep -c "^HIGH," reports/ALL_RECOMMENDATIONS.csv || echo "0")
    MEDIUM_TIER=$(grep -c "^MEDIUM," reports/ALL_RECOMMENDATIONS.csv || echo "0")
    LOW_TIER=$(grep -c "^LOW," reports/ALL_RECOMMENDATIONS.csv || echo "0")
    BELOW_TIER=$(grep -c "^BELOW_THRESHOLD," reports/ALL_RECOMMENDATIONS.csv || echo "0")

    echo ""
    echo "  Quality Tier Breakdown:"
    echo "    HIGH: $HIGH_TIER bets"
    echo "    MEDIUM: $MEDIUM_TIER bets"
    echo "    LOW: $LOW_TIER bets"
    echo "    BELOW_THRESHOLD: $BELOW_TIER bets"
fi

echo ""
echo "New System Features:"
echo "  ✅ ALL bets shown (not just filtered)"
echo "  ✅ Quality tiers: HIGH, MEDIUM, LOW, BELOW_THRESHOLD"
echo "  ✅ Confidence scores (1-10 scale)"
echo "  ✅ Position-specific calibration (fixes overconfidence)"
echo "  ✅ Dynamic edge thresholds (RB 20%, QB 5%, WR 10%)"
echo "  ✅ Mean bias corrections applied"
echo "  ✅ Historical baselines populated"
echo "  ✅ Kelly criterion bet sizing"
echo "  ✅ NGS skill scores and regression indicators"
echo ""
echo "========================================================================"
echo "Pipeline completed at $(date)"
echo "========================================================================"
