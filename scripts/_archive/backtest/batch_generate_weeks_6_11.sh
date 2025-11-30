#!/bin/bash
# Batch generate game simulations and predictions for Weeks 6-11

export PYTHONPATH="/Users/keyonnesession/Desktop/NFL QUANT:$PYTHONPATH"
PYTHON_BIN="/Users/keyonnesession/Desktop/NFL QUANT/.venv/bin/python"
PROJECT_ROOT="/Users/keyonnesession/Desktop/NFL QUANT"

cd "$PROJECT_ROOT"

for week in 6 7 8 9 10 11; do
    echo ""
    echo "=========================================="
    echo "WEEK $week: Running game simulations"
    echo "=========================================="

    "$PYTHON_BIN" scripts/simulate/run_game_simulations.py --week $week

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "WEEK $week: Generating predictions"
        echo "=========================================="

        "$PYTHON_BIN" scripts/predict/generate_model_predictions.py $week

        if [ $? -eq 0 ]; then
            echo "✅ Week $week complete"
        else
            echo "❌ Week $week predictions failed"
        fi
    else
        echo "❌ Week $week game simulations failed"
    fi
done

echo ""
echo "=========================================="
echo "BATCH GENERATION COMPLETE"
echo "=========================================="
