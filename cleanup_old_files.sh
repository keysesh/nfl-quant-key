#!/bin/bash
# Clean up old/confusing files

echo "Removing old/confusing files..."

# Old fetch scripts
rm -f scripts/data/fetch_historical_2023_2024.py
rm -f scripts/data/fetch_historical_props_2023_2024.py
rm -f scripts/data/check_and_fetch_historical.py

# Old retrain scripts (keep main one)
rm -f scripts/train/retrain_calibrator_with_historical_context.py
rm -f scripts/train/retrain_calibrator_from_backtest.py
rm -f scripts/train/retrain_calibrator_full_history.py
rm -f scripts/train/retrain_calibrator_expert.py
rm -f scripts/train/retrain_calibrator_data_driven.py

# Old status docs
rm -f HISTORICAL_DATA_STATUS.md

# Old simulation script (has issues, use backtest instead)
# Keep for now but note it's not working
# rm -f scripts/data/simulate_historical_props.py

echo "✅ Cleanup complete!"
echo ""
echo "Files removed:"
echo "  - Old fetch scripts"
echo "  - Old retrain scripts (kept main one)"
echo "  - Old status docs"
