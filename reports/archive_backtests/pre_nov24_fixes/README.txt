Archived on Mon Nov 24 16:20:23 EST 2025: Pre-Nov 24, 2025 backtests (before calibration-aware models and bug fixes)

Archived files:
- backtest_bet_log_20251118_*.csv - Nov 18 backtest runs before variance/injury fixes
- backtest_summary_20251118_*.csv - Summary files from same runs
- BACKTEST_HYBRID_CALIBRATION.csv - Nov 2 hybrid calibration test (pre-shrinkage)
- backtest_bet_log_NO_DNP.csv - Nov 18 DNP filtering test

These backtests were run BEFORE:
1. Variance calculation fix (was returning near-zero std values)
2. Injury schema integration (OUT players weren't being excluded)
3. Calibrated model training (usage_*_calibrated.joblib)
4. 30% shrinkage calibration implementation

Results from these runs are not comparable to post-Nov 24 backtests.
