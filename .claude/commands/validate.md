Validate a specific player recommendation using comprehensive checks:

1. Snap share verification (from snap_counts.parquet)
2. Historical career stats (mean, max, distribution)
3. Z-score analysis (flag if >3σ from career mean per Framework Rule 8.3)
4. Injury multiplier validation (sample size, cap at 3.0x)
5. Statistical bounds checking using validate_player_projections()

Example usage: /validate John Bates
