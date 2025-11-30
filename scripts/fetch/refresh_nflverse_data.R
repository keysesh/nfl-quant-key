#!/usr/bin/env Rscript
# NFL QUANT - Refresh NFLverse Data
#
# This script fetches fresh data from NFLverse and saves to parquet files.
# Run after games complete to update stats for bet validation.
#
# Usage:
#   Rscript scripts/fetch/refresh_nflverse_data.R
#
# Or from R console:
#   source("scripts/fetch/refresh_nflverse_data.R")

# Set library path to use local Rlib
.libPaths(c("Rlib", .libPaths()))

suppressPackageStartupMessages({
  library(nflreadr)
  library(arrow)
  library(dplyr)
})

cat("=== NFL QUANT - Refreshing NFLverse Data ===\n\n")

# Create data directory if needed
data_dir <- "data/nflverse"
if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

# 1. Weekly Player Stats (most important for bet validation)
cat("1. Fetching weekly player stats (2024-2025)...\n")
weekly_stats <- load_player_stats(seasons = 2024:2025)
cat(sprintf("   Loaded %d rows\n", nrow(weekly_stats)))

# Check latest week
latest_week <- weekly_stats %>%
  filter(season == 2025) %>%
  pull(week) %>%
  max(na.rm = TRUE)
cat(sprintf("   Latest week in 2025: %d\n", latest_week))

# Save to parquet
write_parquet(weekly_stats, file.path(data_dir, "weekly_stats.parquet"))
cat("   Saved to weekly_stats.parquet\n\n")

# 2. Rosters
cat("2. Fetching rosters (2025)...\n")
rosters <- load_rosters(seasons = 2025)
cat(sprintf("   Loaded %d players\n", nrow(rosters)))
write_parquet(rosters, file.path(data_dir, "rosters_2025.parquet"))
write.csv(rosters, file.path(data_dir, "rosters_2025.csv"), row.names = FALSE)
cat("   Saved to rosters_2025.parquet and rosters_2025.csv\n\n")

# 3. Schedules
cat("3. Fetching schedules (2024-2025)...\n")
schedules <- load_schedules(seasons = 2024:2025)
cat(sprintf("   Loaded %d games\n", nrow(schedules)))
write_parquet(schedules, file.path(data_dir, "schedules_2024_2025.parquet"))
write.csv(schedules, file.path(data_dir, "schedules_2024_2025.csv"), row.names = FALSE)
cat("   Saved to schedules_2024_2025.parquet and schedules_2024_2025.csv\n\n")

# 4. Play-by-play (for micro metrics)
cat("4. Fetching play-by-play (2025)...\n")
pbp <- load_pbp(seasons = 2025)
cat(sprintf("   Loaded %d plays\n", nrow(pbp)))
write_parquet(pbp, file.path(data_dir, "pbp_2025.parquet"))
cat("   Saved to pbp_2025.parquet\n\n")

# Summary
cat("=== Refresh Complete ===\n")
cat(sprintf("Weekly stats: %d rows (through Week %d)\n", nrow(weekly_stats), latest_week))
cat(sprintf("Rosters: %d players\n", nrow(rosters)))
cat(sprintf("Schedules: %d games\n", nrow(schedules)))
cat(sprintf("PBP: %d plays\n", nrow(pbp)))
cat("\nYou can now re-run bet validation:\n")
cat("  .venv/bin/python scripts/tracking/track_bet_results.py --week 12\n")
