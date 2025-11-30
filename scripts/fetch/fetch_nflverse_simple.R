#!/usr/bin/env Rscript
#' Quick NFLverse Data Fetch (No Extra Dependencies)
#'
#' Simplified version that just uses base R + nflreadr
#'
#' Usage:
#'   Rscript scripts/fetch/fetch_nflverse_simple.R

suppressPackageStartupMessages(library(nflreadr))

# Determine current season
current_year <- as.integer(format(Sys.Date(), "%Y"))
current_month <- as.integer(format(Sys.Date(), "%m"))
current_season <- if (current_month >= 8) current_year else current_year - 1

# Default: current + last season for training
seasons <- c(current_season - 1, current_season)

cat("================================================================================\n")
cat("NFLVERSE DATA FETCH (Native R - Simple)\n")
cat("================================================================================\n")
cat(sprintf("Current NFL Season: %d\n", current_season))
cat(sprintf("Fetching Seasons: %s\n", paste(seasons, collapse = ", ")))
cat("\n")

# Create output directory
dir.create("data/nflverse", recursive = TRUE, showWarnings = FALSE)

# 1. Fetch Play-by-Play Data
cat("1. Fetching Play-by-Play Data...\n")
pbp <- load_pbp(seasons)
cat(sprintf("   Loaded %s plays\n", format(nrow(pbp), big.mark = ",")))
saveRDS(pbp, "data/nflverse/pbp.rds")
write.csv(pbp, "data/nflverse/pbp.csv", row.names = FALSE)
cat("   ✅ Saved: data/nflverse/pbp.rds and pbp.csv\n")

# 2. Fetch Player Stats
cat("\n2. Fetching Player Stats...\n")
player_stats <- load_player_stats(seasons)
cat(sprintf("   Loaded %s player stat records\n", format(nrow(player_stats), big.mark = ",")))
saveRDS(player_stats, "data/nflverse/player_stats.rds")
write.csv(player_stats, "data/nflverse/player_stats.csv", row.names = FALSE)
cat("   ✅ Saved: data/nflverse/player_stats.rds and player_stats.csv\n")

# 3. Fetch Schedules
cat("\n3. Fetching Schedules...\n")
schedules <- load_schedules(seasons)
cat(sprintf("   Loaded %s games\n", format(nrow(schedules), big.mark = ",")))
saveRDS(schedules, "data/nflverse/schedules.rds")
write.csv(schedules, "data/nflverse/schedules.csv", row.names = FALSE)
cat("   ✅ Saved: data/nflverse/schedules.rds and schedules.csv\n")

# 4. Fetch Rosters
cat("\n4. Fetching Rosters...\n")
rosters <- load_rosters(seasons)
cat(sprintf("   Loaded %s roster entries\n", format(nrow(rosters), big.mark = ",")))
saveRDS(rosters, "data/nflverse/rosters.rds")
write.csv(rosters, "data/nflverse/rosters.csv", row.names = FALSE)
cat("   ✅ Saved: data/nflverse/rosters.rds and rosters.csv\n")

# Summary
cat("\n================================================================================\n")
cat("DATA FETCH COMPLETE\n")
cat("================================================================================\n")
cat(sprintf("Seasons: %s\n", paste(seasons, collapse = ", ")))
cat("Output: data/nflverse/\n")
cat("\nPython can read CSV files with:\n")
cat("  import pandas as pd\n")
cat("  pbp = pd.read_csv('data/nflverse/pbp.csv')\n")
cat("\nOr for better performance, install arrow package and use:\n")
cat("  Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last\n")
cat("================================================================================\n")
