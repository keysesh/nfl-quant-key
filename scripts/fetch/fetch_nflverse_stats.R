#!/usr/bin/env Rscript

# Fetch NFLverse Player Stats for Backtesting
# Uses nflreadr to get actual player statistics for matching with historical odds

library(nflreadr)
library(dplyr)
library(readr)
library(arrow)

# Set output directory
output_dir <- "data/nflverse"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("=== Fetching NFLverse Player Stats ===\n\n")

# Fetch 2024 season player stats
cat("Fetching 2024 season...\n")
stats_2024 <- load_player_stats(seasons = 2024, stat_type = "offense")
cat(sprintf("  2024: %d records, weeks %s\n", nrow(stats_2024), paste(sort(unique(stats_2024$week)), collapse = ", ")))

# Fetch 2025 season player stats
cat("Fetching 2025 season...\n")
stats_2025 <- load_player_stats(seasons = 2025, stat_type = "offense")
cat(sprintf("  2025: %d records, weeks %s\n", nrow(stats_2025), paste(sort(unique(stats_2025$week)), collapse = ", ")))

# Combine
all_stats <- bind_rows(stats_2024, stats_2025)

cat(sprintf("\nTotal records: %d\n", nrow(all_stats)))
cat(sprintf("Unique players: %d\n", length(unique(all_stats$player_id))))

# Select relevant columns for prop matching
relevant_cols <- c(
  "player_id", "player_name", "player_display_name", "position", "position_group",
  "recent_team", "team", "season", "week", "opponent_team",
  # Passing stats
  "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
  "sacks", "sack_yards", "passing_air_yards", "passing_yards_after_catch",
  "passing_first_downs", "passing_epa", "passing_2pt_conversions",
  # Rushing stats
  "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
  "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions",
  # Receiving stats
  "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
  "receiving_fumbles_lost", "receiving_air_yards", "receiving_yards_after_catch",
  "receiving_first_downs", "receiving_epa", "receiving_2pt_conversions",
  "target_share", "air_yards_share", "wopr",
  # Fantasy
  "fantasy_points", "fantasy_points_ppr"
)

# Keep only columns that exist
available_cols <- intersect(relevant_cols, colnames(all_stats))
stats_clean <- all_stats %>% select(all_of(available_cols))

cat(sprintf("\nClean dataset: %d rows, %d columns\n", nrow(stats_clean), ncol(stats_clean)))

# Save as parquet (efficient)
parquet_file <- file.path(output_dir, "player_stats_2024_2025.parquet")
write_parquet(stats_clean, parquet_file)
cat(sprintf("Saved: %s\n", parquet_file))

# Save as CSV (for Python compatibility)
csv_file <- file.path(output_dir, "player_stats_2024_2025.csv")
write_csv(stats_clean, csv_file)
cat(sprintf("Saved: %s\n", csv_file))

# Summary by season and week
cat("\n=== Summary by Season ===\n")
for (season in sort(unique(stats_clean$season))) {
  season_data <- stats_clean %>% filter(season == !!season)
  weeks <- sort(unique(season_data$week))
  cat(sprintf("%d season:\n", season))
  cat(sprintf("  Weeks: %s\n", paste(weeks, collapse = ", ")))
  cat(sprintf("  Total records: %d\n", nrow(season_data)))
  cat(sprintf("  Unique players: %d\n", length(unique(season_data$player_id))))

  # Key stat averages (non-zero)
  avg_pass_yds <- mean(season_data$passing_yards[season_data$passing_yards > 0], na.rm = TRUE)
  avg_rush_yds <- mean(season_data$rushing_yards[season_data$rushing_yards > 0], na.rm = TRUE)
  avg_rec_yds <- mean(season_data$receiving_yards[season_data$receiving_yards > 0], na.rm = TRUE)
  avg_receptions <- mean(season_data$receptions[season_data$receptions > 0], na.rm = TRUE)

  cat(sprintf("  Avg passing yards (non-zero): %.1f\n", avg_pass_yds))
  cat(sprintf("  Avg rushing yards (non-zero): %.1f\n", avg_rush_yds))
  cat(sprintf("  Avg receiving yards (non-zero): %.1f\n", avg_rec_yds))
  cat(sprintf("  Avg receptions (non-zero): %.1f\n", avg_receptions))
  cat("\n")
}

# Also fetch schedule for game_id matching
cat("=== Fetching Schedule Data ===\n")
schedules <- load_schedules(seasons = c(2024, 2025))
cat(sprintf("Schedule records: %d\n", nrow(schedules)))

# Create game_id mapping
schedules <- schedules %>%
  mutate(
    game_date = as.Date(gameday),
    game_id_date = format(game_date, "%Y%m%d"),
    game_id_nflverse = game_id,
    game_id_custom = paste0(game_id_date, "_", away_team, "_", home_team)
  )

schedule_file <- file.path(output_dir, "schedules_2024_2025.csv")
write_csv(schedules, schedule_file)
cat(sprintf("Saved: %s\n", schedule_file))

cat("\n=== DONE ===\n")
