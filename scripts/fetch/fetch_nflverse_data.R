#!/usr/bin/env Rscript
#' Fetch NFLverse Data Using Native R (Much Faster & More Reliable)
#'
#' This replaces the Python nflreadpy version with native R nflreadr.
#' Benefits:
#' - No cache issues
#' - Faster data loading
#' - More features available
#' - Direct from source
#'
#' Usage:
#'   Rscript scripts/fetch/fetch_nflverse_data.R --seasons 2024 2025
#'   Rscript scripts/fetch/fetch_nflverse_data.R --current-season
#'   Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last

suppressPackageStartupMessages({
  library(nflreadr)
  library(arrow)
  library(dplyr)
  library(optparse)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-s", "--seasons"), type = "character", default = NULL,
              help = "Seasons to fetch (space-separated, e.g., '2024 2025')"),
  make_option(c("-c", "--current-season"), action = "store_true", default = FALSE,
              help = "Fetch current season only"),
  make_option(c("-l", "--current-plus-last"), action = "store_true", default = FALSE,
              help = "Fetch current season + last season (default)"),
  make_option(c("-o", "--output-dir"), type = "character", default = "data/nflverse",
              help = "Output directory [default: data/nflverse]"),
  make_option(c("-f", "--format"), type = "character", default = "parquet",
              help = "Output format: parquet, csv, or rds [default: parquet]")
)

parser <- OptionParser(option_list = option_list,
                      description = "Fetch NFLverse data using native R")
args <- parse_args(parser)

# Determine seasons to fetch
current_year <- as.integer(format(Sys.Date(), "%Y"))
current_month <- as.integer(format(Sys.Date(), "%m"))

# NFL season logic: Aug-Dec = current year, Jan-Jul = previous year
current_season <- if (current_month >= 8) current_year else current_year - 1

if (!is.null(args$seasons)) {
  seasons <- as.integer(strsplit(args$seasons, " ")[[1]])
} else if (args$`current-season`) {
  seasons <- current_season
} else {
  # Default: current + last season for training
  seasons <- c(current_season - 1, current_season)
}

cat("================================================================================\n")
cat("NFLVERSE DATA FETCH (Native R)\n")
cat("================================================================================\n")
cat(sprintf("Current NFL Season: %d\n", current_season))
cat(sprintf("Fetching Seasons: %s\n", paste(seasons, collapse = ", ")))
cat(sprintf("Output Directory: %s\n", args$`output-dir`))
cat(sprintf("Output Format: %s\n", args$format))
cat("\n")

# Create output directory
dir.create(args$`output-dir`, recursive = TRUE, showWarnings = FALSE)

# Function to save data in specified format
save_data <- function(data, filename, format = "parquet") {
  filepath <- file.path(args$`output-dir`, filename)

  if (format == "parquet") {
    write_parquet(data, paste0(filepath, ".parquet"))
    cat(sprintf("✅ Saved: %s.parquet (%s rows)\n", filepath, format(nrow(data), big.mark = ",")))
  } else if (format == "csv") {
    write.csv(data, paste0(filepath, ".csv"), row.names = FALSE)
    cat(sprintf("✅ Saved: %s.csv (%s rows)\n", filepath, format(nrow(data), big.mark = ",")))
  } else if (format == "rds") {
    saveRDS(data, paste0(filepath, ".rds"))
    cat(sprintf("✅ Saved: %s.rds (%s rows)\n", filepath, format(nrow(data), big.mark = ",")))
  }
}

# 1. Fetch Play-by-Play Data
cat("\n1. Fetching Play-by-Play Data...\n")
pbp <- load_pbp(seasons)
cat(sprintf("   Loaded %s plays\n", format(nrow(pbp), big.mark = ",")))
save_data(pbp, "pbp", args$format)

# 2. Fetch Player Stats
cat("\n2. Fetching Player Stats...\n")
player_stats <- load_player_stats(seasons)
cat(sprintf("   Loaded %s player stat records\n", format(nrow(player_stats), big.mark = ",")))
save_data(player_stats, "player_stats", args$format)

# 3. Fetch Schedules
cat("\n3. Fetching Schedules...\n")
schedules <- load_schedules(seasons)
cat(sprintf("   Loaded %s games\n", format(nrow(schedules), big.mark = ",")))
save_data(schedules, "schedules", args$format)

# 4. Fetch Rosters
cat("\n4. Fetching Rosters...\n")
rosters <- load_rosters(seasons)
cat(sprintf("   Loaded %s roster entries\n", format(nrow(rosters), big.mark = ",")))
save_data(rosters, "rosters", args$format)

# 5. Fetch Weekly Stats (more granular than player_stats)
cat("\n5. Fetching Weekly Stats...\n")
weekly_stats <- load_player_stats(seasons, stat_type = "offense")
cat(sprintf("   Loaded %s weekly stat records\n", format(nrow(weekly_stats), big.mark = ",")))
save_data(weekly_stats, "weekly_stats", args$format)

# 6. Fetch Snap Counts (CRITICAL for usage prediction)
cat("\n6. Fetching Snap Counts...\n")
tryCatch({
  snap_counts <- load_snap_counts(seasons)
  cat(sprintf("   Loaded %s snap count records\n", format(nrow(snap_counts), big.mark = ",")))
  save_data(snap_counts, "snap_counts", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load snap counts: %s\n", e$message))
})

# 7. Fetch Next Gen Stats - Passing (QB skill metrics)
cat("\n7. Fetching Next Gen Stats - Passing...\n")
tryCatch({
  ngs_passing <- load_nextgen_stats(seasons, stat_type = "passing")
  cat(sprintf("   Loaded %s NGS passing records\n", format(nrow(ngs_passing), big.mark = ",")))
  save_data(ngs_passing, "ngs_passing", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load NGS passing: %s\n", e$message))
})

# 8. Fetch Next Gen Stats - Receiving (WR/TE separation metrics)
cat("\n8. Fetching Next Gen Stats - Receiving...\n")
tryCatch({
  ngs_receiving <- load_nextgen_stats(seasons, stat_type = "receiving")
  cat(sprintf("   Loaded %s NGS receiving records\n", format(nrow(ngs_receiving), big.mark = ",")))
  save_data(ngs_receiving, "ngs_receiving", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load NGS receiving: %s\n", e$message))
})

# 9. Fetch Next Gen Stats - Rushing (RB efficiency metrics)
cat("\n9. Fetching Next Gen Stats - Rushing...\n")
tryCatch({
  ngs_rushing <- load_nextgen_stats(seasons, stat_type = "rushing")
  cat(sprintf("   Loaded %s NGS rushing records\n", format(nrow(ngs_rushing), big.mark = ",")))
  save_data(ngs_rushing, "ngs_rushing", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load NGS rushing: %s\n", e$message))
})

# 10. Fetch Expected Fantasy Points (regression detection)
cat("\n10. Fetching FF Opportunity Data (Expected Fantasy Points)...\n")
tryCatch({
  ff_opportunity <- load_ff_opportunity(seasons)
  cat(sprintf("   Loaded %s FF opportunity records\n", format(nrow(ff_opportunity), big.mark = ",")))
  save_data(ff_opportunity, "ff_opportunity", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load FF opportunity: %s\n", e$message))
})

# 11. Fetch Injuries (weekly practice/game status)
cat("\n11. Fetching Injury Reports...\n")
tryCatch({
  injuries <- load_injuries(seasons)
  cat(sprintf("   Loaded %s injury records\n", format(nrow(injuries), big.mark = ",")))
  save_data(injuries, "injuries", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load injuries: %s\n", e$message))
})

# 12. Fetch Participation Data (route running, personnel)
cat("\n12. Fetching Participation Data...\n")
tryCatch({
  participation <- load_participation(seasons)
  cat(sprintf("   Loaded %s participation records\n", format(nrow(participation), big.mark = ",")))
  save_data(participation, "participation", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load participation: %s\n", e$message))
})

# 13. Fetch Depth Charts
cat("\n13. Fetching Depth Charts...\n")
tryCatch({
  depth_charts <- load_depth_charts(seasons)
  cat(sprintf("   Loaded %s depth chart records\n", format(nrow(depth_charts), big.mark = ",")))
  save_data(depth_charts, "depth_charts", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load depth charts: %s\n", e$message))
})

# 14. Fetch Combine Data (for rookie projections)
cat("\n14. Fetching Combine Data...\n")
tryCatch({
  combine <- load_combine()
  cat(sprintf("   Loaded %s combine records\n", format(nrow(combine), big.mark = ",")))
  save_data(combine, "combine", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load combine: %s\n", e$message))
})

# Summary
cat("\n================================================================================\n")
cat("DATA FETCH COMPLETE\n")
cat("================================================================================\n")
cat(sprintf("Total Seasons: %d\n", length(seasons)))
cat(sprintf("Output Location: %s/\n", args$`output-dir`))
cat("\n")
cat("Files created:\n")
cat(sprintf("  - pbp.%s (play-by-play)\n", args$format))
cat(sprintf("  - player_stats.%s (aggregated stats)\n", args$format))
cat(sprintf("  - schedules.%s (game schedules)\n", args$format))
cat(sprintf("  - rosters.%s (team rosters)\n", args$format))
cat(sprintf("  - weekly_stats.%s (weekly player stats)\n", args$format))
cat(sprintf("  - snap_counts.%s (usage/snap participation)\n", args$format))
cat(sprintf("  - ngs_passing.%s (QB advanced metrics)\n", args$format))
cat(sprintf("  - ngs_receiving.%s (WR/TE separation metrics)\n", args$format))
cat(sprintf("  - ngs_rushing.%s (RB efficiency metrics)\n", args$format))
cat(sprintf("  - ff_opportunity.%s (expected fantasy points)\n", args$format))
cat(sprintf("  - injuries.%s (injury reports)\n", args$format))
cat(sprintf("  - participation.%s (route participation)\n", args$format))
cat(sprintf("  - depth_charts.%s (team depth charts)\n", args$format))
cat(sprintf("  - combine.%s (NFL combine data)\n", args$format))
cat("\n")
cat("Python can read these files with:\n")
cat("  import pandas as pd\n")
cat(sprintf("  pbp = pd.read_parquet('data/nflverse/pbp.%s')\n", args$format))
cat(sprintf("  ngs_receiving = pd.read_parquet('data/nflverse/ngs_receiving.%s')\n", args$format))
cat(sprintf("  ff_opportunity = pd.read_parquet('data/nflverse/ff_opportunity.%s')\n", args$format))
cat("================================================================================\n")

# 15. Fetch Team Descriptions (logos, colors)
cat("\n15. Fetching Team Descriptions (logos, colors)...\n")
tryCatch({
  teams <- load_teams(current = TRUE)
  cat(sprintf("   Loaded %s team records\n", format(nrow(teams), big.mark = ",")))
  save_data(teams, "teams", args$format)
}, error = function(e) {
  cat(sprintf("   ⚠️  Warning: Could not load teams: %s\n", e$message))
})
