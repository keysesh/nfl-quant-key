#!/usr/bin/env Rscript
#' Validate Week Picks Using Native R (Much Faster & Cleaner)
#'
#' This R version is faster and more reliable than the Python equivalent.
#' Benefits:
#' - Native nflverse data access
#' - dplyr for clean data manipulation
#' - Better statistical analysis tools
#' - nflplotR for visualization
#'
#' Usage:
#'   Rscript scripts/backtest/validate_week_r.R --week 10 --season 2024
#'   Rscript scripts/backtest/validate_week_r.R --week 11  # Auto-detects 2025

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(optparse)
  library(lubridate)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-w", "--week"), type = "integer",
              help = "NFL week number"),
  make_option(c("-s", "--season"), type = "integer", default = NULL,
              help = "NFL season year (auto-detects if not provided)"),
  make_option(c("-p", "--picks-file"), type = "character", default = NULL,
              help = "Path to picks CSV file (default: reports/all_picks_ranked_weekN.csv)")
)

parser <- OptionParser(option_list = option_list,
                      description = "Validate week picks against actual results")
args <- parse_args(parser)

# Manual validation for required arguments
if (is.null(args$week)) {
  stop("Error: --week is required")
}

# Season detection logic (matches Python version)
current_year <- year(Sys.Date())
current_month <- month(Sys.Date())
current_season <- if (current_month >= 8) current_year else current_year - 1

if (is.null(args$season)) {
  args$season <- current_season
  cat(sprintf("ℹ️  No season specified, inferred season %d for week %d\n",
              args$season, args$week))
}

# Default picks file location
if (is.null(args$`picks-file`)) {
  args$`picks-file` <- sprintf("reports/all_picks_ranked_week%d.csv", args$week)
}

cat("================================================================================\n")
cat(sprintf("WEEK %d VALIDATION (%d SEASON)\n", args$week, args$season))
cat("================================================================================\n\n")

# 1. Load picks
cat(sprintf("Loading picks from: %s\n", args$`picks-file`))
if (!file.exists(args$`picks-file`)) {
  stop(sprintf("❌ ERROR: Picks file not found: %s", args$`picks-file`))
}

picks <- read_csv(args$`picks-file`, show_col_types = FALSE)
cat(sprintf("Total Picks Generated: %d\n", nrow(picks)))

# Filter to top recommendations (>5% edge)
top_picks <- picks %>% filter(edge_pct > 5.0)
cat(sprintf("Top Recommendations (>5.0%% edge): %d\n\n", nrow(top_picks)))

# 2. Validate picks season against data season
cat("Validating season consistency...\n")

# Load schedule to verify games exist in the specified season
schedule <- load_schedules(args$season)
week_schedule <- schedule %>% filter(week == args$week)

if (nrow(week_schedule) == 0) {
  cat(sprintf("⚠️  WARNING: No games found in schedule for %d season week %d\n",
              args$season, args$week))
}

# Extract unique games from picks
picks_games <- picks %>%
  filter(!is.na(game) & game != "") %>%
  select(game) %>%
  distinct() %>%
  pull(game)

# Validate each game exists in the schedule
invalid_games <- c()
bye_teams <- c()

for (game_str in picks_games) {
  if (grepl("@", game_str)) {
    parts <- strsplit(game_str, " @ ")[[1]]
    away_team <- trimws(parts[1])
    home_team <- trimws(parts[2])

    # Check if this game exists in the schedule
    game_exists <- week_schedule %>%
      filter(away_team == !!away_team & home_team == !!home_team) %>%
      nrow() > 0

    if (!game_exists) {
      invalid_games <- c(invalid_games, game_str)

      # Check if teams are on bye
      teams_playing <- unique(c(week_schedule$home_team, week_schedule$away_team))
      if (!(away_team %in% teams_playing)) {
        bye_teams <- c(bye_teams, away_team)
      }
      if (!(home_team %in% teams_playing)) {
        bye_teams <- c(bye_teams, home_team)
      }
    }
  }
}

if (length(invalid_games) > 0) {
  cat(sprintf("⚠️  WARNING: %d game(s) in picks file don't exist in %d season week %d:\n",
              length(invalid_games), args$season, args$week))
  for (game in invalid_games) {
    cat(sprintf("   - %s\n", game))
  }

  if (length(bye_teams) > 0) {
    cat(sprintf("   Teams on BYE week: %s\n", paste(unique(bye_teams), collapse=", ")))
  }

  # Suggest correct season
  cat(sprintf("\n💡 HINT: If picks were generated for a different season, use:\n"))
  cat(sprintf("   Rscript scripts/backtest/validate_week_r.R --week %d --season YYYY\n\n",
              args$week))
}

cat(sprintf("✓ Season validation complete\n\n"))

# 3. Load actual player stats
cat(sprintf("Loading Week %d player stats for %d season...\n", args$week, args$season))

# Load play-by-play data for the week
pbp <- load_pbp(args$season) %>%
  filter(week == args$week)

if (nrow(pbp) == 0) {
  stop(sprintf("❌ ERROR: No play-by-play data found for Week %d, Season %d",
               args$week, args$season))
}

cat(sprintf("Loaded %s plays from Week %d\n", format(nrow(pbp), big.mark = ","), args$week))

# 3. Calculate player stats from PBP
cat("\nCalculating player stats...\n")

# Passing stats
passing <- pbp %>%
  filter(pass_attempt == 1) %>%
  group_by(passer_player_id, passer_player_name, posteam) %>%
  summarise(
    passing_yards = sum(passing_yards, na.rm = TRUE),
    passing_tds = sum(pass_touchdown, na.rm = TRUE),
    interceptions = sum(interception, na.rm = TRUE),
    completions = sum(complete_pass, na.rm = TRUE),
    attempts = n(),
    .groups = "drop"
  ) %>%
  rename(player_id = passer_player_id,
         player_name = passer_player_name,
         team = posteam)

# Rushing stats
rushing <- pbp %>%
  filter(rush_attempt == 1) %>%
  group_by(rusher_player_id, rusher_player_name, posteam) %>%
  summarise(
    rushing_yards = sum(rushing_yards, na.rm = TRUE),
    rushing_tds = sum(rush_touchdown, na.rm = TRUE),
    rushing_attempts = n(),
    .groups = "drop"
  ) %>%
  rename(player_id = rusher_player_id,
         player_name = rusher_player_name,
         team = posteam)

# Receiving stats
receiving <- pbp %>%
  filter(pass_attempt == 1) %>%
  group_by(receiver_player_id, receiver_player_name, posteam) %>%
  summarise(
    receiving_yards = sum(receiving_yards, na.rm = TRUE),
    receiving_tds = sum(pass_touchdown, na.rm = TRUE),
    receptions = sum(complete_pass, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(player_id = receiver_player_id,
         player_name = receiver_player_name,
         team = posteam)

# Combine all stats
all_stats <- passing %>%
  full_join(rushing, by = c("player_id", "player_name", "team")) %>%
  full_join(receiving, by = c("player_id", "player_name", "team")) %>%
  mutate(across(where(is.numeric), ~replace_na(., 0)))

cat(sprintf("Player stats calculated: %d players\n", nrow(all_stats)))

# 4. Validate picks against actual stats
cat("\nValidating picks...\n")

# Map market types to stat columns
get_actual_value <- function(player_name_arg, game_arg, market_arg) {
  # Extract last name for matching (handles "Josh Jacobs" -> "Jacobs")
  last_name <- sub("^.* ", "", player_name_arg)

  # Parse teams from game string (e.g., "PHI @ GB" -> c("PHI", "GB"))
  teams <- if (!is.na(game_arg) && game_arg != "") {
    strsplit(game_arg, " @ ")[[1]]
  } else {
    c()
  }

  # Search by last name AND team (nflreadr uses abbreviated names like "J.Jacobs")
  player_row <- all_stats %>%
    filter(grepl(last_name, .data$player_name, ignore.case = TRUE) &
           (length(teams) == 0 | .data$team %in% teams))

  if (nrow(player_row) == 0) return(NA)

  player_row <- player_row[1,]  # Take first match

  value <- case_when(
    market_arg == "player_pass_yds" ~ player_row$passing_yards,
    market_arg == "player_pass_tds" ~ player_row$passing_tds,
    market_arg == "player_rush_yds" ~ player_row$rushing_yards,
    market_arg == "player_rush_tds" ~ player_row$rushing_tds,
    market_arg == "player_reception_yds" ~ player_row$receiving_yards,
    market_arg == "player_receptions" ~ player_row$receptions,
    TRUE ~ NA_real_
  )

  return(as.numeric(value))
}

# Evaluate each pick
results <- top_picks %>%
  rowwise() %>%
  mutate(
    # Extract "Over" or "Under" from the pick column
    pick_side = ifelse(grepl("Over", pick), "Over",
                      ifelse(grepl("Under", pick), "Under", NA_character_)),
    actual_value = get_actual_value(player, game, market),
    found_stats = !is.na(actual_value),
    won = case_when(
      !found_stats ~ NA,
      pick_side == "Over" ~ actual_value > line,
      pick_side == "Under" ~ actual_value < line,
      TRUE ~ NA
    ),
    wager = 100,  # Standard $100 wager
    profit = case_when(
      is.na(won) ~ 0,
      won ~ wager * (1 / (abs(odds) / 100)),
      !won ~ -wager,
      TRUE ~ 0
    )
  ) %>%
  ungroup()

# 5. Print results
cat("\n")
cat("================================================================================\n")
cat("RESULTS\n")
cat("================================================================================\n\n")

evaluated <- results %>% filter(found_stats)
no_data <- results %>% filter(!found_stats)

cat(sprintf("📊 Bet Evaluation:\n"))
cat(sprintf("   Total Recommendations: %d\n", nrow(results)))
cat(sprintf("   Evaluated: %d bets\n", nrow(evaluated)))
cat(sprintf("   No Data Available: %d bets\n\n", nrow(no_data)))

if (nrow(evaluated) > 0) {
  wins <- sum(evaluated$won, na.rm = TRUE)
  losses <- sum(!evaluated$won, na.rm = TRUE)
  total_wagered <- sum(evaluated$wager)
  total_profit <- sum(evaluated$profit)

  cat(sprintf("💰 Performance:\n"))
  cat(sprintf("   Wins: %d\n", wins))
  cat(sprintf("   Losses: %d\n", losses))
  cat(sprintf("   Win Rate: %.1f%%\n", 100 * wins / (wins + losses)))
  cat(sprintf("   Total Wagered: $%s\n", format(total_wagered, big.mark = ",")))
  cat(sprintf("   Total Profit: $%+.2f\n", total_profit))
  cat(sprintf("   ROI: %+.1f%%\n", 100 * total_profit / total_wagered))

  # Top 3 picks
  cat(sprintf("\n🏆 Top 3 Picks:\n"))
  top3 <- evaluated %>% arrange(desc(profit)) %>% head(3)
  for (i in 1:min(3, nrow(top3))) {
    row <- top3[i,]
    cat(sprintf("   %d. %s: %s %.1f → $%+.2f\n",
                i, row$player, row$pick, row$line, row$profit))
  }

  # Bottom 3 picks
  cat(sprintf("\n💸 Bottom 3 Picks:\n"))
  bottom3 <- evaluated %>% arrange(profit) %>% head(3)
  for (i in 1:min(3, nrow(bottom3))) {
    row <- bottom3[i,]
    cat(sprintf("   %d. %s: %s %.1f → $%+.2f\n",
                i, row$player, row$pick, row$line, row$profit))
  }
}

# 6. Save results
output_file <- sprintf("reports/%d_WEEK%d_BACKTEST_COMPLETE.csv", args$season, args$week)
write_csv(results, output_file)
cat(sprintf("\n✅ Detailed results saved to: %s\n", output_file))

cat("\n================================================================================\n")
cat("VALIDATION COMPLETE\n")
cat("================================================================================\n")
