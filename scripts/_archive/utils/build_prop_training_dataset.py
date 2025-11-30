#!/usr/bin/env python3
"""
Build a player prop training dataset by combining archived lines with Sleeper outcomes.

Inputs:
  - data/historical/live_archive/player_props_archive.csv (append-only log)
  - data/raw/sleeper_games.json (schedule with week numbers)
  - data/sleeper_stats/stats_week{week}_{season}.csv (weekly player stats)

Output:
  - data/historical/player_prop_training_dataset.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
from pandas.errors import EmptyDataError

# Import player name and team normalization
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.utils.team_names import normalize_team_name

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

MARKET_TO_STAT_COLUMN = {
    "player_pass_yds": "pass_yd",
    "player_pass_tds": "pass_td",
    "player_pass_completions": "pass_cmp",
    "player_pass_attempts": "pass_att",
    "player_pass_interceptions": "pass_int",
    "player_pass_longest_completion": "pass_long",  # may not exist; handled later
    "player_rush_yds": "rush_yd",
    "player_rush_attempts": "rush_att",
    "player_rush_tds": "rush_td",
    "player_rush_longest": "rush_long",
    "player_receptions": "rec",
    "player_reception_yds": "rec_yd",
    "player_reception_tds": "rec_td",
    "player_reception_longest": "rec_long",
}


def load_schedule_map(schedule_path: Path) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str], int]]:
    with open(schedule_path) as f:
        games = json.load(f)

    by_date = {}
    by_teams = {}

    for game in games:
        date = game["date"]
        home = game["home"]
        away = game["away"]
        week = game["week"]
        by_date[(date, home, away)] = week
        by_teams[(home, away)] = week

    return by_date, by_teams


def get_week_for_event(
    commence_time: str,
    home_team: str,
    away_team: str,
    schedule_by_date: Dict[Tuple[str, str, str], int],
    schedule_by_teams: Dict[Tuple[str, str], int],
) -> Optional[int]:
    if not commence_time:
        return None

    commence_date = commence_time.split("T")[0]
    home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team[:3].upper())
    away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team[:3].upper())

    week = schedule_by_date.get((commence_date, home_abbr, away_abbr))
    if week is not None:
        return week

    # Try swapping (some APIs list teams as away/home)
    week = schedule_by_date.get((commence_date, away_abbr, home_abbr))
    if week is not None:
        return week

    # Fallback using team matchup ignoring date
    week = schedule_by_teams.get((home_abbr, away_abbr))
    if week is not None:
        return week

    return schedule_by_teams.get((away_abbr, home_abbr))


def load_week_stats(base_dir: Path, week: int, season: int) -> Optional[pd.DataFrame]:
    stats_file = base_dir / f"stats_week{week}_{season}.csv"
    if not stats_file.exists():
        return None

    try:
        df = pd.read_csv(stats_file)
    except EmptyDataError:
        return None

    if "player_name" not in df.columns:
        return None

    df["player_key"] = df["player_name"].apply(normalize_player_name)
    df["team"] = df["team"].str.upper()
    return df


def compute_actual_value(row: pd.Series, market: str) -> Optional[float]:
    if market == "player_anytime_td":
        rush_td = row.get("rush_td", 0)
        rec_td = row.get("rec_td", 0)
        return float((rush_td or 0) + (rec_td or 0) > 0)

    column = MARKET_TO_STAT_COLUMN.get(market)
    if column is None:
        return None

    if column not in row:
        return None

    value = row.get(column)
    if pd.isna(value):
        return None
    return float(value)


def build_dataset(
    archive_csv: Path,
    stats_dir: Path,
    schedule_path: Path,
    output_path: Path,
    season: int,
) -> pd.DataFrame:
    if not archive_csv.exists():
        raise FileNotFoundError(f"Archive CSV not found: {archive_csv}")

    archive_df = pd.read_csv(archive_csv)
    if archive_df.empty:
        raise ValueError("Archive CSV is empty. Run archive_player_props.py first.")

    # Normalize team names BEFORE week assignment so they match the schedule
    archive_df["home_team"] = archive_df["home_team"].apply(normalize_team_name)
    archive_df["away_team"] = archive_df["away_team"].apply(normalize_team_name)

    schedule_by_date, schedule_by_teams = load_schedule_map(schedule_path)

    archive_df["week"] = archive_df.apply(
        lambda r: get_week_for_event(
            r.get("commence_time"),
            r.get("home_team"),
            r.get("away_team"),
            schedule_by_date,
            schedule_by_teams,
        ),
        axis=1,
    )

    archive_df.dropna(subset=["week"], inplace=True)
    archive_df["week"] = archive_df["week"].astype(int)

    results: List[Dict] = []
    stats_cache: Dict[int, pd.DataFrame] = {}

    for week, week_df in archive_df.groupby("week"):
        stats_df = stats_cache.get(week)
        if stats_df is None:
            stats_df = load_week_stats(stats_dir, week, season)
            if stats_df is not None:
                stats_cache[week] = stats_df
        if stats_df is None:
            continue

        stats_lookup = stats_df.set_index(["player_key", "team"])

        for _, row in week_df.iterrows():
            player = str(row.get("player") or "").strip()
            if not player:
                continue

            market = row.get("market")
            if market not in MARKET_TO_STAT_COLUMN and market != "player_anytime_td":
                continue

            # Use normalize_team_name instead of manual lookup for better consistency
            home_abbr = normalize_team_name(row.get("home_team", ""))
            away_abbr = normalize_team_name(row.get("away_team", ""))

            candidate_teams = {home_abbr, away_abbr}
            player_key = normalize_player_name(player)  # Use normalized name for matching
            stats_row = None
            matched_team = None  # Track which team matched

            for team in candidate_teams:
                key = (player_key, team)
                if key in stats_lookup.index:
                    stats_row = stats_lookup.loc[key]
                    if isinstance(stats_row, pd.DataFrame):
                        stats_row = stats_row.iloc[0]
                    matched_team = team  # Store the team that matched
                    break

            if stats_row is None:
                continue

            actual_value = compute_actual_value(stats_row, market)
            if actual_value is None:
                continue

            line = row.get("line")
            prop_type = row.get("prop_type")

            actual_over = float(actual_value > line) if pd.notna(line) else None
            actual_under = float(actual_value < line) if pd.notna(line) else None

            # Team is part of the multiindex, so get it from matched_team instead of stats_row
            # Position is a regular column, so get it from stats_row
            position_value = stats_row.get("position") if isinstance(stats_row, pd.Series) else stats_row.get("position") if stats_row else None
            
            results.append(
                {
                    "run_timestamp": row.get("run_timestamp"),
                    "event_id": row.get("event_id"),
                    "commence_time": row.get("commence_time"),
                    "week": week,
                    "player": player,
                    "team": matched_team,  # Use the team from our matching logic
                    "position": position_value,
                    "market": market,
                    "prop_type": prop_type,
                    "line": line,
                    "american_price": row.get("american_price"),
                    "actual_value": actual_value,
                    "bet_outcome_over": actual_over,
                    "bet_outcome_under": actual_under,
                }
            )

    if not results:
        print("⚠️  No matched player outcomes for the supplied archive/stats inputs.")
        dataset = pd.DataFrame(columns=[
            "run_timestamp",
            "event_id",
            "commence_time",
            "week",
            "player",
            "team",
            "position",
            "market",
            "prop_type",
            "line",
            "american_price",
            "actual_value",
            "bet_outcome_over",
            "bet_outcome_under",
        ])
    else:
        dataset = pd.DataFrame(results)
        dataset.sort_values(by=["week", "event_id", "player", "market", "run_timestamp"], inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build player prop training dataset from archived lines.")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/historical/live_archive/player_props_archive.csv"),
        help="Path to the archived prop CSV.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("data/sleeper_stats"),
        help="Directory containing weekly Sleeper stat CSVs.",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("data/raw/sleeper_games.json"),
        help="Sleeper schedule JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/historical/player_prop_training_dataset.csv"),
        help="Output CSV path for the merged dataset.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season year for the Sleeper stat files (default: 2025).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        archive_csv=args.archive,
        stats_dir=args.stats_dir,
        schedule_path=args.schedule,
        output_path=args.output,
        season=args.season,
    )
    print(f"✅ Built dataset with {len(dataset)} rows at {args.output}")


if __name__ == "__main__":
    main()
