"""
Fetch historical NFL player prop odds from The Odds API.

Uses the `/v4/historical/...` endpoints to snapshot archived player prop lines
for one or more events and persist them to disk.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Supported NFL player prop markets for historical data.
# Note: Only these markets have historical data available in The Odds API.
# Other markets (completions, attempts, longest, tackles, kicking, etc.) are
# available for live odds but not in historical snapshots.
NFL_PLAYER_PROP_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
    "player_1st_td",
]

# Extended markets available for live fetching but not historical
NFL_PLAYER_PROP_MARKETS_EXTENDED = [
    "player_pass_completions",
    "player_pass_attempts",
    "player_pass_interceptions",
    "player_pass_longest_completion",
    "player_rush_attempts",
    "player_rush_longest",
    "player_reception_longest",
    "player_last_td",
    "player_2+_td",
    "player_3+_td",
    "player_field_goals",
    "player_kicking_points",
    "player_tackles_assists",
]

DEFAULT_SNAPSHOT_OFFSET_HOURS = [1, 3, 6, 12, 24]

ARCHIVE_COLUMNS = [
    "run_timestamp",
    "event_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker_key",
    "bookmaker_title",
    "bookmaker_last_update",
    "market",
    "market_last_update",
    "player",
    "prop_type",
    "line",
    "price",
    "decimal_price",
    "american_price",
]


def load_api_key(override: Optional[str] = None) -> str:
    """Load The Odds API key."""
    if override:
        return override
    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY not found. Add it to your .env file.")
    return api_key


def convert_team_name(full_name: str) -> str:
    """Convert a full team name to its standard abbreviation."""
    team_map = {
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
    return team_map.get(full_name, full_name[:3].upper())


def fetch_events_for_date(
    api_key: str,
    target_date: str,
    regions: str = "us",
    bookmaker_filter: str = "draftkings",
) -> List[Dict]:
    """
    Fetch NFL events that existed at the provided snapshot.

    Uses the historical events endpoint which returns the board as it looked at `target_date`.
    """
    url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"
    params = {
        "apiKey": api_key,
        "date": target_date,
        "regions": regions,
        "bookmakers": bookmaker_filter,
    }
    response = requests.get(url, params=params, timeout=20)

    # Handle 401 Unauthorized (API quota exceeded or invalid key)
    if response.status_code == 401:
        print(f"  ⚠️  API Error 401 (Unauthorized) for {target_date} - skipping")
        return []

    # Handle other HTTP errors
    if response.status_code != 200:
        print(f"  ⚠️  API Error {response.status_code} for {target_date} - skipping")
        return []

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"  ⚠️  HTTP Error for {target_date}: {e} - skipping")
        return []

    payload = response.json()
    events = payload.get("data", payload)
    if not isinstance(events, list):
        raise ValueError(f"Unexpected events payload: {events}")
    return events


def fetch_event_details(api_key: str, event_id: str, snapshot: Optional[str] = None) -> Dict:
    """
    Fetch metadata for a single event.

    We prefer the historical endpoint when a snapshot is provided, otherwise fall back to the live endpoint.
    """
    if snapshot:
        url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}"
        params = {"apiKey": api_key, "date": snapshot}
    else:
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}"
        params = {"apiKey": api_key}
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if snapshot and isinstance(payload, dict):
        data = payload.get("data", payload)
        if isinstance(data, list):
            return data[0] if data else {}
        return data
    return payload


def fetch_historical_player_props_for_event(
    api_key: str,
    event_id: str,
    markets: Iterable[str],
    bookmakers: str,
    regions: str,
    odds_format: str,
    snapshot_date: Optional[str] = None,
) -> Dict:
    """Call the historical odds endpoint for the specified event."""
    base_path = "historical/sports/americanfootball_nfl" if snapshot_date else "sports/americanfootball_nfl"
    url = (
        f"https://api.the-odds-api.com/v4/{base_path}/events/{event_id}/odds"
    )
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "bookmakers": bookmakers,
        "regions": regions,
        "oddsFormat": odds_format,
    }
    if snapshot_date:
        params["date"] = snapshot_date
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_market_snapshots(bookmaker: Dict) -> Iterable[Dict]:
    """
    Yield individual snapshots for each market so they can be flattened.

    The Odds API returns history in slightly different shapes depending on the
    book and market. This helper normalizes to a stream of dictionaries with
    keys: market_key, market_last_update, and outcomes.
    """
    markets = bookmaker.get("markets", {})

    if isinstance(markets, dict):
        # Some responses use a mapping of market -> list of snapshots.
        for market_key, snapshots in markets.items():
            if not isinstance(snapshots, list):
                continue
            for snapshot in snapshots:
                yield {
                    "market_key": market_key,
                    "market_last_update": snapshot.get("last_update"),
                    "outcomes": snapshot.get("outcomes", []),
                }
        return

    if isinstance(markets, list):
        for market in markets:
            market_key = market.get("key")
            # Historical responses sometimes embed `history` under each market.
            history = market.get("history")
            if isinstance(history, list):
                for snapshot in history:
                    yield {
                        "market_key": market_key,
                        "market_last_update": snapshot.get("last_update"),
                        "outcomes": snapshot.get("outcomes", []),
                    }
                continue

            # Standard structure with outcomes only; treat current market as a snapshot.
            yield {
                "market_key": market_key,
                "market_last_update": market.get("last_update"),
                "outcomes": market.get("outcomes", []),
            }


def parse_historical_player_props(event_data: Dict, game_info: Dict) -> List[Dict]:
    """
    Flatten historical odds payload into a list of rows.

    Each entry represents a single snapshot of a line (Over/Under/Yes/No) with
    the associated timestamp and price.
    """
    props: List[Dict] = []
    bookmakers = []

    # Handle historical API response format with nested "data" object
    if isinstance(event_data, dict) and "data" in event_data:
        inner_data = event_data["data"]
        if isinstance(inner_data, dict) and "bookmakers" in inner_data:
            bookmakers = inner_data["bookmakers"]
    elif isinstance(event_data, dict) and "bookmakers" in event_data:
        bookmakers = event_data["bookmakers"]
    elif isinstance(event_data, list):
        bookmakers = event_data

    for bookmaker in bookmakers or []:
        bookmaker_key = bookmaker.get("key") or bookmaker.get("bookmaker_key")
        bookmaker_title = bookmaker.get("title")
        bookmaker_last_update = bookmaker.get("last_update")

        for market_snapshot in normalize_market_snapshots(bookmaker):
            market_key = market_snapshot.get("market_key")
            market_last_update = market_snapshot.get("market_last_update")

            for outcome in market_snapshot.get("outcomes", []):
                # Derive the best timestamp available for this line
                outcome_ts = (
                    outcome.get("last_update")
                    or outcome.get("timestamp")
                    or market_last_update
                    or bookmaker_last_update
                )

                props.append(
                    {
                        "event_id": game_info["event_id"],
                        "game_id": game_info["game_id"],
                        "commence_time": game_info["commence_time"],
                        "away_team": game_info["away_abbr"],
                        "home_team": game_info["home_abbr"],
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": bookmaker_title,
                        "market": market_key,
                        "player": outcome.get("description")
                        or outcome.get("player_name")
                        or outcome.get("name", ""),
                        "prop_type": (outcome.get("name") or "").lower(),
                        "line": outcome.get("point"),
                        "price": outcome.get("price"),
                        "american_price": outcome.get("price_american")
                        or outcome.get("american_price"),
                        "decimal_price": outcome.get("price_decimal")
                        or outcome.get("decimal_price"),
                        "snapshot_timestamp": outcome_ts,
                        "bookmaker_last_update": bookmaker_last_update,
                        "market_last_update": market_last_update,
                        "retrieved_at": datetime.utcnow().isoformat(),
                    }
                )

    return props


def normalize_for_archive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the historical fetch output to the append-only archive schema.

    This enables re-use of the backfilled data by downstream tooling that consumes
    data/historical/live_archive/player_props_archive.csv.
    """
    if df.empty:
        return df

    df = df.copy()

    # Pick the best timestamp we have for the snapshot to serve as run_timestamp.
    run_ts = df["snapshot_timestamp"].fillna(df["retrieved_at"]).fillna("")
    df["run_timestamp"] = run_ts

    # Standardise strings.
    df["player"] = df["player"].fillna("").str.strip()
    df["prop_type"] = df["prop_type"].fillna("").str.lower()

    # Subset to the archive schema, filling any missing numeric columns with NaN.
    archive_df = df.reindex(columns=ARCHIVE_COLUMNS, fill_value=None)

    # Drop rows that are missing a run timestamp or player name to avoid corrupt entries.
    archive_df = archive_df[(archive_df["run_timestamp"] != "") & (archive_df["player"] != "")]

    return archive_df


def build_game_info(event: Dict) -> Dict:
    """Construct a normalized metadata dictionary for downstream parsing."""
    commence_time = event.get("commence_time")
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    home_abbr = convert_team_name(home_team)
    away_abbr = convert_team_name(away_team)

    # Use YYYYMMDD to keep filenames deterministic when no NFL week is provided.
    commence_date = "unknown"
    if commence_time:
        commence_date = commence_time.split("T")[0].replace("-", "")

    game_id = f"{commence_date}_{away_abbr}_{home_abbr}"

    return {
        "event_id": event.get("id"),
        "commence_time": commence_time,
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "game_id": game_id,
    }


def append_to_archive(archive_df: pd.DataFrame, archive_path: Path) -> int:
    """
    Append normalized rows to the live archive CSV, performing basic deduplication.
    """
    if archive_df.empty:
        return 0

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        existing_df = pd.read_csv(archive_path)
    else:
        existing_df = pd.DataFrame(columns=ARCHIVE_COLUMNS)

    combined = pd.concat([existing_df, archive_df], ignore_index=True)
    before = len(existing_df)
    combined.drop_duplicates(
        subset=[
            "run_timestamp",
            "event_id",
            "market",
            "player",
            "prop_type",
            "line",
            "american_price",
        ],
        inplace=True,
    )

    combined.to_csv(archive_path, index=False)
    return max(len(combined) - before, 0)


def collect_historical_player_props(
    api_key: str,
    events: List[Dict],
    markets: Iterable[str],
    bookmakers: str,
    regions: str,
    odds_format: str,
    snapshot: Optional[str],
    snapshot_offsets: Iterable[int],
) -> pd.DataFrame:
    """
    Fetch and parse historical player props for a collection of events.
    """
    all_props: List[Dict] = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        commence_time = event.get("commence_time")
        commence_dt = None
        if commence_time:
            try:
                commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            except ValueError:
                commence_dt = None
        game_info = build_game_info(event)

        attempt_snapshots: List[str] = []
        if snapshot:
            attempt_snapshots.append(snapshot)

        if commence_dt:
            for hours in sorted(set(int(h) for h in snapshot_offsets if h > 0)):
                candidate_dt = commence_dt - timedelta(hours=hours)
                attempt_snapshots.append(candidate_dt.isoformat().replace("+00:00", "Z"))

        # Deduplicate while preserving order
        seen = set()
        attempt_snapshots = [
            snap for snap in attempt_snapshots if not (snap in seen or seen.add(snap))
        ]

        if not attempt_snapshots and snapshot is None:
            # As a last resort, fall back to the provided args.date (if any)
            if commence_time:
                attempt_snapshots.append(commence_time)

        parsed_props: List[Dict] = []
        for snap in attempt_snapshots or [snapshot]:
            if not snap:
                continue
            try:
                event_history = fetch_historical_player_props_for_event(
                    api_key=api_key,
                    event_id=event_id,
                    markets=markets,
                    bookmakers=bookmakers,
                    regions=regions,
                    odds_format=odds_format,
                    snapshot_date=snap,
                )
            except requests.HTTPError:
                continue

            parsed_props = parse_historical_player_props(event_history, game_info)
            if parsed_props:
                print(f"  ✅ {game_info['away_abbr']}@{game_info['home_abbr']} snapshot {snap} → {len(parsed_props)} rows")
                break

        if parsed_props:
            all_props.extend(parsed_props)
        else:
            print(f"  ⚠️  No historical props found for event {event_id} (tried {attempt_snapshots})")

    if not all_props:
        return pd.DataFrame()

    df = pd.DataFrame(all_props)
    df.sort_values(by=["event_id", "market", "player", "snapshot_timestamp"], inplace=True)
    return df


def discover_events(
    api_key: str,
    event_ids: Optional[Iterable[str]],
    snapshot: Optional[str],
    regions: str,
    bookmaker_filter: str,
) -> List[Dict]:
    """
    Locate events to fetch historical props for, using either explicit IDs or a snapshot date.
    """
    events: List[Dict] = []

    if event_ids:
        for event_id in event_ids:
            event = fetch_event_details(api_key, event_id, snapshot=snapshot)
            if event:
                events.append(event)
    else:
        if not snapshot:
            raise ValueError("--date is required when auto-discovering historical events")
        events = fetch_events_for_date(
            api_key=api_key,
            target_date=snapshot,
            regions=regions,
            bookmaker_filter=bookmaker_filter,
        )
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch historical NFL player prop odds from The Odds API."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--event-id",
        nargs="+",
        help="One or more Odds API event IDs to fetch history for.",
    )
    group.add_argument(
        "--date",
        help=(
            "ISO timestamp (e.g. 2024-10-05T00:00:00Z) to pull all events around "
            "that window."
        ),
    )

    parser.add_argument(
        "--markets",
        nargs="+",
        default=NFL_PLAYER_PROP_MARKETS,
        help="Optional subset of player prop markets to request.",
    )
    parser.add_argument(
        "--bookmakers",
        default="draftkings",
        help="Comma-separated list of bookmakers to include.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Odds API regions filter (default: us).",
    )
    parser.add_argument(
        "--odds-format",
        default="american",
        choices=["american", "decimal"],
        help="Target odds format.",
    )
    parser.add_argument(
        "--snapshot",
        help="Optional ISO timestamp to request a specific snapshot from the archive.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the CSV output. Defaults to data/historical/...",
    )
    parser.add_argument(
        "--api-key",
        help="Override ODDS_API_KEY environment variable with an explicit key.",
    )
    parser.add_argument(
        "--append-archive",
        action="store_true",
        help="When set, append the fetched rows to data/historical/live_archive/player_props_archive.csv.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/historical/live_archive/player_props_archive.csv"),
        help="Custom archive path when using --append-archive.",
    )
    parser.add_argument(
        "--snapshot-offset-hours",
        type=int,
        nargs="+",
        default=DEFAULT_SNAPSHOT_OFFSET_HOURS,
        help="Offsets (in hours before kickoff) to probe for historical snapshots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args.api_key)

    if api_key:
        masked = f"{api_key[:4]}...{api_key[-2:]}" if len(api_key) > 6 else "***"
        print(f"🔑 Using The Odds API key: {masked}")

    snapshot_for_events = args.snapshot or args.date
    events = discover_events(
        api_key=api_key,
        event_ids=args.event_id,
        snapshot=snapshot_for_events,
        regions=args.regions,
        bookmaker_filter=args.bookmakers,
    )
    event_ids = [event.get("id") for event in events]

    if not events:
        print("⚠️  No events found for the supplied filters.")
        return

    print("🏈 Fetching historical player prop odds")
    print(f"📅 Events discovered: {len(events)}")

    df = collect_historical_player_props(
        api_key=api_key,
        events=events,
        markets=args.markets,
        bookmakers=args.bookmakers,
        regions=args.regions,
        odds_format=args.odds_format,
        snapshot=snapshot_for_events,
        snapshot_offsets=args.snapshot_offset_hours,
    )

    if df.empty:
        print("\n⚠️  No historical props collected.")
        return

    output_path = args.output
    if not output_path:
        output_dir = Path("data") / "historical"
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = None
        if args.date:
            suffix = args.date.replace(":", "").replace("-", "").replace("T", "_")
        elif len(event_ids) == 1:
            suffix = event_ids[0]
        else:
            suffix = datetime.utcnow().strftime("%Y%m%d_%H%M")

        output_path = output_dir / f"player_props_history_{suffix}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\n✅ Saved {len(df)} rows to {output_path}")
    print(f"   Unique players: {df['player'].nunique()}")
    print(f"   Unique markets: {df['market'].nunique()}")

    if args.append_archive:
        archive_df = normalize_for_archive(df)
        appended = append_to_archive(archive_df, args.archive_path)
        print(f"   📚 Appended {appended} rows to {args.archive_path}")


if __name__ == "__main__":
    main()
