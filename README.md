# NFL Quant Data Fetcher

A minimal data fetcher for NFL data with both Python and TypeScript implementations. Fetches data from public APIs (like Sleeper) and saves raw data to `/data/raw` and processed data to `/data/processed`.

## Project Structure

```
├── .cursor/
│   └── rules                 # Cursor AI guidelines
├── .venv/                    # Python virtual environment
├── data/
│   ├── raw/                  # Raw JSON/CSV files
│   └── processed/            # Processed Parquet/JSON files
├── src/
│   ├── fetch.py              # Python implementation
│   └── fetch.ts              # TypeScript implementation
├── fetch_all.sh              # All-in-one fetcher script
├── requirements.txt           # Python dependencies
├── package.json              # Node.js dependencies and scripts
└── README.md                 # This file
```

## Quick Start

### Option 1: Use the All-in-One Script (Recommended)

```bash
cd "/Users/keyonnesession/Desktop/NFL QUANT"
bash fetch_all.sh
```

This script will:
1. Activate the virtual environment
2. Fetch the NFL schedule
3. Fetch NFL players
4. Fetch NFL games
5. Display saved files

### Option 2: Python Version (Manual)

1. **Install dependencies:**
   ```bash
   cd "/Users/keyonnesession/Desktop/NFL QUANT"
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run commands:**
   ```bash
   # Fetch NFL players
   python src/fetch.py players
   
   # Fetch NFL games (now with correct API endpoint)
   python src/fetch.py games
   
   # Fetch custom CSV
   python src/fetch.py csv "https://example.com/data.csv" "my_data"
   
   # Fetch custom JSON
   python src/fetch.py json "https://api.example.com/data" "my_json"
   ```

### Option 3: TypeScript Version (Manual)

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Run commands:**
   ```bash
   # Fetch NFL players
   npm run fetch:players
   
   # Fetch NFL games  
   npm run fetch:games
   
   # Fetch custom CSV
   npm run fetch:csv -- "https://example.com/data.csv" "my_data"
   
   # Fetch custom JSON
   npm run fetch:json -- "https://api.example.com/data" "my_json"
   ```

## Features

- ✅ **Retry logic** with exponential backoff
- ✅ **Timeout handling** for network requests
- ✅ **Raw data persistence** in `/data/raw`
- ✅ **Processed data** in `/data/processed` (Parquet for Python, JSON for TS)
- ✅ **CLI interface** with clear commands
- ✅ **Error handling** with helpful messages
- ✅ **Type safety** (TypeScript version)
- ✅ **User-Agent headers** (fixes 403 errors from API)

## Data Sources

### Sleeper API (Public)
- **Players:** `https://api.sleeper.app/v1/players/nfl` ✅
- **State/Season:** `https://api.sleeper.app/v1/state/nfl` ✅
- **Games/Schedule:** `https://api.sleeper.com/schedule/nfl/regular/{season}` ✅ (note: .com not .app)

### Other Sources
- **GitHub CSVs:** Use raw URLs (click "Raw" button)
- **Any JSON API:** Just provide the endpoint URL

### Historical Player Prop Odds (The Odds API)
- Archive live props going forward (low credit footprint default):
  - `python archive_player_props.py --bookmakers draftkings --markets player_pass_yds player_rush_yds player_receptions player_reception_yds player_anytime_td`
  - Append-only CSV + raw JSON snapshots land in `data/historical/live_archive/`; request usage from Odds API headers is printed each run
- Snapshot an older slate with `python fetch_historical_player_props.py --date 2024-10-05T12:00:00Z` (uses `v4/historical` endpoints)
- Target specific games via event IDs and a snapshot: `python fetch_historical_player_props.py --event-id <EVENT_ID> --snapshot 2024-10-05T17:00:00Z`
- Optional `--api-key` lets you override `ODDS_API_KEY` explicitly; the script logs a masked key so you can confirm which credential was used
- Outputs land in `data/historical/` by default with timestamps and bookmaker metadata
- If the API responds with 422, the requested markets weren’t archived at that timestamp for your plan—trim the `--markets` list or confirm coverage with The Odds API team
- Weekly calibration loop once games are final:
  1. `python build_prop_training_dataset.py` to merge archived lines with Sleeper outcomes
  2. `python evaluate_archived_player_props.py --output data/historical/player_prop_eval_weekX.csv`
  3. `python train_prop_calibrator.py --evaluation data/historical/player_prop_eval_weekX.csv`
  4. Optional: `python backtest_player_props.py --start-week 1 --end-week 7 --output reports/props_backtest_week1_7.csv` for headline metrics
- Tip: Each Odds API call costs `regions × markets` credits (historical calls cost `10 × regions × markets`). Use the usage summary printed by `archive_player_props.py` to stay within the 20K monthly credit cap.

## Example Output

When you run `fetch_all.sh`, you'll get:

```
🚀 Starting NFL QUANT Data Fetcher...

📅 Fetching NFL Schedule...
✅ Schedule saved: 272 games

👥 Fetching NFL Players...
[fetch] https://api.sleeper.app/v1/players/nfl
[saved] data/raw/sleeper_players.json
✅ Players loaded: 11400

🎮 Fetching NFL Games...
[fetch] https://api.sleeper.app/v1/state/nfl
[saved] data/raw/sleeper_state.json
[fetch] https://api.sleeper.com/schedule/nfl/regular/2025
[saved] data/raw/sleeper_games.json
✅ Games loaded for season 2025: 272

✅ All fetches complete!

📁 Output files:
-rw-r--r-- 37K sleeper_games.json
-rw-r--r-- 16M sleeper_players.json
-rw-r--r-- 37K sleeper_schedule_2025.json
-rw-r--r--  252B sleeper_state.json
```

## Development

### Adding New Data Sources

**Python:**
```python
@app.command()
def my_source():
    """Fetch data from my custom source."""
    data = fetch_json("https://my-api.com/data", "my_source")
    print(f"✅ Loaded: {len(data)} items")
```

**TypeScript:**
```typescript
program
  .command('my-source')
  .description('Fetch data from my custom source')
  .action(async () => {
    const data = await fetchJSON("https://my-api.com/data", "my_source");
    console.log(`✅ Loaded: ${data.length} items`);
  });
```

### Best Practices

1. **Use raw endpoints** for GitHub CSVs
2. **Add retries** for flaky endpoints (done by default)
3. **Add User-Agent headers** (Sleeper API requires this)
4. **Version outputs** with timestamps if needed
5. **Store secrets** in `.env` files (not in code)
6. **Test with mock data** before hitting real APIs

## Troubleshooting

- **403 Forbidden error:** Ensure User-Agent headers are set (now fixed in both implementations)
- **Virtual environment issues:** Delete `.venv` and recreate: `python3 -m venv .venv`
- **Import errors:** Run `pip install -r requirements.txt` or `npm install`
- **TypeScript errors:** Ensure Node.js 18+ is installed

## Next Steps

- Add scheduled runs via GitHub Actions
- Implement data validation schemas
- Add unit tests with mocked network calls
- Create data visualization dashboards
- Add more NFL data sources (nflverse, ESPN, etc.)
- Implement data transformation pipelines
