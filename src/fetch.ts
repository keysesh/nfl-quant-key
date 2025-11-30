import { writeFileSync, mkdirSync } from "fs";
import { dirname } from "path";
import Papa from "papaparse";
import { Command } from "commander";

const RAW_DIR = "data/raw";
const PROC_DIR = "data/processed";

// Ensure directories exist
for (const d of [RAW_DIR, PROC_DIR]) {
  mkdirSync(d, { recursive: true });
}

interface SleeperPlayer {
  player_id: string;
  first_name: string;
  last_name: string;
  position: string;
  team: string;
  status: string;
  [key: string]: any;
}

interface SleeperGame {
  game_id: string;
  week: number;
  season: string;
  status: string;
  home_team: string;
  away_team: string;
  [key: string]: any;
}

async function httpGet(url: string, retries = 3): Promise<Response> {
  const headers = {
    "User-Agent": "Mozilla/5.0 (NFL QUANT Fetcher)"
  };
  for (let i = 0; i < retries; i++) {
    try {
      const r = await fetch(url, { keepalive: true, headers });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      return r;
    } catch (e) {
      if (i === retries - 1) throw e;
      console.log(`[retry ${i+1}/${retries}] ${e}`);
      await new Promise(res => setTimeout(res, 1500 * (i + 1)));
    }
  }
  throw new Error("Unreachable");
}

async function fetchJSON<T>(url: string, outName: string): Promise<T> {
  console.log(`[fetch] ${url}`);
  const r = await httpGet(url);
  const data = await r.json() as T;
  writeFileSync(`${RAW_DIR}/${outName}.json`, JSON.stringify(data, null, 2));
  console.log(`[saved] ${RAW_DIR}/${outName}.json`);
  return data;
}

async function fetchCSV(url: string, outName: string): Promise<any[]> {
  console.log(`[fetch] ${url}`);
  const r = await httpGet(url);
  const text = await r.text();
  writeFileSync(`${RAW_DIR}/${outName}.csv`, text);
  console.log(`[saved] ${RAW_DIR}/${outName}.csv`);

  // Parse CSV
  const parsed = Papa.parse<any>(text, { header: true });
  writeFileSync(`${PROC_DIR}/${outName}.rows.json`, JSON.stringify(parsed.data, null, 2));
  console.log(`[saved] ${PROC_DIR}/${outName}.rows.json`);
  return parsed.data;
}

async function fetchPlayers(): Promise<void> {
  try {
    const players = await fetchJSON<Record<string, SleeperPlayer>>("https://api.sleeper.app/v1/players/nfl", "sleeper_players");
    console.log(`✅ Players loaded: ${Object.keys(players).length}`);
  } catch (e) {
    console.error(`❌ Error fetching players: ${e}`);
    process.exit(1);
  }
}

async function fetchGames(): Promise<void> {
  try {
    // Get current season
    const state = await fetchJSON<{season: string}>("https://api.sleeper.app/v1/state/nfl", "sleeper_state");
    const currentSeason = state.season;
    
    if (currentSeason) {
      const games = await fetchJSON<SleeperGame[]>(`https://api.sleeper.com/schedule/nfl/regular/${currentSeason}`, "sleeper_games");
      console.log(`✅ Games loaded for season ${currentSeason}: ${games.length}`);
    } else {
      console.error("❌ Could not determine current season");
      process.exit(1);
    }
  } catch (e) {
    console.error(`❌ Error fetching games: ${e}`);
    process.exit(1);
  }
}

async function fetchCSVCommand(url: string, name: string): Promise<void> {
  try {
    const data = await fetchCSV(url, name);
    console.log(`✅ CSV loaded: ${data.length} rows`);
  } catch (e) {
    console.error(`❌ Error fetching CSV: ${e}`);
    process.exit(1);
  }
}

async function fetchJSONCommand(url: string, name: string): Promise<void> {
  try {
    const data = await fetchJSON(url, name);
    if (Array.isArray(data)) {
      console.log(`✅ JSON loaded: ${data.length} items`);
    } else if (typeof data === 'object') {
      console.log(`✅ JSON loaded: ${Object.keys(data).length} keys`);
    } else {
      console.log(`✅ JSON loaded: ${typeof data}`);
    }
  } catch (e) {
    console.error(`❌ Error fetching JSON: ${e}`);
    process.exit(1);
  }
}

const program = new Command();

program
  .name('nfl-fetcher')
  .description('NFL data fetcher CLI')
  .version('1.0.0');

program
  .command('players')
  .description('Fetch NFL players data from Sleeper API')
  .action(fetchPlayers);

program
  .command('games')
  .description('Fetch NFL games data from Sleeper API')
  .action(fetchGames);

program
  .command('csv')
  .description('Fetch CSV data from URL')
  .argument('<url>', 'CSV URL to fetch')
  .argument('<name>', 'Output name for the file')
  .action(fetchCSVCommand);

program
  .command('json')
  .description('Fetch JSON data from URL')
  .argument('<url>', 'JSON URL to fetch')
  .argument('<name>', 'Output name for the file')
  .action(fetchJSONCommand);

program.parse();
