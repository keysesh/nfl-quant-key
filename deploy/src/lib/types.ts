// NFL QUANT Dashboard Types

export type Tier = 'elite' | 'strong' | 'moderate' | 'caution';
export type Direction = 'OVER' | 'UNDER';
export type Position = 'QB' | 'RB' | 'WR' | 'TE';

export interface GameHistory {
  weeks: number[];
  opponents: string[];
  receiving_yards: number[];
  receptions: number[];
  rushing_yards: number[];
  rushing_attempts: number[];
  passing_yards: number[];
  passing_attempts: number[];
  completions: number[];
  passing_tds: number[];
  rushing_tds: number[];
  receiving_tds: number[];
  defense_allowed?: (number | null)[];  // Current opponent's recent defensive performance
  defense_weeks?: number[];  // The weeks for defense data (may differ from player weeks)
  defense_opponent?: string;  // Which team's defense trend is shown
  defense_opponents?: string[];  // Teams the defense played each week
}

export interface Pick {
  id: string;
  player: string;
  position: Position;
  depth_position?: string | null;  // e.g., "WR1", "RB2", "TE1"
  team: string;
  opponent: string;
  headshot_url: string | null;
  team_logo_url: string;
  market: string;
  market_display: string;
  line: number;
  pick: Direction;
  projection: number;
  edge: number;
  confidence: number;
  tier: Tier;
  stars: number;
  ev: number;
  opp_rank?: number;
  opp_def_allowed?: number | null;  // Avg yards/receptions opponent allows to position
  opp_def_rank?: number | null;     // Rank 1-32 (1 = worst defense = most allowed)
  l5_rate?: number;
  l5_hits?: string;
  hist_over_rate: number;
  hist_count: number;
  game: string;
  game_history: GameHistory;
  // Weather & Game Script context
  vegas_total?: number | null;      // Implied total points for the game
  vegas_spread?: number | null;     // Point spread (negative = player's team favored)
  roof?: string | null;             // 'dome', 'outdoors', 'open', 'closed'
  temp?: number | null;             // Temperature in Fahrenheit (outdoor games)
  wind?: number | null;             // Wind speed in mph (outdoor games)
}

export interface DashboardStats {
  total_picks: number;
  avg_edge: number;
  games: number;
  elite_count: number;
  strong_count: number;
}

export interface GameInfo {
  game: string;
  normalized: string;
  away_team: string;
  home_team: string;
  gameday: string;
  gametime: string;
  stadium: string;
  roof: string;
  temp: number | null;
  wind: number | null;
}

export interface TeamInfo {
  name: string;
  nick: string;
  color: string;       // Primary color hex
  color2: string;      // Secondary color hex
  color3?: string | null;
  color4?: string | null;
  logo: string;        // ESPN logo URL
  logoSquared: string;
  wordmark: string;
  conf: string;
  division: string;
}

export interface GameLine {
  id: string;
  game: string;
  bet_type: string;
  pick: string;
  line: number;
  fair_line: number;
  confidence: number;
  edge: number;
  ev: number;
  tier: string;
  kelly_units: number;
  home_team: string;
  away_team: string;
}

export interface Parlay {
  id: string;
  rank: number;
  featured: boolean;
  legs: string;
  num_legs: number;
  true_odds: number;
  model_odds: number;
  true_prob: string;
  model_prob: string;
  edge: string;
  stake: string;
  potential_win: string;
  ev: string;
  games: string;
  sources: string;
  units: number;
}

export interface DashboardData {
  week: number;
  generated_at: string;
  stats: DashboardStats;
  teams: Record<string, TeamInfo>;
  games: GameInfo[];
  picks: Pick[];
  gameLines: GameLine[];
  parlays: Parlay[];
}

export interface FilterState {
  search: string;
  market: string | null;
  position: string | null;
  tier: string | null;
  direction: string | null;
}

export type SortField = 'confidence' | 'edge' | 'projection' | 'player' | 'line';
export type SortDirection = 'asc' | 'desc';

export interface SortState {
  field: SortField;
  direction: SortDirection;
}

// Market display names
export const MARKET_LABELS: Record<string, string> = {
  player_receptions: 'Rec',
  player_reception_yds: 'Rec Yds',
  player_rush_yds: 'Rush Yds',
  player_pass_yds: 'Pass Yds',
  player_pass_tds: 'Pass TDs',
  player_rush_attempts: 'Rush Att',
  player_anytime_td: 'Anytime TD',
  player_pass_attempts: 'Pass Att',
};

// Team colors for logos
export const TEAM_COLORS: Record<string, { primary: string; secondary: string }> = {
  ARI: { primary: '#97233F', secondary: '#000000' },
  ATL: { primary: '#A71930', secondary: '#000000' },
  BAL: { primary: '#241773', secondary: '#000000' },
  BUF: { primary: '#00338D', secondary: '#C60C30' },
  CAR: { primary: '#0085CA', secondary: '#101820' },
  CHI: { primary: '#C83803', secondary: '#0B162A' },
  CIN: { primary: '#FB4F14', secondary: '#000000' },
  CLE: { primary: '#311D00', secondary: '#FF3C00' },
  DAL: { primary: '#003594', secondary: '#041E42' },
  DEN: { primary: '#FB4F14', secondary: '#002244' },
  DET: { primary: '#0076B6', secondary: '#B0B7BC' },
  GB: { primary: '#203731', secondary: '#FFB612' },
  HOU: { primary: '#03202F', secondary: '#A71930' },
  IND: { primary: '#002C5F', secondary: '#A2AAAD' },
  JAX: { primary: '#006778', secondary: '#D7A22A' },
  KC: { primary: '#E31837', secondary: '#FFB81C' },
  LAC: { primary: '#0080C6', secondary: '#FFC20E' },
  LAR: { primary: '#003594', secondary: '#FFA300' },
  LV: { primary: '#000000', secondary: '#A5ACAF' },
  MIA: { primary: '#008E97', secondary: '#FC4C02' },
  MIN: { primary: '#4F2683', secondary: '#FFC62F' },
  NE: { primary: '#002244', secondary: '#C60C30' },
  NO: { primary: '#D3BC8D', secondary: '#101820' },
  NYG: { primary: '#0B2265', secondary: '#A71930' },
  NYJ: { primary: '#125740', secondary: '#000000' },
  PHI: { primary: '#004C54', secondary: '#A5ACAF' },
  PIT: { primary: '#FFB612', secondary: '#101820' },
  SEA: { primary: '#002244', secondary: '#69BE28' },
  SF: { primary: '#AA0000', secondary: '#B3995D' },
  TB: { primary: '#D50A0A', secondary: '#FF7900' },
  TEN: { primary: '#0C2340', secondary: '#4B92DB' },
  WAS: { primary: '#5A1414', secondary: '#FFB612' },
};
