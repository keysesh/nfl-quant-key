'use client';

import { GameInfo, TeamInfo } from '@/lib/types';

// Team data lookup type
type TeamsData = Record<string, TeamInfo>;

// Format game time for display (e.g., "1:00 PM")
function formatGameTime(gametime: string): string {
  if (!gametime) return '';
  const [hours, minutes] = gametime.split(':').map(Number);
  const period = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours > 12 ? hours - 12 : hours === 0 ? 12 : hours;
  return `${displayHours}:${minutes.toString().padStart(2, '0')} ${period}`;
}

// Get short day of week (e.g., "Sun", "Mon", "Thu")
function getDayOfWeek(gameday: string): string {
  if (!gameday) return '';
  const date = new Date(gameday + 'T12:00:00'); // Add time to avoid timezone issues
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return days[date.getDay()];
}

// Get weather/venue indicator
function getWeatherIcon(game: GameInfo): { icon: string; label: string } | null {
  if (game.roof === 'dome' || game.roof === 'closed') {
    return null; // No indicator needed for dome games
  }
  if (game.temp !== null && game.temp < 35) {
    return { icon: 'COLD', label: `${game.temp}°F` };
  }
  if (game.wind !== null && game.wind > 15) {
    return { icon: 'WIND', label: `${game.wind}mph` };
  }
  return null; // No indicator for normal outdoor games
}

// Parse hex color to RGB
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (result) {
    return {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    };
  }
  return null;
}

// Calculate relative luminance (0 = black, 1 = white)
function getLuminance(hex: string): number {
  const rgb = hexToRgb(hex);
  if (!rgb) return 0;
  // Using relative luminance formula
  const { r, g, b } = rgb;
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

// Lighten a color by mixing with white
function lightenColor(hex: string, amount: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  const { r, g, b } = rgb;
  const newR = Math.round(r + (255 - r) * amount);
  const newG = Math.round(g + (255 - g) * amount);
  const newB = Math.round(b + (255 - b) * amount);
  return `rgb(${newR}, ${newG}, ${newB})`;
}

// Get team colors with fallback, ensuring visibility on dark background
function getTeamColors(team: string, teams: TeamsData): { primary: string; secondary: string } {
  const teamData = teams[team.toUpperCase()];
  if (teamData) {
    let primary = teamData.color;
    let secondary = teamData.color2;

    // If primary color is too dark, try secondary or lighten it
    const primaryLum = getLuminance(primary);
    if (primaryLum < 0.25) {
      const secondaryLum = getLuminance(secondary);
      if (secondaryLum > primaryLum + 0.1) {
        // Secondary is brighter, use it
        primary = secondary;
      } else {
        // Lighten the primary color
        primary = lightenColor(teamData.color, 0.4);
      }
    }

    return { primary, secondary };
  }
  // Fallback to neutral colors
  return { primary: '#6366f1', secondary: '#818cf8' };
}

// Convert hex to rgba for backgrounds
function hexToRgba(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  if (rgb) {
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
  }
  return `rgba(99, 102, 241, ${alpha})`; // fallback indigo
}

// Sort games by kickoff datetime
function sortGamesByKickoff(games: GameInfo[]): GameInfo[] {
  return [...games].sort((a, b) => {
    const dateTimeA = `${a.gameday}T${a.gametime || '00:00'}`;
    const dateTimeB = `${b.gameday}T${b.gametime || '00:00'}`;
    return dateTimeA.localeCompare(dateTimeB);
  });
}

interface FilterBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  tierFilter: string;
  onTierChange: (tier: string) => void;
  marketFilter: string;
  onMarketChange: (market: string) => void;
  markets: { value: string; label: string }[];
  gameFilter: string;
  onGameChange: (game: string) => void;
  games: GameInfo[];
  teams: TeamsData;
  sortBy: string;
  onSortChange: (sort: string) => void;
  totalPicks: number;
}

export default function FilterBar({
  searchQuery,
  onSearchChange,
  tierFilter,
  onTierChange,
  marketFilter,
  onMarketChange,
  markets,
  gameFilter,
  onGameChange,
  games,
  teams,
  sortBy,
  onSortChange,
  totalPicks,
}: FilterBarProps) {
  const sortOptions = [
    { value: 'confidence_desc', label: 'Confidence (High→Low)' },
    { value: 'edge_desc', label: 'Edge (High→Low)' },
    { value: 'player_asc', label: 'Player A-Z' },
  ];

  const tiers = [
    { value: 'all', label: 'All Tiers' },
    { value: 'elite', label: 'Elite' },
    { value: 'strong', label: 'Strong' },
    { value: 'moderate', label: 'Moderate' },
  ];

  return (
    <div className="sticky top-[97px] md:top-[105px] z-40 glass-nav border-b border-white/[0.04]">
      <div className="px-4 lg:px-6 py-3">
        {/* Desktop Layout */}
        <div className="hidden md:block space-y-3">
          {/* Row 1: Search, Tiers, Market, Sort, Count */}
          <div className="flex items-center gap-4">
            {/* Search - Glass input */}
            <div className="relative flex-1 max-w-sm">
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search players, teams..."
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-white/[0.03] border border-white/[0.06] rounded-xl text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/40 focus:bg-white/[0.05] focus:shadow-[0_0_0_3px_rgba(34,197,94,0.1)] transition-all backdrop-blur-sm"
              />
            </div>

            {/* Tier Filter - Glass buttons */}
            <div className="flex items-center gap-2">
              {tiers.map(tier => (
                <button
                  key={tier.value}
                  onClick={() => onTierChange(tier.value)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                    tierFilter === tier.value
                      ? tier.value === 'elite'
                        ? 'bg-gradient-to-r from-yellow-500/20 to-yellow-500/10 text-yellow-400 border border-yellow-500/30 shadow-[0_0_16px_rgba(234,179,8,0.1)]'
                        : tier.value === 'strong'
                        ? 'bg-gradient-to-r from-cyan-500/20 to-cyan-500/10 text-cyan-400 border border-cyan-500/30 shadow-[0_0_16px_rgba(6,182,212,0.1)]'
                        : 'bg-white/[0.06] text-white border border-white/[0.1]'
                      : 'bg-white/[0.02] border border-white/[0.04] text-zinc-400 hover:bg-white/[0.06] hover:text-zinc-200'
                  }`}
                >
                  {tier.label}
                </button>
              ))}
            </div>

            {/* Market Filter - Glass select */}
            <select
              value={marketFilter}
              onChange={(e) => onMarketChange(e.target.value)}
              className="px-4 py-2.5 bg-white/[0.03] border border-white/[0.06] rounded-xl text-sm text-white focus:outline-none focus:border-emerald-500/40 cursor-pointer backdrop-blur-sm"
            >
              <option value="all">All Markets</option>
              {markets.map(market => (
                <option key={market.value} value={market.value}>
                  {market.label}
                </option>
              ))}
            </select>

            {/* Sort - Glass select */}
            <select
              value={sortBy}
              onChange={(e) => onSortChange(e.target.value)}
              className="px-4 py-2.5 bg-white/[0.03] border border-white/[0.06] rounded-xl text-sm text-white focus:outline-none focus:border-emerald-500/40 cursor-pointer backdrop-blur-sm"
            >
              {sortOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            {/* Results count - Glass pill */}
            <span className="text-sm text-zinc-400 ml-auto px-3 py-1 rounded-lg bg-white/[0.03] border border-white/[0.04]">
              {totalPicks} picks
            </span>
          </div>

          {/* Row 2: Game Filter Pills - Glass chips */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 font-medium">Games:</span>
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                onClick={() => onGameChange('all')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  gameFilter === 'all'
                    ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : 'bg-white/[0.02] border border-white/[0.04] text-zinc-400 hover:bg-white/[0.06]'
                }`}
              >
                All
              </button>
              {sortGamesByKickoff(games).map(game => {
                const weather = getWeatherIcon(game);
                const dayOfWeek = getDayOfWeek(game.gameday);
                const awayColors = getTeamColors(game.away_team, teams);
                const homeColors = getTeamColors(game.home_team, teams);
                const isSelected = gameFilter === game.normalized;

                // Create gradient from away team to home team colors
                const pillStyle = isSelected ? {} : {
                  background: `linear-gradient(135deg, ${hexToRgba(awayColors.primary, 0.15)} 0%, ${hexToRgba(homeColors.primary, 0.15)} 100%)`,
                  borderColor: hexToRgba(awayColors.primary, 0.3),
                };

                return (
                  <button
                    key={game.normalized}
                    onClick={() => onGameChange(game.normalized)}
                    title={`${game.stadium}${weather ? ` • ${weather.label}` : ''}`}
                    className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                      isSelected
                        ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 text-emerald-400 border-emerald-500/30 ring-1 ring-emerald-500/20'
                        : 'hover:brightness-125'
                    }`}
                    style={pillStyle}
                  >
                    <span className={isSelected ? 'text-emerald-500/70' : 'text-zinc-400'} style={{fontSize: '10px'}}>{dayOfWeek}</span>
                    <span
                      className="font-semibold"
                      style={{ color: isSelected ? undefined : awayColors.primary }}
                    >
                      {game.away_team}
                    </span>
                    <span className={isSelected ? 'text-emerald-600' : 'text-zinc-500'} style={{fontSize: '10px'}}>@</span>
                    <span
                      className="font-semibold"
                      style={{ color: isSelected ? undefined : homeColors.primary }}
                    >
                      {game.home_team}
                    </span>
                    <span className={isSelected ? 'text-emerald-600' : 'text-zinc-500'} style={{margin: '0 4px'}}>•</span>
                    <span className={isSelected ? 'text-emerald-500/70' : 'text-zinc-400'} style={{fontSize: '10px'}}>{formatGameTime(game.gametime)}</span>
                    {weather && (
                      <span className={`text-[9px] font-semibold ml-1 px-1 py-0.5 rounded ${
                        weather.icon === 'COLD' ? 'bg-sky-500/20 text-sky-400' : 'bg-amber-500/20 text-amber-400'
                      }`} title={weather.label}>
                        {weather.icon}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Mobile Layout */}
        <div className="md:hidden space-y-2">
          {/* Row 1: Search + Count */}
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-white/[0.03] border border-white/[0.06] rounded-xl text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/40 backdrop-blur-sm"
              />
            </div>
            <span className="text-xs text-zinc-500 whitespace-nowrap px-2 py-1 rounded-lg bg-white/[0.03] border border-white/[0.04]">{totalPicks} picks</span>
          </div>

          {/* Row 2: Tier chips - compact glass */}
          <div className="flex items-center gap-1.5">
            {tiers.map(tier => (
              <button
                key={tier.value}
                onClick={() => onTierChange(tier.value)}
                className={`flex-1 py-1.5 rounded-lg text-[11px] font-semibold transition-all ${
                  tierFilter === tier.value
                    ? tier.value === 'elite'
                      ? 'bg-gradient-to-r from-yellow-500/20 to-yellow-500/10 text-yellow-400 border border-yellow-500/30'
                      : tier.value === 'strong'
                      ? 'bg-gradient-to-r from-cyan-500/20 to-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                      : 'bg-white/[0.06] text-white border border-white/[0.1]'
                    : 'bg-white/[0.02] border border-white/[0.04] text-zinc-500'
                }`}
              >
                {tier.value === 'all' ? 'All' : tier.value.charAt(0).toUpperCase() + tier.value.slice(1)}
              </button>
            ))}
          </div>

          {/* Row 3: Dropdowns in a grid - Glass selects */}
          <div className="grid grid-cols-3 gap-1.5">
            {/* Market select */}
            <select
              value={marketFilter}
              onChange={(e) => onMarketChange(e.target.value)}
              className="w-full px-2 py-1.5 bg-white/[0.03] border border-white/[0.06] rounded-lg text-[11px] text-zinc-300 focus:outline-none backdrop-blur-sm"
            >
              <option value="all">Market</option>
              {markets.map(market => (
                <option key={market.value} value={market.value}>
                  {market.label}
                </option>
              ))}
            </select>

            {/* Game select */}
            <select
              value={gameFilter}
              onChange={(e) => onGameChange(e.target.value)}
              className="w-full px-2 py-1.5 bg-white/[0.03] border border-white/[0.06] rounded-lg text-[11px] text-zinc-300 focus:outline-none backdrop-blur-sm"
            >
              <option value="all">Game</option>
              {sortGamesByKickoff(games).map(game => (
                <option key={game.normalized} value={game.normalized}>
                  {getDayOfWeek(game.gameday)} {game.away_team}@{game.home_team}
                </option>
              ))}
            </select>

            {/* Sort select */}
            <select
              value={sortBy}
              onChange={(e) => onSortChange(e.target.value)}
              className="w-full px-2 py-1.5 bg-white/[0.03] border border-white/[0.06] rounded-lg text-[11px] text-zinc-300 focus:outline-none backdrop-blur-sm"
            >
              <option value="confidence_desc">Confidence</option>
              <option value="edge_desc">Edge</option>
              <option value="player_asc">A-Z</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}
