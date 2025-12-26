'use client';

import { GameInfo } from '@/lib/types';

// Get team logo URL - ESPN circular logos
function getTeamLogoUrl(team: string): string {
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${team.toLowerCase()}.png`;
}

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

// Get weather/venue icon
function getWeatherIcon(game: GameInfo): { icon: string; label: string } {
  if (game.roof === 'dome' || game.roof === 'closed') {
    return { icon: '🏟️', label: 'Dome' };
  }
  if (game.temp !== null && game.temp < 35) {
    return { icon: '❄️', label: `${game.temp}°F` };
  }
  if (game.wind !== null && game.wind > 15) {
    return { icon: '💨', label: `${game.wind}mph` };
  }
  if (game.temp !== null) {
    return { icon: '☀️', label: `${game.temp}°F` };
  }
  return { icon: '🏈', label: 'Outdoor' };
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
                return (
                  <button
                    key={game.normalized}
                    onClick={() => onGameChange(game.normalized)}
                    title={`${game.stadium} • ${weather.label}`}
                    className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      gameFilter === game.normalized
                        ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : 'bg-white/[0.02] border border-white/[0.04] text-zinc-400 hover:bg-white/[0.06]'
                    }`}
                  >
                    <span className="text-zinc-500 text-[10px]">{dayOfWeek}</span>
                    <span className="font-semibold">{game.away_team}</span>
                    <span className="text-zinc-600 text-[10px]">@</span>
                    <span className="font-semibold">{game.home_team}</span>
                    <span className="text-zinc-600 mx-1">•</span>
                    <span className="text-zinc-500 text-[10px]">{formatGameTime(game.gametime)}</span>
                    {weather.icon !== '🏈' && <span className="text-[10px] ml-0.5" title={weather.label}>{weather.icon}</span>}
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
