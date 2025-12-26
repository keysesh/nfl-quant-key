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
    <div className="sticky top-[73px] z-40 bg-[#0a0a0c]/95 backdrop-blur-sm border-b border-zinc-800/50">
      <div className="px-4 lg:px-6 py-3">
        {/* Desktop Layout */}
        <div className="hidden md:block space-y-3">
          {/* Row 1: Search, Tiers, Market, Sort, Count */}
          <div className="flex items-center gap-4">
            {/* Search */}
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
                className="w-full pl-10 pr-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20"
              />
            </div>

            {/* Tier Filter */}
            <div className="flex items-center gap-2">
              {tiers.map(tier => (
                <button
                  key={tier.value}
                  onClick={() => onTierChange(tier.value)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    tierFilter === tier.value
                      ? tier.value === 'elite'
                        ? 'bg-yellow-500/20 text-yellow-400 ring-1 ring-yellow-500/30'
                        : tier.value === 'strong'
                        ? 'bg-cyan-500/20 text-cyan-400 ring-1 ring-cyan-500/30'
                        : 'bg-zinc-700 text-white ring-1 ring-zinc-600'
                      : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300'
                  }`}
                >
                  {tier.label}
                </button>
              ))}
            </div>

            {/* Market Filter */}
            <select
              value={marketFilter}
              onChange={(e) => onMarketChange(e.target.value)}
              className="px-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer"
            >
              <option value="all">All Markets</option>
              {markets.map(market => (
                <option key={market.value} value={market.value}>
                  {market.label}
                </option>
              ))}
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => onSortChange(e.target.value)}
              className="px-4 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer"
            >
              {sortOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            {/* Results count */}
            <span className="text-sm text-zinc-500 ml-auto">
              {totalPicks} picks
            </span>
          </div>

          {/* Row 2: Game Filter Pills */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500 font-medium">Games:</span>
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                onClick={() => onGameChange('all')}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  gameFilter === 'all'
                    ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30'
                    : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                All
              </button>
              {sortGamesByKickoff(games).map(game => {
                const weather = getWeatherIcon(game);
                return (
                  <button
                    key={game.normalized}
                    onClick={() => onGameChange(game.normalized)}
                    title={`${game.stadium} • ${weather.label}`}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      gameFilter === game.normalized
                        ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30'
                        : 'bg-zinc-900 text-zinc-400 hover:bg-zinc-800'
                    }`}
                  >
                    <img src={getTeamLogoUrl(game.away_team)} alt={game.away_team} className="w-4 h-4 rounded-full" />
                    <span className="text-zinc-600">@</span>
                    <img src={getTeamLogoUrl(game.home_team)} alt={game.home_team} className="w-4 h-4 rounded-full" />
                    <span className="text-zinc-500 text-[10px] ml-0.5">{formatGameTime(game.gametime)}</span>
                    <span className="text-[10px]" title={weather.label}>{weather.icon}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Mobile Layout */}
        <div className="md:hidden space-y-3">
          {/* Search */}
          <div className="relative">
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
              placeholder="Search players..."
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-zinc-900 border border-zinc-800 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Filters row */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1 -mx-4 px-4 scrollbar-hide">
            {/* Tier chips */}
            {tiers.map(tier => (
              <button
                key={tier.value}
                onClick={() => onTierChange(tier.value)}
                className={`flex-shrink-0 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                  tierFilter === tier.value
                    ? tier.value === 'elite'
                      ? 'bg-yellow-500/20 text-yellow-400'
                      : tier.value === 'strong'
                      ? 'bg-cyan-500/20 text-cyan-400'
                      : 'bg-zinc-700 text-white'
                    : 'bg-zinc-900 text-zinc-400'
                }`}
              >
                {tier.label}
              </button>
            ))}

            {/* Divider */}
            <div className="w-px h-4 bg-zinc-700 flex-shrink-0" />

            {/* Market select */}
            <select
              value={marketFilter}
              onChange={(e) => onMarketChange(e.target.value)}
              className="flex-shrink-0 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-300 focus:outline-none"
            >
              <option value="all">All Markets</option>
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
              className="flex-shrink-0 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-300 focus:outline-none"
            >
              <option value="all">All Games</option>
              {sortGamesByKickoff(games).map(game => (
                <option key={game.normalized} value={game.normalized}>
                  {game.away_team} @ {game.home_team} {formatGameTime(game.gametime)}
                </option>
              ))}
            </select>

            {/* Sort select */}
            <select
              value={sortBy}
              onChange={(e) => onSortChange(e.target.value)}
              className="flex-shrink-0 px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-full text-xs text-zinc-300 focus:outline-none"
            >
              {sortOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            {/* Results count */}
            <span className="flex-shrink-0 text-xs text-zinc-500 ml-auto">
              {totalPicks}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
