'use client';

import { useState, useMemo } from 'react';
import Header, { MobileView } from '@/components/Header';
import PlayerCard from '@/components/PlayerCard';
import PickModal from '@/components/PickModal';
import FilterBar from '@/components/FilterBar';
import StatsPanel from '@/components/StatsPanel';
import ViewTabs, { ViewType } from '@/components/ViewTabs';
import GameLinesPanel from '@/components/GameLinesPanel';
import ParlaysPanel from '@/components/ParlaysPanel';
import picksData from '@/data/picks.json';
import { Pick, DashboardData } from '@/lib/types';

const dashboardData = picksData as unknown as DashboardData;

export default function Dashboard() {
  const [modalPick, setModalPick] = useState<Pick | null>(null);
  const [mobileView, setMobileView] = useState<MobileView>('picks');
  const [activeTab, setActiveTab] = useState<ViewType>('picks');
  const [searchQuery, setSearchQuery] = useState('');
  const [tierFilter, setTierFilter] = useState<string>('all');
  const [marketFilter, setMarketFilter] = useState<string>('all');
  const [gameFilter, setGameFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('confidence_desc');

  const { week, stats, picks, games: gamesMetadata, teams = {}, gameLines = [], parlays = [] } = dashboardData;

  // Normalize game name so "ARI vs CIN" and "CIN vs ARI" are the same
  const normalizeGame = (game: string): string => {
    const parts = game.split(' vs ');
    if (parts.length === 2) {
      return parts.sort().join(' vs ');
    }
    return game;
  };

  // Filter out games that have already started
  const upcomingGames = useMemo(() => {
    const now = new Date();
    return (gamesMetadata || []).filter(game => {
      if (!game.gameday || !game.gametime) return true; // Keep if no time info
      try {
        // Parse game datetime - gametime is in US Eastern Time
        // Append EST/EDT offset to handle timezone correctly on Vercel (UTC)
        // Use -05:00 (EST) as conservative default (games are later in EDT -04:00)
        const gameDateTime = new Date(`${game.gameday}T${game.gametime}:00-05:00`);
        // Keep games that haven't started yet (with 5 min buffer)
        return gameDateTime > new Date(now.getTime() - 5 * 60 * 1000);
      } catch {
        return true; // Keep if parsing fails
      }
    });
  }, [gamesMetadata]);

  // Get set of upcoming game keys for filtering picks
  const upcomingGameKeys = useMemo(() => {
    return new Set(upcomingGames.map(g => g.normalized));
  }, [upcomingGames]);

  // Filter game lines to only show upcoming games
  const filteredGameLines = useMemo(() => {
    if (upcomingGameKeys.size === 0) return gameLines;
    return gameLines.filter(line => {
      const gameKey = normalizeGame(`${line.away_team} vs ${line.home_team}`);
      return upcomingGameKeys.has(gameKey);
    });
  }, [gameLines, upcomingGameKeys]);

  // Filter parlays to only show those where ALL games are upcoming
  const filteredParlays = useMemo(() => {
    if (upcomingGameKeys.size === 0) return parlays;
    return parlays.filter(parlay => {
      // Parse games string (e.g., "ARI vs CIN | SEA vs CAR")
      const parlayGames = parlay.games.split(' | ').map(g => g.trim());
      // Keep parlay only if ALL games are upcoming
      return parlayGames.every(game => {
        const gameKey = normalizeGame(game);
        return upcomingGameKeys.has(gameKey);
      });
    });
  }, [parlays, upcomingGameKeys]);

  // Filter and sort picks
  const filteredPicks = useMemo(() => {
    const filtered = picks.filter(pick => {
      // Filter out picks from games that have started
      const pickGameKey = normalizeGame(pick.game);
      if (upcomingGameKeys.size > 0 && !upcomingGameKeys.has(pickGameKey)) return false;

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesPlayer = pick.player.toLowerCase().includes(query);
        const matchesTeam = pick.team.toLowerCase().includes(query);
        const matchesOpponent = pick.opponent.toLowerCase().includes(query);
        if (!matchesPlayer && !matchesTeam && !matchesOpponent) return false;
      }

      // Tier filter
      if (tierFilter !== 'all' && pick.tier !== tierFilter) return false;

      // Market filter
      if (marketFilter !== 'all' && pick.market !== marketFilter) return false;

      // Game filter (normalized)
      if (gameFilter !== 'all' && pickGameKey !== gameFilter) return false;

      return true;
    });

    // Sort picks
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'confidence_desc':
          return b.confidence - a.confidence;
        case 'edge_desc':
          return Math.abs(b.edge) - Math.abs(a.edge);
        case 'player_asc':
          return a.player.localeCompare(b.player);
        default:
          return b.confidence - a.confidence;
      }
    });
  }, [picks, searchQuery, tierFilter, marketFilter, gameFilter, sortBy, upcomingGameKeys]);

  // Get unique markets for filter
  const markets = useMemo(() => {
    const uniqueMarkets = [...new Set(picks.map(p => p.market))];
    return uniqueMarkets.map(m => ({
      value: m,
      label: picks.find(p => p.market === m)?.market_display || m
    }));
  }, [picks]);

  const handleAnalyzePick = (pick: Pick) => {
    setModalPick(pick);
  };

  // Tab counts (use filtered counts)
  const tabCounts = {
    picks: filteredPicks.length,
    lines: filteredGameLines.length,
    parlays: filteredParlays.length,
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c]">
      <Header
        week={week}
        stats={stats}
        mobileView={mobileView}
        onMobileViewChange={setMobileView}
      />

      {/* Main Content Area */}
      <div className="max-w-[1800px] mx-auto">
        <main className="flex-1 min-w-0">
          {/* View Tabs */}
          <ViewTabs
            active={activeTab}
            onChange={setActiveTab}
            counts={tabCounts}
          />

          {/* Filter Bar - only show for picks tab */}
          {activeTab === 'picks' && (
            <div className={mobileView === 'stats' ? 'hidden md:block' : ''}>
              <FilterBar
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                tierFilter={tierFilter}
                onTierChange={setTierFilter}
                marketFilter={marketFilter}
                onMarketChange={setMarketFilter}
                markets={markets}
                gameFilter={gameFilter}
                onGameChange={setGameFilter}
                games={upcomingGames}
                teams={teams}
                sortBy={sortBy}
                onSortChange={setSortBy}
                totalPicks={filteredPicks.length}
              />
            </div>
          )}

          {/* Mobile Stats Panel (shown when stats tab active) */}
          <div className={`md:hidden ${mobileView === 'stats' ? 'block overflow-y-auto' : 'hidden'}`}>
            <div className="px-4 py-4">
              <StatsPanel stats={stats} picks={picks} />
            </div>
          </div>

          {/* Content based on active tab */}
          <div className={`px-4 lg:px-6 py-4 ${mobileView === 'stats' ? 'hidden md:block' : ''}`}>
            {/* Player Props Tab */}
            {activeTab === 'picks' && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                  {filteredPicks.map((pick, index) => (
                    <div
                      key={pick.id}
                      className="animate-fade-in"
                      style={{ animationDelay: `${Math.min(index, 20) * 30}ms` }}
                    >
                      <PlayerCard
                        pick={pick}
                        isSelected={false}
                        onSelect={() => {}}
                        onAnalyze={handleAnalyzePick}
                      />
                    </div>
                  ))}
                </div>

                {filteredPicks.length === 0 && (
                  <div className="text-center py-12">
                    <p className="text-zinc-500 text-lg">No picks match your filters</p>
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setTierFilter('all');
                        setMarketFilter('all');
                        setGameFilter('all');
                      }}
                      className="mt-4 text-emerald-500 hover:text-emerald-400"
                    >
                      Clear filters
                    </button>
                  </div>
                )}
              </>
            )}

            {/* Game Lines Tab */}
            {activeTab === 'lines' && (
              <GameLinesPanel gameLines={filteredGameLines} teams={teams} />
            )}

            {/* Parlays Tab */}
            {activeTab === 'parlays' && (
              <ParlaysPanel parlays={filteredParlays} />
            )}
          </div>
        </main>
      </div>

      {/* Pick Modal */}
      <PickModal
        pick={modalPick}
        onClose={() => setModalPick(null)}
      />
    </div>
  );
}
