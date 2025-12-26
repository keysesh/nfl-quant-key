'use client';

import { useState, useMemo } from 'react';
import Header, { MobileView } from '@/components/Header';
import PlayerCard from '@/components/PlayerCard';
import PickModal from '@/components/PickModal';
import FilterBar from '@/components/FilterBar';
import StatsPanel from '@/components/StatsPanel';
import picksData from '@/data/picks.json';
import { Pick, DashboardData } from '@/lib/types';

const dashboardData = picksData as unknown as DashboardData;

export default function Dashboard() {
  const [modalPick, setModalPick] = useState<Pick | null>(null);
  const [mobileView, setMobileView] = useState<MobileView>('picks');
  const [searchQuery, setSearchQuery] = useState('');
  const [tierFilter, setTierFilter] = useState<string>('all');
  const [marketFilter, setMarketFilter] = useState<string>('all');
  const [gameFilter, setGameFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('confidence_desc');

  const { week, stats, picks, games: gamesMetadata } = dashboardData;

  // Normalize game name so "ARI vs CIN" and "CIN vs ARI" are the same
  const normalizeGame = (game: string): string => {
    const parts = game.split(' vs ');
    if (parts.length === 2) {
      return parts.sort().join(' vs ');
    }
    return game;
  };

  // Filter and sort picks
  const filteredPicks = useMemo(() => {
    const filtered = picks.filter(pick => {
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
      if (gameFilter !== 'all' && normalizeGame(pick.game) !== gameFilter) return false;

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
  }, [picks, searchQuery, tierFilter, marketFilter, gameFilter, sortBy]);

  // Get unique markets for filter
  const markets = useMemo(() => {
    const uniqueMarkets = [...new Set(picks.map(p => p.market))];
    return uniqueMarkets.map(m => ({
      value: m,
      label: picks.find(p => p.market === m)?.market_display || m
    }));
  }, [picks]);

  // Games metadata from JSON (already sorted by kickoff in FilterBar)
  const gamesFromMetadata = gamesMetadata || [];

  const handleAnalyzePick = (pick: Pick) => {
    setModalPick(pick);
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
          {/* Filter Bar - hidden on mobile when stats tab active */}
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
              games={gamesFromMetadata}
              sortBy={sortBy}
              onSortChange={setSortBy}
              totalPicks={filteredPicks.length}
            />
          </div>

          {/* Mobile Stats Panel (shown when stats tab active) */}
          <div className={`md:hidden ${mobileView === 'stats' ? 'block overflow-y-auto' : 'hidden'}`}>
            <div className="px-4 py-4">
              <StatsPanel stats={stats} picks={picks} />
            </div>
          </div>

          {/* Cards Grid - 3 cols on xl, 2 cols on lg, 1 on mobile */}
          <div className={`px-4 lg:px-6 py-4 ${mobileView === 'stats' ? 'hidden md:block' : ''}`}>
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
