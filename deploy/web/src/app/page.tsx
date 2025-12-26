'use client';

import { useState, useMemo } from 'react';
import Header from '@/components/Header';
import PlayerCard from '@/components/PlayerCard';
import PickModal from '@/components/PickModal';
import BetSlip from '@/components/BetSlip';
import BottomNav, { NavItem } from '@/components/BottomNav';
import FilterBar from '@/components/FilterBar';
import StatsPanel from '@/components/StatsPanel';
import picksData from '@/data/picks.json';
import { Pick, DashboardData } from '@/lib/types';

const dashboardData = picksData as unknown as DashboardData;

export default function Dashboard() {
  const [selectedPicks, setSelectedPicks] = useState<Pick[]>([]);
  const [modalPick, setModalPick] = useState<Pick | null>(null);
  const [betSlipOpen, setBetSlipOpen] = useState(false);
  const [mobileTab, setMobileTab] = useState<NavItem>('picks');
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
  // Keep local normalizeGame for filtering picks
  const gamesFromMetadata = gamesMetadata || [];

  const handleSelectPick = (pick: Pick) => {
    setSelectedPicks(prev => {
      const exists = prev.find(p => p.id === pick.id);
      if (exists) {
        return prev.filter(p => p.id !== pick.id);
      }
      return [...prev, pick];
    });
  };

  const handleRemovePick = (pickId: string) => {
    setSelectedPicks(prev => prev.filter(p => p.id !== pickId));
  };

  const handleClearSlip = () => {
    setSelectedPicks([]);
  };

  const handleAnalyzePick = (pick: Pick) => {
    setModalPick(pick);
  };

  const handleAddFromModal = (pick: Pick) => {
    handleSelectPick(pick);
    setModalPick(null);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0c]">
      <Header week={week} stats={stats} />

      {/* Main Content Area - Desktop: with sidebar, Mobile: full width */}
      <div className="max-w-[1800px] mx-auto flex">
        {/* Main Content */}
        <main className="flex-1 min-w-0">
          {/* Filter Bar - hidden on mobile when stats tab active */}
          <div className={mobileTab === 'stats' ? 'hidden lg:block' : ''}>
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
          <div className={`lg:hidden ${mobileTab === 'stats' ? 'block' : 'hidden'}`}>
            <StatsPanel stats={stats} picks={picks} />
          </div>

          {/* Cards Grid - 3 cols on xl, 2 cols on lg, 1 on mobile */}
          <div className={`px-4 lg:px-6 py-4 pb-24 lg:pb-8 ${mobileTab === 'stats' ? 'hidden lg:block' : ''}`}>
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredPicks.map((pick, index) => (
                <div
                  key={pick.id}
                  className="animate-fade-in"
                  style={{ animationDelay: `${Math.min(index, 20) * 30}ms` }}
                >
                  <PlayerCard
                    pick={pick}
                    isSelected={selectedPicks.some(p => p.id === pick.id)}
                    onSelect={handleSelectPick}
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

        {/* Desktop Bet Slip Sidebar */}
        <aside className="hidden lg:block w-80 xl:w-96 border-l border-zinc-800 bg-[#0d0d0f] sticky top-0 h-screen overflow-y-auto">
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">Bet Slip</h2>
              {selectedPicks.length > 0 && (
                <span className="px-2 py-0.5 bg-emerald-500 text-black text-xs font-bold rounded-full">
                  {selectedPicks.length}
                </span>
              )}
            </div>

            {selectedPicks.length === 0 ? (
              <div className="text-center py-8">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-zinc-800 flex items-center justify-center">
                  <svg className="w-8 h-8 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                  </svg>
                </div>
                <p className="text-zinc-500 text-sm">Click + on picks to add them here</p>
              </div>
            ) : (
              <div className="space-y-3">
                {selectedPicks.map(pick => (
                  <div key={pick.id} className="bg-zinc-900 rounded-lg p-3 relative group">
                    <button
                      onClick={() => handleRemovePick(pick.id)}
                      className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                    <p className="font-medium text-white text-sm">{pick.player}</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      {pick.pick} {pick.line} {pick.market_display}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                      <span className={`text-xs font-medium ${pick.pick === 'OVER' ? 'text-emerald-500' : 'text-blue-500'}`}>
                        Proj: {pick.projection}
                      </span>
                      <span className="text-xs text-zinc-500">
                        Edge: {pick.edge > 0 ? '+' : ''}{pick.edge}
                      </span>
                    </div>
                  </div>
                ))}

                <div className="pt-4 border-t border-zinc-800">
                  <button
                    onClick={handleClearSlip}
                    className="w-full py-2 text-sm text-red-400 hover:text-red-300"
                  >
                    Clear All
                  </button>
                  <button
                    className="w-full py-3 mt-2 bg-emerald-500 hover:bg-emerald-400 text-black font-semibold rounded-lg transition-colors"
                  >
                    Export Picks
                  </button>
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Pick Modal */}
      <PickModal
        pick={modalPick}
        onClose={() => setModalPick(null)}
        onAddToSlip={handleAddFromModal}
      />

      {/* Mobile Bet Slip Panel */}
      <div className="lg:hidden">
        <BetSlip
          picks={selectedPicks}
          isOpen={betSlipOpen}
          onClose={() => setBetSlipOpen(false)}
          onRemovePick={handleRemovePick}
          onClearAll={handleClearSlip}
        />
      </div>

      {/* Bottom Navigation - Mobile only */}
      <div className="lg:hidden">
        <BottomNav
          activeTab={mobileTab}
          onTabChange={setMobileTab}
          slipCount={selectedPicks.length}
          onOpenSlip={() => setBetSlipOpen(true)}
        />
      </div>
    </div>
  );
}
