'use client';

import { DashboardStats } from '@/lib/types';

export type MobileView = 'picks' | 'stats';

interface HeaderProps {
  week: number;
  stats: DashboardStats;
  mobileView?: MobileView;
  onMobileViewChange?: (view: MobileView) => void;
}

export default function Header({ week, stats, mobileView = 'picks', onMobileViewChange }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 glass-nav border-b border-white/[0.06]">
      <div className="max-w-[1800px] mx-auto px-4 lg:px-6 py-4">
        {/* Desktop Layout */}
        <div className="hidden md:flex items-center justify-between">
          {/* Logo & Week */}
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                NFL <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">QUANT</span>
              </h1>
              <p className="text-xs text-zinc-500 mt-0.5">Week {week} • 2025 Season</p>
            </div>
          </div>

          {/* Stats Row - Glass Pills */}
          <div className="flex items-center gap-3">
            {/* Total Picks */}
            <div className="px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] backdrop-blur-sm">
              <p className="text-xl font-bold text-white">{stats.total_picks}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Picks</p>
            </div>

            {/* Elite */}
            <div className="px-4 py-2 rounded-xl bg-yellow-500/[0.08] border border-yellow-500/20 backdrop-blur-sm">
              <p className="text-xl font-bold text-yellow-400">{stats.elite_count}</p>
              <p className="text-[10px] text-yellow-500/70 uppercase tracking-wider">Elite</p>
            </div>

            {/* Strong */}
            <div className="px-4 py-2 rounded-xl bg-cyan-500/[0.08] border border-cyan-500/20 backdrop-blur-sm">
              <p className="text-xl font-bold text-cyan-400">{stats.strong_count}</p>
              <p className="text-[10px] text-cyan-500/70 uppercase tracking-wider">Strong</p>
            </div>

            {/* Games */}
            <div className="px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] backdrop-blur-sm">
              <p className="text-xl font-bold text-white">{stats.games}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Games</p>
            </div>

            {/* Edge Badge - Glowing */}
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-500/[0.12] border border-emerald-500/30 backdrop-blur-sm glow-ring-emerald">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-sm font-semibold text-emerald-400">+{stats.avg_edge.toFixed(1)}% Edge</span>
            </div>
          </div>
        </div>

        {/* Mobile Layout */}
        <div className="md:hidden">
          {/* Top row */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-white">
                NFL <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">QUANT</span>
              </h1>
              <p className="text-xs text-zinc-500 mt-0.5">Week {week} • {stats.total_picks} Picks</p>
            </div>

            {/* View Toggle - Replaces Bottom Nav */}
            <div className="flex items-center gap-2">
              <div className="flex items-center rounded-xl bg-white/[0.04] border border-white/[0.08] p-1">
                <button
                  onClick={() => onMobileViewChange?.('picks')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    mobileView === 'picks'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </button>
                <button
                  onClick={() => onMobileViewChange?.('stats')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    mobileView === 'stats'
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </button>
              </div>

              {/* Edge Badge */}
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/[0.12] border border-emerald-500/30 backdrop-blur-sm shadow-[0_0_20px_rgba(34,197,94,0.15)]">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-semibold text-emerald-400">+{stats.avg_edge.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          {/* Stats row - Glass Pills */}
          <div className="flex gap-2 mt-4">
            <div className="flex-1 px-3 py-2 rounded-xl bg-yellow-500/[0.08] border border-yellow-500/20 backdrop-blur-sm text-center">
              <p className="text-xl font-bold text-yellow-400">{stats.elite_count}</p>
              <p className="text-[9px] text-yellow-500/70 uppercase tracking-wide">Elite</p>
            </div>
            <div className="flex-1 px-3 py-2 rounded-xl bg-cyan-500/[0.08] border border-cyan-500/20 backdrop-blur-sm text-center">
              <p className="text-xl font-bold text-cyan-400">{stats.strong_count}</p>
              <p className="text-[9px] text-cyan-500/70 uppercase tracking-wide">Strong</p>
            </div>
            <div className="flex-1 px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] backdrop-blur-sm text-center">
              <p className="text-xl font-bold text-white">{stats.games}</p>
              <p className="text-[9px] text-zinc-500 uppercase tracking-wide">Games</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
