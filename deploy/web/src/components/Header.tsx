'use client';

import { DashboardStats } from '@/lib/types';

interface HeaderProps {
  week: number;
  stats: DashboardStats;
}

export default function Header({ week, stats }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 bg-[#0a0a0c]/95 backdrop-blur-md border-b border-zinc-800/50">
      <div className="max-w-[1800px] mx-auto px-4 lg:px-6 py-4">
        {/* Desktop Layout */}
        <div className="hidden md:flex items-center justify-between">
          {/* Logo & Week */}
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">NFL QUANT</h1>
              <p className="text-xs text-zinc-500 mt-0.5">Week {week} • 2025 Season</p>
            </div>
          </div>

          {/* Stats Row */}
          <div className="flex items-center gap-8">
            <div className="text-center">
              <p className="text-2xl font-bold text-white">{stats.total_picks}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Total Picks</p>
            </div>
            <div className="w-px h-8 bg-zinc-800" />
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-500">{stats.elite_count}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Elite</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-cyan-500">{stats.strong_count}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Strong</p>
            </div>
            <div className="w-px h-8 bg-zinc-800" />
            <div className="text-center">
              <p className="text-2xl font-bold text-white">{stats.games}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wider">Games</p>
            </div>
            <div className="w-px h-8 bg-zinc-800" />
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-sm font-semibold text-emerald-400">+{stats.avg_edge.toFixed(1)}% Avg Edge</span>
            </div>
          </div>
        </div>

        {/* Mobile Layout */}
        <div className="md:hidden">
          {/* Top row */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-white">NFL QUANT</h1>
              <p className="text-xs text-zinc-500 mt-0.5">Week {week} • {stats.total_picks} Picks</p>
            </div>
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
              <span className="text-xs font-medium text-emerald-500">+{stats.avg_edge.toFixed(1)}%</span>
            </div>
          </div>

          {/* Stats row */}
          <div className="flex gap-6 mt-4">
            <div>
              <p className="text-2xl font-bold text-yellow-500">{stats.elite_count}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Elite</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-cyan-500">{stats.strong_count}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Strong</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{stats.games}</p>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide">Games</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
