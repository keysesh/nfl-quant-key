'use client';

import { Pick, DashboardStats } from '@/lib/types';

interface StatsPanelProps {
  stats: DashboardStats;
  picks: Pick[];
}

export default function StatsPanel({ stats, picks }: StatsPanelProps) {
  // Calculate tier breakdown
  const tierCounts = picks.reduce((acc, pick) => {
    acc[pick.tier] = (acc[pick.tier] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Calculate market breakdown
  const marketCounts = picks.reduce((acc, pick) => {
    acc[pick.market_display] = (acc[pick.market_display] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Calculate direction breakdown
  const overCount = picks.filter(p => p.pick === 'OVER').length;
  const underCount = picks.filter(p => p.pick === 'UNDER').length;

  // Calculate avg confidence by tier
  const avgConfByTier = ['elite', 'strong', 'moderate'].map(tier => {
    const tierPicks = picks.filter(p => p.tier === tier);
    const avgConf = tierPicks.length > 0
      ? tierPicks.reduce((sum, p) => sum + p.confidence, 0) / tierPicks.length
      : 0;
    return { tier, count: tierPicks.length, avgConf };
  });

  // Top markets by count
  const topMarkets = Object.entries(marketCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-zinc-900 rounded-xl p-4">
          <p className="text-xs text-zinc-500 mb-1">Total Picks</p>
          <p className="text-2xl font-bold text-white">{stats.total_picks}</p>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4">
          <p className="text-xs text-zinc-500 mb-1">Avg Edge</p>
          <p className="text-2xl font-bold text-emerald-400">+{stats.avg_edge.toFixed(1)}</p>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4">
          <p className="text-xs text-zinc-500 mb-1">Games</p>
          <p className="text-2xl font-bold text-white">{stats.games}</p>
        </div>
        <div className="bg-zinc-900 rounded-xl p-4">
          <p className="text-xs text-zinc-500 mb-1">Elite + Strong</p>
          <p className="text-2xl font-bold text-yellow-400">{stats.elite_count + stats.strong_count}</p>
        </div>
      </div>

      {/* Direction Breakdown */}
      <div className="bg-zinc-900 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Direction Split</h3>
        <div className="flex items-center gap-2 mb-2">
          <div
            className="h-3 bg-emerald-500 rounded-full transition-all"
            style={{ width: `${(overCount / picks.length) * 100}%` }}
          />
          <div
            className="h-3 bg-blue-500 rounded-full transition-all"
            style={{ width: `${(underCount / picks.length) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs">
          <span className="text-emerald-400">OVER: {overCount} ({((overCount / picks.length) * 100).toFixed(0)}%)</span>
          <span className="text-blue-400">UNDER: {underCount} ({((underCount / picks.length) * 100).toFixed(0)}%)</span>
        </div>
      </div>

      {/* Tier Breakdown */}
      <div className="bg-zinc-900 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">By Confidence Tier</h3>
        <div className="space-y-3">
          {avgConfByTier.map(({ tier, count, avgConf }) => (
            <div key={tier} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${
                  tier === 'elite' ? 'bg-yellow-400' :
                  tier === 'strong' ? 'bg-cyan-400' :
                  'bg-zinc-500'
                }`} />
                <span className="text-sm text-zinc-300 capitalize">{tier}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-zinc-500">{(avgConf * 100).toFixed(0)}% avg</span>
                <span className={`text-sm font-semibold px-2 py-0.5 rounded ${
                  tier === 'elite' ? 'bg-yellow-500/20 text-yellow-400' :
                  tier === 'strong' ? 'bg-cyan-500/20 text-cyan-400' :
                  'bg-zinc-800 text-zinc-400'
                }`}>
                  {count}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Top Markets */}
      <div className="bg-zinc-900 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Top Markets</h3>
        <div className="space-y-2">
          {topMarkets.map(([market, count]) => (
            <div key={market} className="flex items-center justify-between">
              <span className="text-sm text-zinc-400">{market}</span>
              <div className="flex items-center gap-2">
                <div className="w-24 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full"
                    style={{ width: `${(count / picks.length) * 100}%` }}
                  />
                </div>
                <span className="text-xs text-zinc-500 w-8 text-right">{count}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence Distribution */}
      <div className="bg-zinc-900 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Confidence Distribution</h3>
        <div className="grid grid-cols-5 gap-1">
          {[50, 60, 70, 80, 90].map((threshold) => {
            const count = picks.filter(p => {
              const conf = p.confidence * 100;
              return conf >= threshold && conf < threshold + 10;
            }).length;
            const height = Math.max(20, (count / picks.length) * 150);
            return (
              <div key={threshold} className="flex flex-col items-center gap-1">
                <div
                  className={`w-full rounded-t transition-all ${
                    threshold >= 70 ? 'bg-emerald-500' :
                    threshold >= 60 ? 'bg-yellow-500' :
                    'bg-zinc-600'
                  }`}
                  style={{ height: `${height}px` }}
                />
                <span className="text-[10px] text-zinc-500">{threshold}%</span>
                <span className="text-xs text-zinc-400">{count}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
