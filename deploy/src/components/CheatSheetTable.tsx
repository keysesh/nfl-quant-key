'use client';

import { useState, useMemo } from 'react';
import { Pick, SortField, SortDirection } from '@/lib/types';
import Image from 'next/image';

interface CheatSheetTableProps {
  picks: Pick[];
  onSelectPick?: (pick: Pick) => void;
  onAnalyzePick?: (pick: Pick) => void;
}

function TierBadge({ tier }: { tier: string }) {
  const styles: Record<string, string> = {
    elite: 'tier-elite',
    strong: 'tier-strong',
    moderate: 'tier-moderate',
    caution: 'bg-red-500/15 text-red-400 border border-red-500/30',
  };
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${styles[tier] || styles.moderate}`}>
      {tier}
    </span>
  );
}

function StarRating({ stars }: { stars: number }) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`text-xs ${i <= stars ? 'star-filled' : 'star-empty'}`}>
          ★
        </span>
      ))}
    </div>
  );
}

export default function CheatSheetTable({ picks, onSelectPick, onAnalyzePick }: CheatSheetTableProps) {
  const [sortField, setSortField] = useState<SortField>('confidence');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const sortedPicks = useMemo(() => {
    return [...picks].sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      switch (sortField) {
        case 'confidence':
          aVal = a.confidence;
          bVal = b.confidence;
          break;
        case 'edge':
          aVal = a.edge;
          bVal = b.edge;
          break;
        case 'projection':
          aVal = a.projection;
          bVal = b.projection;
          break;
        case 'player':
          aVal = a.player;
          bVal = b.player;
          break;
        case 'line':
          aVal = a.line;
          bVal = b.line;
          break;
      }

      if (typeof aVal === 'string') {
        return sortDirection === 'asc'
          ? aVal.localeCompare(bVal as string)
          : (bVal as string).localeCompare(aVal);
      }
      return sortDirection === 'asc' ? aVal - (bVal as number) : (bVal as number) - aVal;
    });
  }, [picks, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <th
      onClick={() => handleSort(field)}
      className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors whitespace-nowrap"
    >
      <div className="flex items-center gap-1">
        {children}
        {sortField === field && (
          <span className="text-blue-400">
            {sortDirection === 'asc' ? '↑' : '↓'}
          </span>
        )}
      </div>
    </th>
  );

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700/50 bg-[#0f1729]">
      <table className="w-full min-w-[800px]">
        <thead className="bg-slate-800/50 border-b border-slate-700/50">
          <tr>
            <SortHeader field="player">Player</SortHeader>
            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">Prop</th>
            <SortHeader field="projection">Proj.</SortHeader>
            <SortHeader field="edge">Diff</SortHeader>
            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">Rating</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">EV</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">L-5</th>
            <th className="px-3 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">Pick</th>
            <th className="px-3 py-3 text-center text-xs font-semibold text-slate-400 uppercase tracking-wider w-12"></th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/30">
          {sortedPicks.map((pick) => {
            const isOver = pick.pick === 'OVER';
            const edgeColor = pick.edge > 0 ? 'text-emerald-400' : 'text-red-400';
            const edgeSign = pick.edge > 0 ? '+' : '';

            return (
              <tr
                key={pick.id}
                onClick={() => onSelectPick?.(pick)}
                className={`
                  hover:bg-slate-800/30 cursor-pointer transition-colors
                  ${pick.tier === 'elite' ? 'bg-amber-500/5' : ''}
                  ${pick.tier === 'strong' ? 'bg-cyan-500/5' : ''}
                `}
              >
                {/* Player */}
                <td className="px-3 py-3">
                  <div className="flex items-center gap-3">
                    <div className="relative flex-shrink-0">
                      <div className="w-10 h-10 rounded-full bg-slate-700 overflow-hidden">
                        {pick.headshot_url ? (
                          <Image
                            src={pick.headshot_url}
                            alt={pick.player}
                            width={40}
                            height={40}
                            className="w-full h-full object-cover"
                            unoptimized
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-slate-400 font-bold text-sm">
                            {pick.player.split(' ').map(n => n[0]).join('')}
                          </div>
                        )}
                      </div>
                      <div className="absolute -bottom-0.5 -right-0.5 w-4 h-4 rounded-full bg-slate-800 border border-[#0f1729] overflow-hidden">
                        <Image
                          src={pick.team_logo_url}
                          alt={pick.team}
                          width={16}
                          height={16}
                          className="w-full h-full object-contain"
                          unoptimized
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-white">{pick.player}</span>
                        <TierBadge tier={pick.tier} />
                      </div>
                      <span className="text-xs text-slate-500">
                        {pick.position} · {pick.team} vs {pick.opponent}
                      </span>
                    </div>
                  </div>
                </td>

                {/* Prop */}
                <td className="px-3 py-3">
                  <div className="flex flex-col">
                    <span className="font-mono text-white">{pick.line}</span>
                    <span className="text-xs text-slate-500">{pick.market_display}</span>
                  </div>
                </td>

                {/* Projection */}
                <td className="px-3 py-3 font-mono text-white">
                  {pick.projection.toFixed(1)}
                </td>

                {/* Diff/Edge */}
                <td className={`px-3 py-3 font-mono font-semibold ${edgeColor}`}>
                  {edgeSign}{pick.edge.toFixed(1)}
                </td>

                {/* Rating */}
                <td className="px-3 py-3">
                  <StarRating stars={pick.stars} />
                </td>

                {/* EV */}
                <td className={`px-3 py-3 font-mono text-sm ${pick.ev > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                  {pick.ev > 0 ? '+' : ''}{pick.ev.toFixed(1)}%
                </td>

                {/* L-5 */}
                <td className="px-3 py-3">
                  {pick.l5_rate !== undefined ? (
                    <div className={`inline-flex px-2 py-0.5 rounded text-xs font-medium ${
                      pick.l5_rate >= 60 ? 'bg-emerald-500/20 text-emerald-400' :
                      pick.l5_rate >= 40 ? 'bg-amber-500/20 text-amber-400' :
                      'bg-red-500/20 text-red-400'
                    }`}>
                      {pick.l5_rate}%
                    </div>
                  ) : pick.l5_hits ? (
                    <span className="text-xs text-slate-400">{pick.l5_hits}</span>
                  ) : (
                    <span className="text-xs text-slate-600">-</span>
                  )}
                </td>

                {/* Pick */}
                <td className="px-3 py-3">
                  <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${isOver ? 'pick-over' : 'pick-under'}`}>
                    {pick.pick.charAt(0)} {pick.line}
                  </span>
                </td>

                {/* Actions */}
                <td className="px-3 py-3 text-center">
                  <button
                    onClick={(e) => { e.stopPropagation(); onAnalyzePick?.(pick); }}
                    className="w-8 h-8 rounded-lg bg-slate-800/80 flex items-center justify-center text-slate-400 hover:bg-slate-700 hover:text-white transition-colors mx-auto"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
