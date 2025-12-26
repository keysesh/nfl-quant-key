'use client';

import { Pick } from '@/lib/types';
import Image from 'next/image';
import { useState } from 'react';

// Get team logo URL - ESPN circular logos (500x500, better quality)
function getTeamLogoUrl(team: string): string {
  const abbr = team.toLowerCase();
  return `https://a.espncdn.com/i/teamlogos/nfl/500/${abbr}.png`;
}

// Generate star rating
function StarRating({ stars, maxStars = 5 }: { stars: number; maxStars?: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: maxStars }).map((_, i) => (
        <span
          key={i}
          className={`text-xs ${i < stars ? 'text-yellow-400' : 'text-zinc-700'}`}
        >
          ★
        </span>
      ))}
    </div>
  );
}

interface PlayerCardProps {
  pick: Pick;
  isSelected?: boolean;
  onSelect?: (pick: Pick) => void;
  onAnalyze?: (pick: Pick) => void;
}

export default function PlayerCard({ pick, isSelected = false, onSelect, onAnalyze }: PlayerCardProps) {
  const [logoError, setLogoError] = useState(false);
  const isOver = pick.pick === 'OVER';
  const edgeSign = pick.edge >= 0 ? '+' : '';

  // Confidence-based border color
  const confidenceBorder = pick.confidence >= 0.7
    ? 'border-emerald-500/40 ring-1 ring-emerald-500/10'
    : pick.confidence >= 0.6
    ? 'border-yellow-500/30 ring-1 ring-yellow-500/10'
    : 'border-zinc-800/50';

  const tierColors = {
    elite: {
      badge: 'bg-yellow-500/20 text-yellow-400 ring-1 ring-yellow-500/30',
      text: 'text-yellow-400',
    },
    strong: {
      badge: 'bg-cyan-500/20 text-cyan-400 ring-1 ring-cyan-500/30',
      text: 'text-cyan-400',
    },
    moderate: {
      badge: 'bg-zinc-700/50 text-zinc-300 ring-1 ring-zinc-600/30',
      text: 'text-zinc-400',
    },
    caution: {
      badge: 'bg-orange-500/20 text-orange-400 ring-1 ring-orange-500/30',
      text: 'text-orange-400',
    },
  }[pick.tier] || { badge: 'bg-zinc-700/50 text-zinc-300', text: 'text-zinc-400' };

  // Shared avatar component
  const Avatar = ({ size = 'normal' }: { size?: 'normal' | 'small' }) => {
    const dims = size === 'small' ? 'w-10 h-10' : 'w-12 h-12';
    const logoDims = size === 'small' ? 'w-4 h-4' : 'w-5 h-5';

    return (
      <div className="relative flex-shrink-0">
        <div className={`${dims} rounded-full bg-zinc-800 overflow-hidden ring-2 ring-zinc-700/50`}>
          {pick.headshot_url ? (
            <Image
              src={pick.headshot_url}
              alt={pick.player}
              width={48}
              height={48}
              className="w-full h-full object-cover"
              unoptimized
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-zinc-500 font-semibold text-sm bg-gradient-to-br from-zinc-700 to-zinc-800">
              {pick.player.split(' ').map(n => n[0]).join('')}
            </div>
          )}
        </div>
        <div className={`absolute -bottom-0.5 -right-0.5 ${logoDims} rounded-full bg-zinc-900 ring-1 ring-zinc-700 overflow-hidden shadow-sm`}>
          {logoError ? (
            <span className="w-full h-full flex items-center justify-center text-[6px] font-bold text-zinc-600 bg-zinc-200">{pick.team}</span>
          ) : (
            <img
              src={getTeamLogoUrl(pick.team)}
              alt={pick.team}
              className="w-full h-full object-contain"
              onError={() => setLogoError(true)}
            />
          )}
        </div>
      </div>
    );
  };

  return (
    <div
      className={`relative bg-zinc-900/80 rounded-xl border transition-all cursor-pointer hover:bg-zinc-900 ${
        isSelected
          ? 'border-emerald-500/50 ring-2 ring-emerald-500/20'
          : `${confidenceBorder} hover:border-zinc-600`
      }`}
      onClick={() => onAnalyze?.(pick)}
    >
      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-emerald-500 rounded-full flex items-center justify-center z-10">
          <svg className="w-3 h-3 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      )}

      {/* ========== MOBILE LAYOUT ========== */}
      <div className="md:hidden p-3">
        {/* Row 1: Avatar + Player info + Chart button */}
        <div className="flex items-center gap-2.5">
          <Avatar size="small" />

          <div className="flex-1 min-w-0">
            {/* Player vs Opponent */}
            <h3 className="font-semibold text-white text-sm truncate">
              {pick.player} <span className="text-zinc-500 font-normal">vs {pick.opponent}</span>
            </h3>
            {/* Line + Market */}
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className={`text-xs font-semibold ${isOver ? 'text-emerald-400' : 'text-blue-400'}`}>
                {isOver ? 'o' : 'u'}{pick.line}
              </span>
              <span className="text-xs text-zinc-500">{pick.market_display}</span>
            </div>
          </div>

          {/* Chart/Analyze button */}
          <button
            onClick={(e) => { e.stopPropagation(); onAnalyze?.(pick); }}
            className="flex-shrink-0 w-8 h-8 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400 hover:bg-zinc-700 hover:text-white"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          </button>
        </div>

        {/* Row 2: Projection + Edge + Stars */}
        <div className="flex items-center justify-between mt-2.5 px-1">
          <div className="flex items-center gap-2">
            <span className="text-zinc-400 text-xs">Proj.</span>
            <span className={`text-sm font-bold ${isOver ? 'text-emerald-400' : 'text-blue-400'}`}>
              {pick.projection.toFixed(1)}
            </span>
            <span className={`text-xs font-medium ${pick.edge >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              ({edgeSign}{pick.edge.toFixed(1)})
            </span>
          </div>
          <StarRating stars={pick.stars} />
        </div>

        {/* Row 3: Stats pills */}
        <div className="flex items-center gap-1.5 mt-2.5">
          {/* OPP Defense - what they allow to this position */}
          {pick.opp_def_rank && (
            <span className={`text-[10px] font-semibold px-2 py-1 rounded ${
              pick.opp_def_rank <= 8 ? 'bg-emerald-500/20 text-emerald-400' :
              pick.opp_def_rank >= 25 ? 'bg-red-500/20 text-red-400' :
              'bg-zinc-800 text-zinc-300'
            }`}>
              {pick.opp_def_allowed ? `${pick.opp_def_allowed.toFixed(1)} ` : ''}
              #{pick.opp_def_rank}
            </span>
          )}

          {/* L5 Rate */}
          {pick.l5_rate !== undefined && (
            <span className={`text-[10px] font-semibold px-2 py-1 rounded ${
              pick.l5_rate >= 60 ? 'bg-emerald-500/20 text-emerald-400' :
              pick.l5_rate >= 40 ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-red-500/20 text-red-400'
            }`}>
              {pick.l5_rate}% L-5
            </span>
          )}

          {/* Depth position */}
          {pick.depth_position && (
            <span className="text-[10px] font-semibold px-2 py-1 rounded bg-zinc-800 text-zinc-400">
              {pick.depth_position}
            </span>
          )}

          {/* Add to slip button */}
          <button
            onClick={(e) => { e.stopPropagation(); onSelect?.(pick); }}
            className={`ml-auto flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center transition-all ${
              isSelected
                ? 'bg-emerald-500 text-black'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
            }`}
          >
            {isSelected ? (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* ========== DESKTOP LAYOUT ========== */}
      <div className="hidden md:block p-4">
        {/* Top section */}
        <div className="flex items-start gap-3">
          <Avatar />

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-white truncate">{pick.player}</h3>
              <span className={`flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${tierColors.badge}`}>
                {pick.tier}
              </span>
            </div>
            <p className="text-xs text-zinc-500 mt-0.5">
              {pick.position}{pick.depth_position ? ` (${pick.depth_position})` : ''} • {pick.team} vs {pick.opponent}
            </p>
          </div>

          {/* Add button */}
          <button
            onClick={(e) => { e.stopPropagation(); onSelect?.(pick); }}
            className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center transition-all ${
              isSelected
                ? 'bg-emerald-500 text-black'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-white'
            }`}
          >
            {isSelected ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            )}
          </button>
        </div>

        {/* Prop line */}
        <div className="mt-4 p-3 rounded-lg bg-zinc-950/50 border border-zinc-800/30">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">{pick.market_display}</p>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center px-2.5 py-1 rounded text-sm font-bold ${
                  isOver ? 'bg-emerald-500/20 text-emerald-400' : 'bg-blue-500/20 text-blue-400'
                }`}>
                  {isOver ? 'OVER' : 'UNDER'} {pick.line}
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">Projection</p>
              <p className={`text-xl font-bold ${isOver ? 'text-emerald-400' : 'text-blue-400'}`}>
                {pick.projection.toFixed(1)}
              </p>
            </div>
          </div>

          {/* Edge bar */}
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${isOver ? 'bg-emerald-500' : 'bg-blue-500'}`}
                style={{ width: `${Math.min(Math.max((pick.confidence) * 100, 10), 100)}%` }}
              />
            </div>
            <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
              pick.confidence >= 0.7 ? 'bg-emerald-500/20 text-emerald-400' :
              pick.confidence >= 0.6 ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-zinc-700/50 text-zinc-400'
            }`}>
              {(pick.confidence * 100).toFixed(0)}%
            </span>
            <span className={`text-sm font-semibold min-w-[50px] text-right ${
              Math.abs(pick.edge) > 0.5 ? (isOver ? 'text-emerald-400' : 'text-blue-400') : 'text-zinc-500'
            }`}>
              {pick.edge > 0 ? '+' : ''}{pick.edge.toFixed(1)}
            </span>
          </div>
        </div>

        {/* Bottom stats */}
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          {pick.l5_rate !== undefined && (
            <span className={`text-xs font-medium px-2 py-1 rounded ${
              pick.l5_rate >= 60 ? 'bg-emerald-500/15 text-emerald-400' :
              pick.l5_rate >= 40 ? 'bg-yellow-500/15 text-yellow-400' :
              'bg-red-500/15 text-red-400'
            }`}>
              {pick.l5_rate}% L5
            </span>
          )}
          {pick.opp_def_rank && (
            <span className={`text-xs font-medium px-2 py-1 rounded ${
              pick.opp_def_rank <= 8 ? 'bg-emerald-500/15 text-emerald-400' :
              pick.opp_def_rank >= 25 ? 'bg-red-500/15 text-red-400' :
              'bg-zinc-700 text-zinc-300'
            }`}>
              {pick.opp_def_allowed ? `${pick.opp_def_allowed.toFixed(1)} ` : ''}
              (#{pick.opp_def_rank} vs {pick.position})
            </span>
          )}
          {pick.ev > 0 && (
            <span className="text-xs font-medium text-emerald-400 px-2 py-1 rounded bg-emerald-500/15">
              +{pick.ev.toFixed(0)}% EV
            </span>
          )}
          <span className="text-xs text-zinc-500 ml-auto">
            {pick.hist_count} games
          </span>
        </div>
      </div>
    </div>
  );
}
