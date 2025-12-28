'use client';

import React, { useState } from 'react';
import { Pick } from '@/lib/types';
import Image from 'next/image';

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

// Game context badges - shows vegas lines and weather factors (inline)
function GameContextBadges({ pick }: { pick: Pick }) {
  const badges: React.ReactNode[] = [];
  const { vegas_total, vegas_spread, roof, temp, wind, market } = pick;

  const isOutdoor = roof !== 'dome' && roof !== 'closed';
  const isPassingMarket = market.includes('pass') || market.includes('reception') || market.includes('rec');

  // High scoring game indicator
  if (vegas_total && vegas_total >= 47) {
    badges.push(
      <span key="total" className="text-xs font-medium px-2 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
        O/U {vegas_total}
      </span>
    );
  }

  // Spread indicator (shows game script risk)
  if (vegas_spread !== null && vegas_spread !== undefined) {
    const isFavored = vegas_spread < 0;
    const isBlowout = Math.abs(vegas_spread) >= 7;
    badges.push(
      <span key="spread" className={`text-xs font-medium px-2 py-1 rounded-md border ${
        isBlowout
          ? isFavored
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-orange-500/10 border-orange-500/20 text-orange-400'
          : 'bg-zinc-500/10 border-zinc-500/20 text-zinc-400'
      }`}>
        {isFavored ? 'FAV' : 'DOG'} {vegas_spread > 0 ? '+' : ''}{vegas_spread.toFixed(1)}
      </span>
    );
  }

  // Wind warning for passing games
  if (isOutdoor && wind && wind >= 15 && isPassingMarket) {
    badges.push(
      <span key="wind" className="text-xs font-medium px-2 py-1 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-400">
        {wind}mph wind
      </span>
    );
  }

  // Cold weather
  if (isOutdoor && temp !== null && temp !== undefined && temp <= 35) {
    badges.push(
      <span key="temp" className="text-xs font-medium px-2 py-1 rounded-md bg-sky-500/10 border border-sky-500/20 text-sky-400">
        {temp}°F
      </span>
    );
  }

  if (badges.length === 0) return null;

  return <>{badges}</>;
}

interface PlayerCardProps {
  pick: Pick;
  isSelected?: boolean;
  onSelect?: (pick: Pick) => void;
  onAnalyze?: (pick: Pick) => void;
}

export default function PlayerCard({ pick, onAnalyze }: PlayerCardProps) {
  const [logoError, setLogoError] = useState(false);
  const isOver = pick.pick === 'OVER';
  const edgeSign = pick.edge >= 0 ? '+' : '';

  const tierColors = {
    elite: {
      badge: 'bg-gradient-to-r from-yellow-500/25 to-yellow-500/15 text-yellow-400 border border-yellow-500/30 shadow-[0_0_20px_rgba(234,179,8,0.12)]',
      text: 'text-yellow-400',
      glow: 'shadow-[0_0_30px_rgba(234,179,8,0.15)]',
    },
    strong: {
      badge: 'bg-gradient-to-r from-cyan-500/25 to-cyan-500/15 text-cyan-400 border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.12)]',
      text: 'text-cyan-400',
      glow: 'shadow-[0_0_30px_rgba(6,182,212,0.15)]',
    },
    moderate: {
      badge: 'bg-white/[0.04] text-zinc-300 border border-white/[0.08]',
      text: 'text-zinc-400',
      glow: '',
    },
    caution: {
      badge: 'bg-gradient-to-r from-orange-500/25 to-orange-500/15 text-orange-400 border border-orange-500/30',
      text: 'text-orange-400',
      glow: '',
    },
  }[pick.tier] || { badge: 'bg-white/[0.04] text-zinc-300', text: 'text-zinc-400', glow: '' };

  // Shared avatar component
  const Avatar = ({ size = 'normal' }: { size?: 'normal' | 'small' }) => {
    const dims = size === 'small' ? 'w-10 h-10' : 'w-12 h-12';
    const logoDims = size === 'small' ? 'w-4 h-4' : 'w-5 h-5';

    return (
      <div className="relative flex-shrink-0">
        <div className={`${dims} rounded-full overflow-hidden ring-2 ring-white/10 shadow-lg`}
          style={{ background: 'linear-gradient(135deg, rgba(39,39,42,0.8), rgba(24,24,27,0.9))' }}>
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
            <div className="w-full h-full flex items-center justify-center text-zinc-400 font-semibold text-sm bg-gradient-to-br from-zinc-700/50 to-zinc-800/50 backdrop-blur-sm">
              {pick.player.split(' ').map(n => n[0]).join('')}
            </div>
          )}
        </div>
        <div className={`absolute -bottom-0.5 -right-0.5 ${logoDims} rounded-full bg-zinc-900/90 ring-1 ring-white/10 overflow-hidden shadow-md backdrop-blur-sm`}>
          {logoError ? (
            <span className="w-full h-full flex items-center justify-center text-[6px] font-bold text-zinc-500 bg-zinc-800">{pick.team}</span>
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
      className={`relative rounded-2xl border transition-all duration-200 cursor-pointer backdrop-blur-xl border-white/[0.06] hover:border-white/[0.12] hover:shadow-[0_8px_32px_rgba(0,0,0,0.3)] ${tierColors.glow}`}
      style={{
        background: 'linear-gradient(135deg, rgba(24,26,32,0.8) 0%, rgba(16,18,24,0.6) 100%)',
      }}
      onClick={() => onAnalyze?.(pick)}
    >

      {/* ========== MOBILE LAYOUT ========== */}
      <div className="md:hidden p-3">
        {/* Row 1: Avatar + Player info + Tier */}
        <div className="flex items-center gap-2">
          <Avatar size="small" />

          <div className="flex-1 min-w-0">
            {/* Player name + Position */}
            <div className="flex items-center gap-1.5">
              <h3 className="font-bold text-white text-[15px] truncate">{pick.player}</h3>
              {pick.depth_position && (
                <span className="text-[10px] text-zinc-500">{pick.depth_position}</span>
              )}
            </div>
            {/* Team vs Opponent + Market */}
            <div className="flex items-center gap-1 mt-0.5">
              <span className="text-xs text-zinc-400">{pick.team} vs {pick.opponent}</span>
              <span className="text-zinc-600">•</span>
              <span className="text-xs text-zinc-500">{pick.market_display}</span>
            </div>
          </div>

          {/* Tier badge */}
          <span className={`flex-shrink-0 px-2 py-0.5 rounded text-[10px] font-bold uppercase ${tierColors.badge}`}>
            {pick.tier}
          </span>
        </div>

        {/* Row 2: Prop line box - Glass inner panel */}
        <div className="mt-2.5 p-2.5 rounded-xl bg-black/30 border border-white/[0.04] backdrop-blur-sm">
          <div className="flex items-center justify-between">
            {/* Pick direction + Line */}
            <div className="flex items-center gap-2">
              <span className={`px-2.5 py-1 rounded-lg text-sm font-bold ${
                isOver
                  ? 'bg-gradient-to-r from-emerald-500/25 to-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                  : 'bg-gradient-to-r from-blue-500/25 to-blue-500/15 text-blue-400 border border-blue-500/30'
              }`}>
                {isOver ? 'OVER' : 'UNDER'} {pick.line}
              </span>
              <div className="flex items-center gap-1">
                <span className="text-zinc-600 text-xs">→</span>
                <span className={`text-lg font-bold ${isOver ? 'text-emerald-400' : 'text-blue-400'}`}>
                  {pick.projection.toFixed(1)}
                </span>
              </div>
            </div>

            {/* Confidence + Edge */}
            <div className="text-right">
              <div className={`text-lg font-bold ${
                pick.confidence >= 0.7 ? 'text-emerald-400' :
                pick.confidence >= 0.6 ? 'text-yellow-400' : 'text-zinc-400'
              }`}>
                {(pick.confidence * 100).toFixed(0)}%
              </div>
              <div className={`text-xs font-medium ${pick.edge >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {edgeSign}{pick.edge.toFixed(1)} edge
              </div>
            </div>
          </div>
        </div>

        {/* Row 3: Stats row */}
        <div className="flex items-center justify-between mt-2">
          {/* Left: Defense + L5 + Weather/Game Script */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {/* OPP Defense */}
            {pick.opp_def_rank && (
              <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                pick.opp_def_rank <= 8 ? 'bg-emerald-500/20 text-emerald-400' :
                pick.opp_def_rank >= 25 ? 'bg-red-500/20 text-red-400' :
                'bg-zinc-800 text-zinc-300'
              }`}>
                {pick.opp_def_allowed ? `${pick.opp_def_allowed.toFixed(0)} avg ` : ''}#{pick.opp_def_rank}
              </span>
            )}

            {/* L5 Rate */}
            {pick.l5_rate !== undefined && (
              <span className={`text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                pick.l5_rate >= 60 ? 'bg-emerald-500/20 text-emerald-400' :
                pick.l5_rate >= 40 ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-red-500/20 text-red-400'
              }`}>
                {pick.l5_rate}% L5
              </span>
            )}

            {/* Game context badges inline */}
            <GameContextBadges pick={pick} />
          </div>

          {/* Right: Stars + Chart button */}
          <div className="flex items-center gap-2">
            <StarRating stars={pick.stars} />
            <button
              onClick={(e) => { e.stopPropagation(); onAnalyze?.(pick); }}
              className="w-7 h-7 rounded-lg bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-zinc-400 hover:bg-white/[0.08] hover:text-white transition-all"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            </button>
          </div>
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
        </div>

        {/* Prop line - Glass inner panel */}
        <div className="mt-4 p-3 rounded-xl bg-black/30 border border-white/[0.04] backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">{pick.market_display}</p>
              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center px-2.5 py-1 rounded-lg text-sm font-bold ${
                  isOver
                    ? 'bg-gradient-to-r from-emerald-500/25 to-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                    : 'bg-gradient-to-r from-blue-500/25 to-blue-500/15 text-blue-400 border border-blue-500/30'
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

          {/* Edge bar - Glass style */}
          <div className="mt-3 flex items-center gap-3">
            <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  isOver
                    ? 'bg-gradient-to-r from-emerald-500 to-emerald-400'
                    : 'bg-gradient-to-r from-blue-500 to-blue-400'
                }`}
                style={{ width: `${Math.min(Math.max((pick.confidence) * 100, 10), 100)}%` }}
              />
            </div>
            <span className={`text-xs font-medium px-2 py-0.5 rounded-md ${
              pick.confidence >= 0.7 ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20' :
              pick.confidence >= 0.6 ? 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/20' :
              'bg-white/[0.04] text-zinc-400 border border-white/[0.06]'
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

        {/* Bottom stats - Glass pills */}
        <div className="mt-3 flex items-center gap-2 flex-wrap">
          {pick.l5_rate !== undefined && (
            <span className={`text-xs font-medium px-2 py-1 rounded-md border ${
              pick.l5_rate >= 60 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
              pick.l5_rate >= 40 ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
              'bg-red-500/10 text-red-400 border-red-500/20'
            }`}>
              {pick.l5_rate}% L5
            </span>
          )}
          {pick.opp_def_rank && (
            <span className={`text-xs font-medium px-2 py-1 rounded-md border ${
              pick.opp_def_rank <= 8 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
              pick.opp_def_rank >= 25 ? 'bg-red-500/10 text-red-400 border-red-500/20' :
              'bg-white/[0.03] text-zinc-300 border-white/[0.06]'
            }`}>
              {pick.opp_def_allowed ? `${pick.opp_def_allowed.toFixed(1)} ` : ''}
              (#{pick.opp_def_rank} vs {pick.position})
            </span>
          )}
          {pick.ev > 0 && (
            <span className="text-xs font-medium text-emerald-400 px-2 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20">
              +{pick.ev.toFixed(0)}% EV
            </span>
          )}

          {/* Game context badges inline */}
          <GameContextBadges pick={pick} />

          <span className="text-xs text-zinc-500 ml-auto">
            {pick.hist_count} games
          </span>
        </div>
      </div>
    </div>
  );
}
