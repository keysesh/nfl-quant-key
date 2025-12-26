'use client';

import { Pick } from '@/lib/types';
import Image from 'next/image';
import { useEffect } from 'react';

interface PickModalProps {
  pick: Pick | null;
  onClose: () => void;
}

function StarRating({ stars }: { stars: number }) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((i) => (
        <span key={i} className={`text-lg ${i <= stars ? 'star-filled' : 'star-empty'}`}>
          ★
        </span>
      ))}
    </div>
  );
}

function GameHistoryChart({ pick }: { pick: Pick }) {
  // Get relevant stat based on market
  const getStatData = () => {
    const history = pick.game_history;
    const market = pick.market;
    if (market.includes('pass_yds')) return history.passing_yards;
    if (market.includes('pass_att') || market.includes('pass_attempts')) return history.passing_attempts;
    if (market.includes('pass_comp') || market.includes('completions')) return history.completions;
    if (market.includes('rush_yds')) return history.rushing_yards;
    if (market.includes('rush_att')) return history.rushing_attempts;
    if (market.includes('reception_yds') || market.includes('rec_yds')) return history.receiving_yards;
    if (market.includes('receptions')) return history.receptions;
    if (market.includes('td')) {
      return history.weeks.map((_, i) =>
        (history.passing_tds[i] || 0) + (history.rushing_tds[i] || 0) + (history.receiving_tds[i] || 0)
      );
    }
    return history.passing_yards;
  };

  const playerData = getStatData() || [];
  const playerWeeks = pick.game_history.weeks || [];

  if (playerWeeks.length === 0 || playerData.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-zinc-500">
        No game history available
      </div>
    );
  }

  // Defense data
  const defenseData = pick.game_history.defense_allowed || [];
  const defenseWeeks = pick.game_history.defense_weeks || [];
  const defenseOpponent = pick.game_history.defense_opponent || pick.opponent;
  const hasDefenseData = defenseData.length > 0 && defenseData.some(d => d !== null);

  // Calculate max for scaling (include defense data if available)
  const allValues = [...playerData, pick.line];
  if (hasDefenseData) {
    allValues.push(...defenseData.filter((d): d is number => d !== null));
  }
  const maxVal = Math.max(...allValues) * 1.15;

  // Determine colors based on our pick direction
  const isOverPick = pick.pick === 'OVER';

  // Build comparison data - align by recency (most recent first)
  const numGames = Math.max(playerWeeks.length, defenseWeeks.length);

  // Reverse to show most recent first
  const playerReversed = [...playerData].reverse();
  const playerWeeksReversed = [...playerWeeks].reverse();
  const playerOpponents = [...(pick.game_history.opponents || [])].reverse();
  const defenseReversed = [...defenseData].reverse();
  const defenseWeeksReversed = [...defenseWeeks].reverse();
  const defenseOpponents = [...(pick.game_history.defense_opponents || [])].reverse();

  // Combined max for consistent scaling
  const allVals = [...playerData, ...defenseData.filter((d): d is number => d !== null), pick.line];
  const chartMax = Math.max(...allVals) * 1.15;

  return (
    <div className="space-y-2">
      {/* Legend */}
      <div className="flex items-center justify-between text-[11px]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-gradient-to-t from-emerald-600 to-emerald-400" />
            <span className="text-zinc-400">{pick.pick}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2.5 h-2.5 rounded-sm bg-gradient-to-t from-red-600 to-red-400" />
            <span className="text-zinc-400">{pick.pick === 'OVER' ? 'UNDER' : 'OVER'}</span>
          </div>
          {hasDefenseData && (
            <div className="flex items-center gap-1">
              <div className="w-2.5 h-2.5 rounded-sm bg-gradient-to-t from-purple-600 to-purple-400" />
              <span className="text-zinc-400">{defenseOpponent} DEF</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 border-t-2 border-dashed border-yellow-500" />
          <span className="text-yellow-500 font-medium">{pick.line}</span>
        </div>
      </div>

      {/* Integrated Chart */}
      <div className="relative h-48">
        {/* Line reference */}
        <div
          className="absolute left-0 right-0 border-t-2 border-dashed border-yellow-500/50 z-10"
          style={{ top: `${Math.max(8, Math.min(85, 100 - (pick.line / chartMax) * 100))}%` }}
        />

        {/* Two-section chart */}
        <div className="flex h-full">
          {/* Player Section */}
          <div className="flex-1 border-r border-white/10 pr-1">
            <div className="text-[10px] text-zinc-500 text-center mb-1 font-medium">{pick.player.split(' ')[1] || pick.player}</div>
            <div className="flex items-end justify-around h-[calc(100%-20px)] gap-0.5">
              {playerWeeksReversed.slice(0, 6).map((week, i) => {
                const stat = playerReversed[i] || 0;
                const wentOver = stat > pick.line;
                const matchesOurPick = isOverPick ? wentOver : !wentOver;
                const heightPct = (stat / chartMax) * 100;
                const opp = playerOpponents[i] || '';

                return (
                  <div key={`p-${week}`} className="flex-1 flex flex-col items-center h-full justify-end">
                    <span className={`text-[10px] font-bold ${matchesOurPick ? 'text-emerald-400' : 'text-red-400'}`}>
                      {stat}
                    </span>
                    <div
                      className={`w-full max-w-[20px] rounded-t ${
                        matchesOurPick
                          ? 'bg-gradient-to-t from-emerald-600 to-emerald-400'
                          : 'bg-gradient-to-t from-red-600 to-red-400'
                      }`}
                      style={{ height: `${Math.max(heightPct, 3)}%`, minHeight: '4px' }}
                    />
                    <span className="text-[9px] text-zinc-500 mt-0.5">W{week}</span>
                    <span className="text-[8px] text-zinc-600">{opp}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Defense Section */}
          {hasDefenseData && (
            <div className="flex-1 pl-1">
              <div className="text-[10px] text-zinc-500 text-center mb-1 font-medium">{defenseOpponent} vs {pick.position}</div>
              <div className="flex items-end justify-around h-[calc(100%-20px)] gap-0.5">
                {defenseWeeksReversed.slice(0, 6).map((week, i) => {
                  const allowed = defenseReversed[i];
                  if (allowed === null) return null;
                  const aboveLine = allowed > pick.line;
                  const heightPct = (allowed / chartMax) * 100;
                  const vsTeam = defenseOpponents[i] || '';

                  return (
                    <div key={`d-${week}`} className="flex-1 flex flex-col items-center h-full justify-end">
                      <span className={`text-[10px] font-bold ${aboveLine ? 'text-purple-300' : 'text-purple-400/70'}`}>
                        {allowed.toFixed(0)}
                      </span>
                      <div
                        className={`w-full max-w-[20px] rounded-t ${
                          aboveLine
                            ? 'bg-gradient-to-t from-purple-600 to-purple-400'
                            : 'bg-gradient-to-t from-purple-700/60 to-purple-500/60'
                        }`}
                        style={{ height: `${Math.max(heightPct, 3)}%`, minHeight: '4px' }}
                      />
                      <span className="text-[9px] text-zinc-500 mt-0.5">W{week}</span>
                      <span className="text-[8px] text-zinc-600">{vsTeam}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function PickModal({ pick, onClose }: PickModalProps) {
  if (!pick) return null;

  const isOver = pick.pick === 'OVER';
  const edgeColor = pick.edge > 0 ? 'text-emerald-400' : 'text-red-400';
  const edgeSign = pick.edge > 0 ? '+' : '';

  // Handle escape key and lock body scroll
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    // Lock body scroll when modal is open
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    window.addEventListener('keydown', handleEscape);
    return () => {
      window.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = originalOverflow;
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-md"
        onClick={onClose}
      />

      {/* Modal - Glass design */}
      <div
        className="relative w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-t-2xl sm:rounded-2xl border border-white/[0.08] shadow-2xl"
        style={{
          background: 'linear-gradient(180deg, rgba(12, 14, 20, 0.98) 0%, rgba(7, 8, 12, 0.99) 100%)',
          backdropFilter: 'blur(24px) saturate(180%)',
          WebkitBackdropFilter: 'blur(24px) saturate(180%)',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
        }}
      >
        {/* Header - Glass panel */}
        <div
          className="sticky top-0 z-10 border-b border-white/[0.06] p-4"
          style={{
            background: 'linear-gradient(180deg, rgba(16, 18, 24, 0.95) 0%, rgba(12, 14, 20, 0.9) 100%)',
            backdropFilter: 'blur(12px)',
          }}
        >
          <div className="flex items-start gap-3">
            {/* Avatar */}
            <div className="relative flex-shrink-0">
              <div
                className="w-16 h-16 rounded-full overflow-hidden ring-2 ring-white/10"
                style={{ background: 'linear-gradient(135deg, rgba(39,39,42,0.8), rgba(24,24,27,0.9))' }}
              >
                {pick.headshot_url ? (
                  <Image
                    src={pick.headshot_url}
                    alt={pick.player}
                    width={64}
                    height={64}
                    className="w-full h-full object-cover"
                    unoptimized
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-400 font-bold text-xl">
                    {pick.player.split(' ').map(n => n[0]).join('')}
                  </div>
                )}
              </div>
              <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-zinc-900/90 ring-2 ring-white/10 overflow-hidden">
                <Image
                  src={pick.team_logo_url}
                  alt={pick.team}
                  width={28}
                  height={28}
                  className="w-full h-full object-contain"
                  unoptimized
                />
              </div>
            </div>

            {/* Info */}
            <div className="flex-1">
              <h2 className="text-xl font-bold text-white">{pick.player}</h2>
              <p className="text-zinc-400">
                {pick.position}
                {pick.depth_position && <span className="text-zinc-500"> ({pick.depth_position})</span>}
                {' · '}{pick.team} vs {pick.opponent}
              </p>
              <p className="text-sm text-zinc-500 mt-1">{pick.game}</p>
            </div>

            {/* Close button */}
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/[0.08] transition-all"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Pick summary - Glass panel */}
          <div
            className="rounded-xl p-4 border border-white/[0.06]"
            style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
          >
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-sm text-zinc-400 mb-1">{pick.market_display}</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl font-bold px-3 py-1 rounded-lg ${
                    isOver
                      ? 'text-emerald-400 bg-emerald-500/15 border border-emerald-500/30'
                      : 'text-blue-400 bg-blue-500/15 border border-blue-500/30'
                  }`}>
                    {isOver ? 'OVER' : 'UNDER'} {pick.line}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-zinc-400 mb-1">Projection</p>
                <p className={`text-2xl font-bold ${edgeColor}`}>
                  {pick.projection.toFixed(1)}
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-white/[0.06]">
              <div className="flex items-center gap-4">
                <div>
                  <p className="text-xs text-zinc-500">Edge</p>
                  <p className={`font-bold ${edgeColor}`}>{edgeSign}{pick.edge.toFixed(1)}</p>
                </div>
                <div>
                  <p className="text-xs text-zinc-500">Confidence</p>
                  <p className="font-bold text-white">{(pick.confidence * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <p className="text-xs text-zinc-500">EV</p>
                  <p className={`font-bold ${pick.ev > 0 ? 'text-emerald-400' : 'text-zinc-400'}`}>
                    {pick.ev > 0 ? '+' : ''}{pick.ev.toFixed(1)}%
                  </p>
                </div>
              </div>
              <StarRating stars={pick.stars} />
            </div>
          </div>

          {/* Matchup Context - Defense vs Position */}
          {(pick.opp_def_allowed || pick.depth_position) && (
            <div
              className="rounded-xl p-4 border border-white/[0.06]"
              style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
            >
              <h3 className="text-sm font-semibold text-zinc-300 mb-3">Matchup Context</h3>
              <div className="space-y-3">
                {/* Depth Position */}
                {pick.depth_position && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-zinc-400">Depth Chart</span>
                    <span className="text-sm font-semibold text-white bg-white/[0.06] border border-white/[0.08] px-2 py-0.5 rounded">
                      {pick.depth_position}
                    </span>
                  </div>
                )}

                {/* Defense vs Position */}
                {pick.opp_def_allowed && pick.opp_def_rank && (
                  <div className="bg-black/30 rounded-lg p-3 border border-white/[0.04]">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-zinc-400">{pick.opponent} vs {pick.position}s</span>
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded border ${
                        pick.opp_def_rank <= 8 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' :
                        pick.opp_def_rank >= 25 ? 'bg-red-500/15 text-red-400 border-red-500/30' :
                        'bg-white/[0.04] text-zinc-300 border-white/[0.08]'
                      }`}>
                        #{pick.opp_def_rank} {pick.opp_def_rank <= 8 ? '(Soft)' : pick.opp_def_rank >= 25 ? '(Tough)' : ''}
                      </span>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-2xl font-bold text-white">{pick.opp_def_allowed.toFixed(1)}</span>
                      <span className="text-sm text-zinc-400">
                        {pick.market === 'player_pass_yds' ? 'pass yds' :
                         pick.market === 'player_pass_attempts' ? 'pass att' :
                         pick.market === 'player_pass_completions' ? 'completions' :
                         pick.market === 'player_pass_tds' ? 'pass TDs' :
                         pick.market === 'player_rush_yds' ? 'rush yds' :
                         pick.market === 'player_rush_attempts' ? 'rush att' :
                         pick.market === 'player_reception_yds' ? 'rec yds' :
                         pick.market === 'player_receptions' ? 'receptions' :
                         pick.market === 'player_anytime_td' ? 'TDs' : 'stat'}/game allowed
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 mt-1">
                      {pick.opp_def_rank <= 8
                        ? `${pick.opponent} is a favorable matchup for ${pick.position}s`
                        : pick.opp_def_rank >= 25
                        ? `${pick.opponent} is a tough matchup for ${pick.position}s`
                        : `${pick.opponent} is average against ${pick.position}s`
                      }
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Game History Chart */}
          <div
            className="rounded-xl p-4 border border-white/[0.06]"
            style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
          >
            <h3 className="text-sm font-semibold text-zinc-300 mb-3">Recent Performance</h3>
            <GameHistoryChart pick={pick} />
          </div>

          {/* Game History Table */}
          {pick.game_history.weeks.length > 0 && (
            <div
              className="rounded-xl overflow-hidden border border-white/[0.06]"
              style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
            >
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-black/30">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-zinc-400">Week</th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-zinc-400">Opp</th>
                      <th className="px-3 py-2 text-right text-xs font-semibold text-zinc-400">Result</th>
                      <th className="px-3 py-2 text-center text-xs font-semibold text-zinc-400">Hit?</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {pick.game_history.weeks.map((week, i) => {
                      let stat = 0;
                      const market = pick.market;
                      if (market.includes('pass_yds')) stat = pick.game_history.passing_yards[i] || 0;
                      else if (market.includes('pass_att') || market.includes('pass_attempts')) stat = pick.game_history.passing_attempts[i] || 0;
                      else if (market.includes('pass_comp') || market.includes('completions')) stat = pick.game_history.completions[i] || 0;
                      else if (market.includes('rush_yds')) stat = pick.game_history.rushing_yards[i] || 0;
                      else if (market.includes('rush_att')) stat = pick.game_history.rushing_attempts[i] || 0;
                      else if (market.includes('reception_yds') || market.includes('rec_yds')) stat = pick.game_history.receiving_yards[i] || 0;
                      else if (market.includes('receptions')) stat = pick.game_history.receptions[i] || 0;
                      else if (market.includes('td')) {
                        stat = (pick.game_history.passing_tds[i] || 0) +
                               (pick.game_history.rushing_tds[i] || 0) +
                               (pick.game_history.receiving_tds[i] || 0);
                      }

                      const hit = pick.pick === 'OVER' ? stat > pick.line : stat < pick.line;

                      return (
                        <tr key={week} className="hover:bg-white/[0.02]">
                          <td className="px-3 py-2 text-zinc-300">W{week}</td>
                          <td className="px-3 py-2 text-zinc-400">{pick.game_history.opponents[i]}</td>
                          <td className="px-3 py-2 text-right font-mono text-white">{stat}</td>
                          <td className="px-3 py-2 text-center">
                            <span className={`w-5 h-5 inline-flex items-center justify-center rounded-md text-xs border ${
                              hit
                                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                                : 'bg-red-500/15 text-red-400 border-red-500/30'
                            }`}>
                              {hit ? '✓' : '✗'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Historical stats */}
          <div className="grid grid-cols-2 gap-3">
            <div
              className="rounded-xl p-3 border border-white/[0.06]"
              style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
            >
              <p className="text-xs text-zinc-500 mb-1">Historical Over Rate</p>
              <p className="text-lg font-bold text-white">{(pick.hist_over_rate * 100).toFixed(0)}%</p>
              <p className="text-xs text-zinc-500">{pick.hist_count} games</p>
            </div>
            {pick.opp_rank && (
              <div
                className="rounded-xl p-3 border border-white/[0.06]"
                style={{ background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)' }}
              >
                <p className="text-xs text-zinc-500 mb-1">vs {pick.opponent} Defense</p>
                <p className="text-lg font-bold text-white">{pick.opp_rank}th</p>
                <p className="text-xs text-zinc-500">in league</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer - Glass panel */}
        <div
          className="sticky bottom-0 border-t border-white/[0.06] p-4"
          style={{
            background: 'linear-gradient(180deg, rgba(12, 14, 20, 0.95) 0%, rgba(7, 8, 12, 0.98) 100%)',
            backdropFilter: 'blur(12px)',
          }}
        >
          <button
            onClick={onClose}
            className="w-full py-3 rounded-xl font-semibold text-white transition-all bg-white/[0.06] border border-white/[0.1] hover:bg-white/[0.1]"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
