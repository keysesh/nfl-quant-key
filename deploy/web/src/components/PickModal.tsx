'use client';

import { Pick } from '@/lib/types';
import Image from 'next/image';
import { useEffect, useRef } from 'react';

interface PickModalProps {
  pick: Pick | null;
  onClose: () => void;
  onAddToSlip?: (pick: Pick) => void;
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
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
      // Combine all TDs
      return history.weeks.map((_, i) =>
        (history.passing_tds[i] || 0) + (history.rushing_tds[i] || 0) + (history.receiving_tds[i] || 0)
      );
    }
    return history.passing_yards; // default
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = getStatData();
    if (!data || data.length === 0) return;

    // Clear canvas
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Calculate scales
    const maxVal = Math.max(...data, pick.line) * 1.1;
    const minVal = Math.min(...data, pick.line) * 0.9;
    const range = maxVal - minVal;

    const xScale = chartWidth / (data.length - 1 || 1);
    const yScale = chartHeight / range;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartHeight / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Draw line (the betting line)
    const lineY = padding.top + (maxVal - pick.line) * yScale;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, lineY);
    ctx.lineTo(width - padding.right, lineY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw area under curve
    ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    data.forEach((val, i) => {
      const x = padding.left + i * xScale;
      const y = padding.top + (maxVal - val) * yScale;
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(padding.left + (data.length - 1) * xScale, padding.top + chartHeight);
    ctx.closePath();
    ctx.fill();

    // Draw line chart
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = padding.left + i * xScale;
      const y = padding.top + (maxVal - val) * yScale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw dots
    data.forEach((val, i) => {
      const x = padding.left + i * xScale;
      const y = padding.top + (maxVal - val) * yScale;
      const hitLine = pick.pick === 'OVER' ? val > pick.line : val < pick.line;

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = hitLine ? '#10b981' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#0f1729';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Y-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = maxVal - (range / 4) * i;
      const y = padding.top + (chartHeight / 4) * i;
      ctx.fillText(val.toFixed(0), padding.left - 8, y + 4);
    }

    // X-axis labels (weeks)
    ctx.textAlign = 'center';
    pick.game_history.weeks.forEach((week, i) => {
      const x = padding.left + i * xScale;
      ctx.fillText(`W${week}`, x, height - 10);
    });
  }, [pick]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="w-full h-48"
        style={{ width: '100%', height: '192px' }}
      />
      <div className="absolute top-2 right-2 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-red-500/50" style={{ borderStyle: 'dashed' }} />
          <span className="text-slate-400">Line ({pick.line})</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-slate-400">Hit</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-red-500" />
          <span className="text-slate-400">Miss</span>
        </div>
      </div>
    </div>
  );
}

export default function PickModal({ pick, onClose, onAddToSlip }: PickModalProps) {
  if (!pick) return null;

  const isOver = pick.pick === 'OVER';
  const edgeColor = pick.edge > 0 ? 'text-emerald-400' : 'text-red-400';
  const edgeSign = pick.edge > 0 ? '+' : '';

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg max-h-[90vh] overflow-y-auto bg-[#0a0f1c] rounded-t-2xl sm:rounded-2xl border border-slate-700/50 shadow-2xl animate-slide-up">
        {/* Header */}
        <div className="sticky top-0 bg-[#0a0f1c] border-b border-slate-700/50 p-4">
          <div className="flex items-start gap-3">
            {/* Avatar */}
            <div className="relative flex-shrink-0">
              <div className="w-16 h-16 rounded-full bg-slate-700 overflow-hidden">
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
                  <div className="w-full h-full flex items-center justify-center text-slate-400 font-bold text-xl">
                    {pick.player.split(' ').map(n => n[0]).join('')}
                  </div>
                )}
              </div>
              <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-slate-800 border-2 border-[#0a0f1c] overflow-hidden">
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
              <p className="text-slate-400">{pick.position} · {pick.team} vs {pick.opponent}</p>
              <p className="text-sm text-slate-500 mt-1">{pick.game}</p>
            </div>

            {/* Close button */}
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Pick summary */}
          <div className="bg-slate-800/50 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-sm text-slate-400 mb-1">{pick.market_display}</p>
                <div className="flex items-center gap-2">
                  <span className={`text-2xl font-bold ${isOver ? 'text-emerald-400' : 'text-blue-400'}`}>
                    {isOver ? 'OVER' : 'UNDER'} {pick.line}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-slate-400 mb-1">Projection</p>
                <p className={`text-2xl font-bold ${edgeColor}`}>
                  {pick.projection.toFixed(1)}
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-700/50">
              <div className="flex items-center gap-4">
                <div>
                  <p className="text-xs text-slate-500">Edge</p>
                  <p className={`font-bold ${edgeColor}`}>{edgeSign}{pick.edge.toFixed(1)}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Confidence</p>
                  <p className="font-bold text-white">{(pick.confidence * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">EV</p>
                  <p className={`font-bold ${pick.ev > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                    {pick.ev > 0 ? '+' : ''}{pick.ev.toFixed(1)}%
                  </p>
                </div>
              </div>
              <StarRating stars={pick.stars} />
            </div>
          </div>

          {/* Game History Chart */}
          <div className="bg-slate-800/50 rounded-xl p-4">
            <h3 className="text-sm font-semibold text-slate-300 mb-3">Recent Performance</h3>
            {pick.game_history.weeks.length > 0 ? (
              <GameHistoryChart pick={pick} />
            ) : (
              <div className="h-48 flex items-center justify-center text-slate-500">
                No game history available
              </div>
            )}
          </div>

          {/* Game History Table */}
          {pick.game_history.weeks.length > 0 && (
            <div className="bg-slate-800/50 rounded-xl overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900/50">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400">Week</th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-slate-400">Opp</th>
                      <th className="px-3 py-2 text-right text-xs font-semibold text-slate-400">Result</th>
                      <th className="px-3 py-2 text-center text-xs font-semibold text-slate-400">Hit?</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/30">
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
                        <tr key={week} className="hover:bg-slate-800/30">
                          <td className="px-3 py-2 text-slate-300">W{week}</td>
                          <td className="px-3 py-2 text-slate-400">{pick.game_history.opponents[i]}</td>
                          <td className="px-3 py-2 text-right font-mono text-white">{stat}</td>
                          <td className="px-3 py-2 text-center">
                            <span className={`w-5 h-5 inline-flex items-center justify-center rounded-full text-xs ${
                              hit ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
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
            <div className="bg-slate-800/50 rounded-xl p-3">
              <p className="text-xs text-slate-500 mb-1">Historical Over Rate</p>
              <p className="text-lg font-bold text-white">{(pick.hist_over_rate * 100).toFixed(0)}%</p>
              <p className="text-xs text-slate-500">{pick.hist_count} games</p>
            </div>
            {pick.opp_rank && (
              <div className="bg-slate-800/50 rounded-xl p-3">
                <p className="text-xs text-slate-500 mb-1">vs {pick.opponent} Defense</p>
                <p className="text-lg font-bold text-white">{pick.opp_rank}th</p>
                <p className="text-xs text-slate-500">in league</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[#0a0f1c] border-t border-slate-700/50 p-4">
          <button
            onClick={() => onAddToSlip?.(pick)}
            className={`w-full py-3 rounded-xl font-semibold text-white transition-colors ${
              isOver
                ? 'bg-emerald-500 hover:bg-emerald-600'
                : 'bg-blue-500 hover:bg-blue-600'
            }`}
          >
            Add {pick.pick} {pick.line} to Bet Slip
          </button>
        </div>
      </div>
    </div>
  );
}
