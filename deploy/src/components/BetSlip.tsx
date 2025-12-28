'use client';

import { Pick } from '@/lib/types';
import Image from 'next/image';
import { useState, useEffect } from 'react';

interface BetSlipProps {
  picks: Pick[];
  isOpen: boolean;
  onClose: () => void;
  onRemovePick: (pickId: string) => void;
  onClearAll: () => void;
}

export default function BetSlip({ picks, isOpen, onClose, onRemovePick, onClearAll }: BetSlipProps) {
  const [copied, setCopied] = useState(false);

  // Lock body scroll when panel is open
  useEffect(() => {
    if (isOpen) {
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = originalOverflow;
      };
    }
  }, [isOpen]);

  const handleExportJSON = () => {
    const exportData = {
      week: 17,
      exported_at: new Date().toISOString(),
      picks: picks.map(p => ({
        player: p.player,
        team: p.team,
        market: p.market,
        market_display: p.market_display,
        line: p.line,
        pick: p.pick,
        confidence: p.confidence,
        edge: p.edge,
        projection: p.projection,
      })),
    };

    const json = JSON.stringify(exportData, null, 2);
    navigator.clipboard.writeText(json);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadJSON = () => {
    const exportData = {
      week: 17,
      exported_at: new Date().toISOString(),
      picks: picks.map(p => ({
        player: p.player,
        team: p.team,
        market: p.market,
        market_display: p.market_display,
        line: p.line,
        pick: p.pick,
        confidence: p.confidence,
        edge: p.edge,
        projection: p.projection,
      })),
    };

    const json = JSON.stringify(exportData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bet_slip_week17_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      {/* Backdrop - Enhanced blur */}
      <div
        className={`fixed inset-0 bg-black/60 backdrop-blur-md z-40 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* Panel - Layered glass design */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-md z-50 transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        style={{
          background: 'linear-gradient(180deg, rgba(12, 14, 20, 0.95) 0%, rgba(7, 8, 12, 0.98) 100%)',
          backdropFilter: 'blur(24px) saturate(180%)',
          WebkitBackdropFilter: 'blur(24px) saturate(180%)',
          borderLeft: '1px solid rgba(255, 255, 255, 0.06)',
          boxShadow: '-8px 0 32px rgba(0, 0, 0, 0.4)',
        }}
      >
        {/* Header - Glass panel */}
        <div className="flex items-center justify-between p-4 border-b border-white/[0.06]"
          style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)' }}>
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-bold text-white">Bet Slip</h2>
            <span className="px-2.5 py-0.5 rounded-full bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 text-emerald-400 text-sm font-semibold border border-emerald-500/30">
              {picks.length}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {picks.length > 0 && (
              <button
                onClick={onClearAll}
                className="px-3 py-1.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors border border-transparent hover:border-red-500/20"
              >
                Clear All
              </button>
            )}
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
        <div className="flex-1 overflow-y-auto p-4" style={{ height: 'calc(100vh - 180px)' }}>
          {picks.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center mb-4 backdrop-blur-sm">
                <svg className="w-8 h-8 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">No picks yet</h3>
              <p className="text-zinc-500 text-sm">
                Click the + button on any pick to add it to your bet slip
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {picks.map((pick) => {
                const isOver = pick.pick === 'OVER';
                return (
                  <div
                    key={pick.id}
                    className="rounded-xl p-3 border border-white/[0.06] backdrop-blur-sm"
                    style={{
                      background: 'linear-gradient(135deg, rgba(24, 26, 32, 0.8) 0%, rgba(16, 18, 24, 0.6) 100%)',
                    }}
                  >
                    <div className="flex items-start gap-3">
                      {/* Avatar */}
                      <div className="relative flex-shrink-0">
                        <div className="w-10 h-10 rounded-full overflow-hidden ring-2 ring-white/10"
                          style={{ background: 'linear-gradient(135deg, rgba(39,39,42,0.8), rgba(24,24,27,0.9))' }}>
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
                            <div className="w-full h-full flex items-center justify-center text-zinc-400 font-bold text-sm">
                              {pick.player.split(' ').map(n => n[0]).join('')}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-white truncate">{pick.player}</h4>
                          <button
                            onClick={() => onRemovePick(pick.id)}
                            className="w-6 h-6 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-zinc-400 hover:bg-red-500/20 hover:text-red-400 hover:border-red-500/30 transition-all"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <p className="text-xs text-zinc-500">{pick.team} vs {pick.opponent}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className={`px-2 py-0.5 rounded-md text-xs font-bold ${
                            isOver
                              ? 'bg-gradient-to-r from-emerald-500/25 to-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                              : 'bg-gradient-to-r from-blue-500/25 to-blue-500/15 text-blue-400 border border-blue-500/30'
                          }`}>
                            {pick.pick.charAt(0)} {pick.line}
                          </span>
                          <span className="text-xs text-zinc-500">{pick.market_display}</span>
                        </div>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-white/[0.04] text-xs">
                      <div>
                        <span className="text-zinc-500">Proj:</span>{' '}
                        <span className="text-white font-medium">{pick.projection.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-zinc-500">Edge:</span>{' '}
                        <span className={pick.edge > 0 ? 'text-emerald-400' : 'text-red-400'}>
                          {pick.edge > 0 ? '+' : ''}{pick.edge.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-zinc-500">Conf:</span>{' '}
                        <span className="text-white font-medium">{(pick.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer - Glass panel */}
        {picks.length > 0 && (
          <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-white/[0.06]"
            style={{
              background: 'linear-gradient(180deg, rgba(12, 14, 20, 0.9) 0%, rgba(7, 8, 12, 0.98) 100%)',
              backdropFilter: 'blur(20px)',
            }}>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={handleExportJSON}
                className="flex items-center justify-center gap-2 py-3 rounded-xl bg-white/[0.04] border border-white/[0.08] text-white font-semibold hover:bg-white/[0.08] transition-all"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                {copied ? 'Copied!' : 'Copy JSON'}
              </button>
              <button
                onClick={handleDownloadJSON}
                className="flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-semibold hover:from-emerald-400 hover:to-emerald-500 transition-all shadow-[0_0_20px_rgba(34,197,94,0.3)]"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
