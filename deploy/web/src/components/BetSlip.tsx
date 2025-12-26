'use client';

import { Pick } from '@/lib/types';
import Image from 'next/image';
import { useState } from 'react';

interface BetSlipProps {
  picks: Pick[];
  isOpen: boolean;
  onClose: () => void;
  onRemovePick: (pickId: string) => void;
  onClearAll: () => void;
}

export default function BetSlip({ picks, isOpen, onClose, onRemovePick, onClearAll }: BetSlipProps) {
  const [copied, setCopied] = useState(false);

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
      {/* Backdrop */}
      <div
        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-md bg-[#0a0f1c] border-l border-slate-700/50 z-50 transform transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-bold text-white">Bet Slip</h2>
            <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 text-sm font-semibold">
              {picks.length}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {picks.length > 0 && (
              <button
                onClick={onClearAll}
                className="px-3 py-1.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
              >
                Clear All
              </button>
            )}
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-slate-400 hover:text-white transition-colors"
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
              <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">No picks yet</h3>
              <p className="text-slate-400 text-sm">
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
                    className="bg-slate-800/50 rounded-xl p-3 border border-slate-700/50"
                  >
                    <div className="flex items-start gap-3">
                      {/* Avatar */}
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
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-white truncate">{pick.player}</h4>
                          <button
                            onClick={() => onRemovePick(pick.id)}
                            className="w-6 h-6 rounded-full bg-slate-700 flex items-center justify-center text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-colors"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                        <p className="text-xs text-slate-500">{pick.team} vs {pick.opponent}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                            isOver ? 'pick-over' : 'pick-under'
                          }`}>
                            {pick.pick.charAt(0)} {pick.line}
                          </span>
                          <span className="text-xs text-slate-400">{pick.market_display}</span>
                        </div>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-4 mt-3 pt-3 border-t border-slate-700/30 text-xs">
                      <div>
                        <span className="text-slate-500">Proj:</span>{' '}
                        <span className="text-white font-medium">{pick.projection.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Edge:</span>{' '}
                        <span className={pick.edge > 0 ? 'text-emerald-400' : 'text-red-400'}>
                          {pick.edge > 0 ? '+' : ''}{pick.edge.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-500">Conf:</span>{' '}
                        <span className="text-white font-medium">{(pick.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        {picks.length > 0 && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-[#0a0f1c] border-t border-slate-700/50">
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={handleExportJSON}
                className="flex items-center justify-center gap-2 py-3 rounded-xl bg-slate-800 text-white font-semibold hover:bg-slate-700 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                {copied ? 'Copied!' : 'Copy JSON'}
              </button>
              <button
                onClick={handleDownloadJSON}
                className="flex items-center justify-center gap-2 py-3 rounded-xl bg-blue-500 text-white font-semibold hover:bg-blue-600 transition-colors"
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
