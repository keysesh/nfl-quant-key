'use client';

import { GameLine, TeamInfo } from '@/lib/types';

interface GameLinesPanelProps {
  gameLines: GameLine[];
  teams: Record<string, TeamInfo>;
}

// Get tier badge style
function getTierStyle(tier: string): string {
  switch (tier.toLowerCase()) {
    case 'elite':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    case 'strong':
      return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
    case 'moderate':
      return 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30';
    default:
      return 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30';
  }
}

// Format confidence as percentage
function formatConfidence(conf: number): string {
  return `${Math.round(conf * 100)}%`;
}

export default function GameLinesPanel({ gameLines, teams }: GameLinesPanelProps) {
  if (gameLines.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-zinc-500 text-lg">No game line recommendations available</p>
      </div>
    );
  }

  // Group by bet type (spread vs total)
  const spreadLines = gameLines.filter(gl => gl.bet_type === 'spread');
  const totalLines = gameLines.filter(gl => gl.bet_type === 'total');

  return (
    <div className="space-y-6">
      {/* Spread Picks */}
      {spreadLines.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            Spread Picks
            <span className="text-sm text-zinc-500 font-normal">({spreadLines.length})</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {spreadLines.map((line) => (
              <GameLineCard key={line.id} line={line} teams={teams} />
            ))}
          </div>
        </div>
      )}

      {/* Total Picks */}
      {totalLines.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            Total (O/U) Picks
            <span className="text-sm text-zinc-500 font-normal">({totalLines.length})</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {totalLines.map((line) => (
              <GameLineCard key={line.id} line={line} teams={teams} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function GameLineCard({ line, teams }: { line: GameLine; teams: Record<string, TeamInfo> }) {
  const awayTeam = teams[line.away_team?.toUpperCase()] || null;
  const homeTeam = teams[line.home_team?.toUpperCase()] || null;

  const isSpread = line.bet_type === 'spread';
  const pickValue = line.pick;

  // Determine if it's over/under for totals or team pick for spreads
  const isOver = pickValue.toLowerCase().includes('over');
  const isUnder = pickValue.toLowerCase().includes('under');

  return (
    <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 hover:bg-white/[0.05] transition-all">
      {/* Header: Game matchup */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {awayTeam?.logo && (
            <img src={awayTeam.logo} alt={line.away_team} className="w-6 h-6 object-contain" />
          )}
          <span className="text-white font-medium">{line.away_team}</span>
          <span className="text-zinc-500">@</span>
          <span className="text-white font-medium">{line.home_team}</span>
          {homeTeam?.logo && (
            <img src={homeTeam.logo} alt={line.home_team} className="w-6 h-6 object-contain" />
          )}
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full border ${getTierStyle(line.tier)}`}>
          {line.tier.toUpperCase()}
        </span>
      </div>

      {/* Pick display */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-2xl font-bold text-white">
            {pickValue}
          </div>
          <div className="text-xs text-zinc-500">
            {isSpread ? 'Spread' : 'Total'} @ {line.line > 0 ? '+' : ''}{line.line}
          </div>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold text-emerald-400">
            {formatConfidence(line.confidence)}
          </div>
          <div className="text-xs text-zinc-500">confidence</div>
        </div>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-1">
          <span className="text-zinc-500">Edge:</span>
          <span className={line.edge > 0 ? 'text-emerald-400' : 'text-red-400'}>
            {line.edge > 0 ? '+' : ''}{line.edge.toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-zinc-500">Fair:</span>
          <span className="text-white">{line.fair_line > 0 ? '+' : ''}{line.fair_line.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-zinc-500">EV:</span>
          <span className={line.ev > 0 ? 'text-emerald-400' : 'text-zinc-400'}>
            {line.ev > 0 ? '+' : ''}{line.ev.toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}
