'use client';

import { Parlay } from '@/lib/types';

interface ParlaysPanelProps {
  parlays: Parlay[];
}

export default function ParlaysPanel({ parlays }: ParlaysPanelProps) {
  if (parlays.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-zinc-500 text-lg">No parlay recommendations available</p>
      </div>
    );
  }

  // Split into featured and regular
  const featuredParlays = parlays.filter(p => p.featured);
  const regularParlays = parlays.filter(p => !p.featured);

  return (
    <div className="space-y-6">
      {/* Featured Parlays */}
      {featuredParlays.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
            Featured Parlays
            <span className="text-sm text-zinc-500 font-normal">({featuredParlays.length})</span>
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {featuredParlays.map((parlay) => (
              <ParlayCard key={parlay.id} parlay={parlay} featured />
            ))}
          </div>
        </div>
      )}

      {/* Regular Parlays */}
      {regularParlays.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-zinc-500"></span>
            Other Parlays
            <span className="text-sm text-zinc-500 font-normal">({regularParlays.length})</span>
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {regularParlays.map((parlay) => (
              <ParlayCard key={parlay.id} parlay={parlay} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ParlayCard({ parlay, featured = false }: { parlay: Parlay; featured?: boolean }) {
  // Parse legs into individual items
  // New format: "Market Player Direction Line" (e.g., "Receptions Tyler Warren UNDER 4.5")
  // Or: "Anytime TD Player Direction Line" (e.g., "Anytime TD Tee Higgins YES 0.5")
  const legs = parlay.legs.split(' | ').map(leg => {
    const parts = leg.trim().split(' ');

    // Find direction (OVER/UNDER/YES/NO)
    const directionIndex = parts.findIndex(p => ['OVER', 'UNDER', 'YES', 'NO'].includes(p.toUpperCase()));
    const direction = directionIndex >= 0 ? parts[directionIndex] : '';
    const line = parts[parts.length - 1];

    // Known market prefixes (order matters - longer prefixes first)
    const marketPrefixes = [
      'Player Pass Completions', 'Player Rush Attempts', 'Player Reception Yds',
      'Player Rush Yds', 'Player Pass Yds', 'Anytime TD', 'Receptions'
    ];

    let market = '';
    let playerStartIndex = 0;

    // Check for known market prefixes
    const fullLeg = leg.trim();
    for (const prefix of marketPrefixes) {
      if (fullLeg.startsWith(prefix + ' ')) {
        market = prefix;
        playerStartIndex = prefix.split(' ').length;
        break;
      }
    }

    // Player name is between market and direction
    const playerEndIndex = directionIndex >= 0 ? directionIndex : parts.length - 1;
    const player = parts.slice(playerStartIndex, playerEndIndex).join(' ');

    return { player, direction: direction || '', line, market };
  });

  // Parse games into individual matchups
  const games = parlay.games.split(' | ');

  return (
    <div className={`
      rounded-xl p-4 border transition-all
      ${featured
        ? 'bg-gradient-to-br from-yellow-500/10 to-yellow-500/5 border-yellow-500/30 hover:border-yellow-500/50'
        : 'bg-white/[0.03] border-white/[0.06] hover:bg-white/[0.05]'}
    `}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {featured && (
            <span className="text-yellow-400 text-sm font-semibold">FEATURED</span>
          )}
          <span className="text-white font-semibold">{parlay.num_legs}-Leg Parlay</span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-700 text-zinc-300">
            #{parlay.rank}
          </span>
        </div>
        <div className="text-right">
          <div className="text-lg font-bold text-emerald-400">+{parlay.true_odds}</div>
          <div className="text-xs text-zinc-500">odds</div>
        </div>
      </div>

      {/* Legs */}
      <div className="space-y-2 mb-4">
        {legs.map((leg, idx) => (
          <div key={idx} className="flex items-center justify-between py-1.5 px-3 rounded-lg bg-black/20">
            <div className="flex flex-col">
              <span className="text-white font-medium text-sm">{leg.player}</span>
              {leg.market && (
                <span className="text-xs text-zinc-500">{leg.market}</span>
              )}
            </div>
            <span className={`text-sm font-semibold ${
              leg.direction.toUpperCase() === 'OVER' || leg.direction.toUpperCase() === 'YES'
                ? 'text-emerald-400'
                : 'text-red-400'
            }`}>
              {leg.direction} {leg.line}
            </span>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-2 text-center border-t border-white/[0.06] pt-3">
        <div>
          <div className="text-xs text-zinc-500">Model Prob</div>
          <div className="text-sm font-semibold text-white">{parlay.model_prob}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">True Prob</div>
          <div className="text-sm font-semibold text-zinc-400">{parlay.true_prob}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">Edge</div>
          <div className="text-sm font-semibold text-emerald-400">{parlay.edge}</div>
        </div>
        <div>
          <div className="text-xs text-zinc-500">EV</div>
          <div className="text-sm font-semibold text-emerald-400">{parlay.ev}</div>
        </div>
      </div>

      {/* Stake recommendation */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <span className="text-zinc-500">Recommended stake:</span>
        <span className="text-white font-semibold">{parlay.stake} ({parlay.units}u)</span>
      </div>
      <div className="flex items-center justify-between text-sm">
        <span className="text-zinc-500">Potential win:</span>
        <span className="text-emerald-400 font-semibold">{parlay.potential_win}</span>
      </div>
    </div>
  );
}
