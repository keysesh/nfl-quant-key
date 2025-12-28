'use client';

export type ViewType = 'picks' | 'lines' | 'parlays';

interface ViewTabsProps {
  active: ViewType;
  onChange: (view: ViewType) => void;
  counts: {
    picks: number;
    lines: number;
    parlays: number;
  };
}

export default function ViewTabs({ active, onChange, counts }: ViewTabsProps) {
  const tabs: { id: ViewType; label: string; count: number; icon: React.ReactNode }[] = [
    {
      id: 'picks',
      label: 'Player Props',
      count: counts.picks,
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      ),
    },
    {
      id: 'lines',
      label: 'Game Lines',
      count: counts.lines,
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      id: 'parlays',
      label: 'Parlays',
      count: counts.parlays,
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      ),
    },
  ];

  return (
    <div className="flex gap-2 px-4 lg:px-6 py-3 bg-[#0a0a0c]/80 backdrop-blur-sm border-b border-white/[0.04]">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all
            ${active === tab.id
              ? 'bg-gradient-to-r from-emerald-500/20 to-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-[0_0_16px_rgba(16,185,129,0.1)]'
              : 'bg-white/[0.02] border border-white/[0.04] text-zinc-400 hover:bg-white/[0.06] hover:text-zinc-200'
            }
          `}
        >
          {tab.icon}
          <span className="hidden sm:inline">{tab.label}</span>
          <span className={`
            px-2 py-0.5 rounded-full text-xs font-bold
            ${active === tab.id
              ? 'bg-emerald-500/20 text-emerald-400'
              : 'bg-white/[0.06] text-zinc-500'
            }
          `}>
            {tab.count}
          </span>
        </button>
      ))}
    </div>
  );
}
