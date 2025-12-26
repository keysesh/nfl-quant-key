'use client';

type ViewType = 'picks' | 'cheatsheet' | 'parlays';

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
  const tabs: { id: ViewType; label: string; count: number }[] = [
    { id: 'picks', label: 'Picks', count: counts.picks },
    { id: 'cheatsheet', label: 'Lines', count: counts.lines },
    { id: 'parlays', label: 'Parlays', count: counts.parlays },
  ];

  return (
    <div className="flex gap-2 px-4 py-2 bg-[#0a0f1c] border-b border-slate-800/50 overflow-x-auto">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all whitespace-nowrap
            ${active === tab.id
              ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/25'
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
            }
          `}
        >
          {tab.label}
          <span className={`
            px-2 py-0.5 rounded-full text-xs font-bold
            ${active === tab.id
              ? 'bg-white/20 text-white'
              : 'bg-slate-700 text-slate-400'
            }
          `}>
            {tab.count}
          </span>
        </button>
      ))}
    </div>
  );
}
