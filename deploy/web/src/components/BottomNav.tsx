'use client';

export type NavItem = 'picks' | 'stats' | 'slip';

interface BottomNavProps {
  activeTab: NavItem;
  onTabChange: (tab: NavItem) => void;
  slipCount?: number;
  onOpenSlip?: () => void;
}

export default function BottomNav({ activeTab, onTabChange, slipCount = 0, onOpenSlip }: BottomNavProps) {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 glass-nav border-t border-white/5 pb-safe">
      <div className="flex items-center justify-around py-2 px-4 max-w-lg mx-auto">
        {/* Picks */}
        <button
          onClick={() => onTabChange('picks')}
          className={`flex flex-col items-center gap-1 px-6 py-2 rounded-xl transition-all ${
            activeTab === 'picks' ? 'bg-white/5' : ''
          }`}
        >
          <svg className={`w-6 h-6 ${activeTab === 'picks' ? 'text-emerald-500' : 'text-zinc-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <span className={`text-[10px] font-medium ${activeTab === 'picks' ? 'text-emerald-500' : 'text-zinc-500'}`}>Picks</span>
        </button>

        {/* Stats */}
        <button
          onClick={() => onTabChange('stats')}
          className={`flex flex-col items-center gap-1 px-6 py-2 rounded-xl transition-all ${
            activeTab === 'stats' ? 'bg-white/5' : ''
          }`}
        >
          <svg className={`w-6 h-6 ${activeTab === 'stats' ? 'text-emerald-500' : 'text-zinc-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span className={`text-[10px] font-medium ${activeTab === 'stats' ? 'text-emerald-500' : 'text-zinc-500'}`}>Stats</span>
        </button>

        {/* Bet Slip */}
        <button
          onClick={onOpenSlip}
          className="relative flex flex-col items-center gap-1 px-6 py-2 rounded-xl"
        >
          <div className="relative">
            <svg className="w-6 h-6 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
            </svg>
            {slipCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 text-[10px] font-bold text-black flex items-center justify-center">
                {slipCount}
              </span>
            )}
          </div>
          <span className="text-[10px] font-medium text-zinc-500">Slip</span>
        </button>
      </div>
    </nav>
  );
}
