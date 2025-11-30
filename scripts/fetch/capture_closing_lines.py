#!/usr/bin/env python3
"""
Capture Closing Lines - Run ~30 minutes before games start

This script captures the "closing lines" from DraftKings which are the final
odds before a game begins. Comparing opening recommendations to closing lines
is the gold standard for evaluating betting model quality (Closing Line Value / CLV).

Usage:
    # Capture all upcoming game props (run 30 mins before games)
    python scripts/fetch/capture_closing_lines.py

    # Capture for specific week
    python scripts/fetch/capture_closing_lines.py --week 12

Recommended scheduling:
    - Thursday 7:45 PM ET (before TNF)
    - Sunday 12:30 PM ET (before 1 PM games)
    - Sunday 3:55 PM ET (before 4:25 PM games)
    - Sunday 7:45 PM ET (before SNF)
    - Monday 7:45 PM ET (before MNF)
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.draftkings_client import DKClient, CORE_MARKETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def capture_closing_lines(week: int = None) -> Path:
    """
    Capture current DraftKings player props as closing lines.

    Returns:
        Path to saved CSV file
    """
    output_dir = PROJECT_ROOT / 'data' / 'closing_lines'
    output_dir.mkdir(parents=True, exist_ok=True)

    client = DKClient()

    print("🎯 Capturing closing lines from DraftKings...")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print()

    # Get all props
    df = client.get_all_props(markets=CORE_MARKETS)

    if df.empty:
        print("⚠️  No props found. Games may have already started.")
        return None

    # Add metadata
    df['capture_timestamp'] = datetime.utcnow().isoformat()
    df['capture_type'] = 'closing_line'

    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if week:
        filename = output_dir / f'closing_lines_week{week}_{timestamp}.csv'
    else:
        filename = output_dir / f'closing_lines_{timestamp}.csv'

    df.to_csv(filename, index=False)

    # Summary
    print(f"✅ Captured {len(df)} prop lines")
    print()

    # Group by game
    games = df.groupby(['home_team', 'away_team']).size()
    print("Props by game:")
    for (home, away), count in games.items():
        print(f"  {away} @ {home}: {count} props")

    print()
    print(f"💾 Saved to: {filename}")
    print(f"📊 API quota remaining: {client.remaining}")

    return filename


def calculate_clv(
    opening_lines_file: Path,
    closing_lines_file: Path,
    recommendations_file: Path
) -> pd.DataFrame:
    """
    Calculate Closing Line Value for recommendations.

    CLV = (Closing implied prob) - (Opening implied prob when we bet)
    Positive CLV = line moved in our favor = good

    Args:
        opening_lines_file: Odds when recommendations were made
        closing_lines_file: Final odds before game start
        recommendations_file: Our betting recommendations

    Returns:
        DataFrame with CLV analysis
    """
    # Load data
    opening = pd.read_csv(opening_lines_file)
    closing = pd.read_csv(closing_lines_file)
    recs = pd.read_csv(recommendations_file)

    results = []

    for _, rec in recs.iterrows():
        player = rec.get('player', rec.get('player_name', ''))
        market = rec.get('market', '')
        direction = rec.get('direction', rec.get('pick', '')).upper()

        # Find matching closing line
        close_match = closing[
            (closing['player_name'].str.lower() == player.lower()) &
            (closing['market'] == market)
        ]

        if close_match.empty:
            continue

        close = close_match.iloc[0]

        # Get probabilities
        open_prob = rec.get('dk_prob', rec.get('market_prob', 0.5))

        if 'OVER' in direction:
            close_prob = close['no_vig_over']
        else:
            close_prob = 1 - close['no_vig_over']

        # CLV = closing prob - opening prob (for our side)
        # If we bet OVER and closing OVER prob is higher, we got good CLV
        clv = close_prob - open_prob

        results.append({
            'player': player,
            'market': market,
            'direction': direction,
            'line': rec.get('line', close['line']),
            'open_prob': round(open_prob, 4),
            'close_prob': round(close_prob, 4),
            'clv': round(clv, 4),
            'clv_pct': round(clv * 100, 2),
        })

    df = pd.DataFrame(results)

    if not df.empty:
        print("\n" + "=" * 60)
        print("CLOSING LINE VALUE (CLV) ANALYSIS")
        print("=" * 60)
        print(f"Total bets analyzed: {len(df)}")
        print(f"Average CLV: {df['clv_pct'].mean():.2f}%")
        print(f"Positive CLV bets: {(df['clv'] > 0).sum()} ({(df['clv'] > 0).mean()*100:.1f}%)")
        print(f"Best CLV: {df['clv_pct'].max():.2f}%")
        print(f"Worst CLV: {df['clv_pct'].min():.2f}%")
        print()

        if df['clv_pct'].mean() > 0:
            print("✅ Positive average CLV - model is beating the market!")
        else:
            print("⚠️  Negative average CLV - model may be following the market")

    return df


def main():
    parser = argparse.ArgumentParser(description='Capture closing lines from DraftKings')
    parser.add_argument('--week', type=int, help='NFL week number')
    parser.add_argument('--analyze-clv', action='store_true', help='Analyze CLV for previous captures')
    parser.add_argument('--opening', type=str, help='Opening lines file for CLV analysis')
    parser.add_argument('--closing', type=str, help='Closing lines file for CLV analysis')
    parser.add_argument('--recs', type=str, help='Recommendations file for CLV analysis')

    args = parser.parse_args()

    if args.analyze_clv:
        if not all([args.opening, args.closing, args.recs]):
            print("❌ CLV analysis requires --opening, --closing, and --recs files")
            return

        clv_df = calculate_clv(
            Path(args.opening),
            Path(args.closing),
            Path(args.recs)
        )

        if not clv_df.empty:
            output = PROJECT_ROOT / 'reports' / f'clv_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
            clv_df.to_csv(output, index=False)
            print(f"💾 Saved CLV analysis to: {output}")
        return

    # Capture closing lines
    capture_closing_lines(args.week)


if __name__ == '__main__':
    main()
