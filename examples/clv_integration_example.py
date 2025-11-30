"""
Example: How to integrate CLV tracking into your betting workflow.

This shows how to:
1. Log bets when placing them
2. Update with closing lines
3. Generate CLV reports
4. Use CLV to validate your edge
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from nfl_quant.validation.clv_calculator import CLVTracker


def example_workflow():
    """Example betting workflow with CLV tracking."""

    # Initialize tracker
    tracker = CLVTracker()

    # Step 1: When placing a bet
    bet_id = tracker.log_bet(
        game_id='2025_09_KC_BUF',
        market_type='spread',
        bet_side='KC -6.5',
        odds_at_bet=-110,
        bet_size=10.00,
        our_probability=0.58,
        line_at_bet=-6.5,
        our_edge=0.05,
        week=9,
        season=2025
    )

    print(f"✅ Logged bet #{bet_id}")

    # Step 2: After game, update with closing line
    # (This would be automated via line scraper)
    tracker.update_bet_clv(
        bet_id=bet_id,
        closing_line=-7.5,  # Line moved in our favor!
        closing_odds=-115,
        closing_opposite_odds=-105
    )

    print("✅ Updated with closing line")

    # Step 3: Generate weekly CLV report
    report = tracker.get_weekly_clv_report(week=9, season=2025)

    print(f"\n📊 Week 9 CLV Report:")
    print(f"   Total Bets: {report['total_bets']}")
    print(f"   Average CLV: {report['avg_clv']:.2f}%")
    print(f"   Positive CLV Rate: {report['positive_clv_rate']:.1f}%")

    tracker.close()


if __name__ == "__main__":
    example_workflow()
