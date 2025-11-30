#!/usr/bin/env python3
"""
Optimal Bet Portfolio Optimizer

Maximizes ROI by selecting the optimal subset of bets given:
- Bankroll constraint ($50 default)
- Kelly criterion bet sizing
- Multiple betting markets (props, anytime TD, etc.)
- Parlays and alt lines (separate from main bankroll)

Uses portfolio optimization to find the best combination of bets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import itertools


class BetPortfolioOptimizer:
    """Optimize bet selection to maximize ROI within bankroll constraints."""

    def __init__(self, bankroll: float = 50.0, kelly_fraction: float = 0.25, max_bet_pct: float = 3.0):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet = 1.0

    def kelly_bet_size(self, american_odds: float, win_prob: float) -> float:
        """Calculate Kelly criterion bet size."""
        if american_odds < 0:
            b = 100 / abs(american_odds)
        else:
            b = american_odds / 100

        p = win_prob
        q = 1 - p

        # Full Kelly
        full_kelly = (b * p - q) / b
        full_kelly = max(0, full_kelly)  # No negative bets

        # Apply Kelly fraction
        kelly_scaled = full_kelly * self.kelly_fraction

        # Cap at max bet percentage
        max_bet_amount = self.bankroll * (self.max_bet_pct / 100)
        kelly_bet_amount = self.bankroll * kelly_scaled
        bet_amount = min(kelly_bet_amount, max_bet_amount)

        # Apply minimum bet
        bet_amount = max(bet_amount, self.min_bet)

        return bet_amount

    def calculate_expected_value(self, bet_size: float, american_odds: float, win_prob: float) -> float:
        """Calculate expected value of a bet."""
        if american_odds < 0:
            win_amount = bet_size * (100 / abs(american_odds))
        else:
            win_amount = bet_size * (american_odds / 100)

        expected_profit = (win_prob * win_amount) - ((1 - win_prob) * bet_size)
        return expected_profit

    def calculate_roi(self, bet_size: float, american_odds: float, win_prob: float) -> float:
        """Calculate ROI percentage."""
        ev = self.calculate_expected_value(bet_size, american_odds, win_prob)
        roi = (ev / bet_size) * 100 if bet_size > 0 else 0
        return roi

    def optimize_portfolio_greedy(self, bets_df: pd.DataFrame, max_bankroll_usage: float = 0.9) -> List[Dict]:
        """
        Greedy optimization: Select bets with highest ROI per dollar until bankroll exhausted.

        Args:
            bets_df: DataFrame with columns: ['pick', 'american_odds', 'model_prob', 'edge', 'player', 'market']
            max_bankroll_usage: Maximum % of bankroll to use (default 90%)

        Returns:
            List of selected bets with bet sizes
        """
        # Calculate bet size and ROI for each bet
        bets_with_sizing = []
        for _, row in bets_df.iterrows():
            bet_size = self.kelly_bet_size(row['american_odds'], row['model_prob'])
            roi = self.calculate_roi(bet_size, row['american_odds'], row['model_prob'])
            ev = self.calculate_expected_value(bet_size, row['american_odds'], row['model_prob'])

            bets_with_sizing.append({
                'index': row.name,
                'pick': row['pick'],
                'player': row.get('player', ''),
                'market': row.get('market', ''),
                'american_odds': row['american_odds'],
                'model_prob': row['model_prob'],
                'edge': row['edge'],
                'bet_size': bet_size,
                'roi': roi,
                'expected_value': ev,
                'roi_per_dollar': roi / bet_size if bet_size > 0 else 0
            })

        # Sort by ROI per dollar (efficiency)
        bets_with_sizing.sort(key=lambda x: x['roi_per_dollar'], reverse=True)

        # Greedy selection
        selected_bets = []
        total_risk = 0.0
        max_total_risk = self.bankroll * max_bankroll_usage

        for bet in bets_with_sizing:
            if total_risk + bet['bet_size'] <= max_total_risk:
                selected_bets.append(bet)
                total_risk += bet['bet_size']
            else:
                # Try to fit a smaller bet if possible
                remaining = max_total_risk - total_risk
                if remaining >= self.min_bet:
                    # Scale down bet size to fit remaining bankroll
                    bet['bet_size'] = remaining
                    bet['expected_value'] = self.calculate_expected_value(remaining, bet['american_odds'], bet['model_prob'])
                    bet['roi'] = self.calculate_roi(remaining, bet['american_odds'], bet['model_prob'])
                    selected_bets.append(bet)
                    total_risk += remaining
                    break

        return selected_bets

    def optimize_portfolio_knapsack(self, bets_df: pd.DataFrame, max_bankroll_usage: float = 0.9) -> List[Dict]:
        """
        Knapsack optimization: Select optimal combination to maximize expected value.

        Uses dynamic programming approach for small sets, greedy approximation for large sets.
        """
        if len(bets_df) <= 20:
            return self._knapsack_exact(bets_df, max_bankroll_usage)
        else:
            return self._knapsack_approximate(bets_df, max_bankroll_usage)

    def _knapsack_exact(self, bets_df: pd.DataFrame, max_bankroll_usage: float) -> List[Dict]:
        """Exact solution using dynamic programming (works for small sets only)."""
        # Calculate all bet sizes and values
        bets_data = []
        for idx, row in bets_df.iterrows():
            bet_size = self.kelly_bet_size(row['american_odds'], row['model_prob'])
            ev = self.calculate_expected_value(bet_size, row['american_odds'], row['model_prob'])

            bets_data.append({
                'index': idx,
                'size': bet_size,
                'value': ev,
                'row': row
            })

        # Round to nearest cent for DP
        max_bankroll_cents = int(self.bankroll * max_bankroll_usage * 100)

        # DP table: dp[i][w] = max value using first i bets with weight w
        n = len(bets_data)
        dp = [[0.0] * (max_bankroll_cents + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            bet_size_cents = int(bets_data[i-1]['size'] * 100)
            bet_value = bets_data[i-1]['value']

            for w in range(max_bankroll_cents + 1):
                dp[i][w] = dp[i-1][w]  # Don't take bet i

                if w >= bet_size_cents:
                    # Option: take bet i
                    dp[i][w] = max(dp[i][w], dp[i-1][w - bet_size_cents] + bet_value)

        # Backtrack to find selected bets
        selected_indices = []
        w = max_bankroll_cents
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_indices.append(bets_data[i-1]['index'])
                bet_size_cents = int(bets_data[i-1]['size'] * 100)
                w -= bet_size_cents

        # Build result
        selected_bets = []
        total_risk = 0.0
        for idx in selected_indices:
            row = bets_df.loc[idx]
            bet_size = self.kelly_bet_size(row['american_odds'], row['model_prob'])
            ev = self.calculate_expected_value(bet_size, row['american_odds'], row['model_prob'])
            roi = self.calculate_roi(bet_size, row['american_odds'], row['model_prob'])

            selected_bets.append({
                'index': idx,
                'pick': row['pick'],
                'player': row.get('player', ''),
                'market': row.get('market', ''),
                'american_odds': row['american_odds'],
                'model_prob': row['model_prob'],
                'edge': row['edge'],
                'bet_size': bet_size,
                'roi': roi,
                'expected_value': ev
            })
            total_risk += bet_size

        return selected_bets

    def _knapsack_approximate(self, bets_df: pd.DataFrame, max_bankroll_usage: float) -> List[Dict]:
        """Approximate solution using greedy approach (faster for large sets)."""
        return self.optimize_portfolio_greedy(bets_df, max_bankroll_usage)

    def generate_parlay_recommendations(self, selected_bets: List[Dict], max_parlays: int = 10) -> List[Dict]:
        """
        Generate parlay recommendations from selected bets.

        Note: Parlays don't count against main bankroll - they're additive opportunities.
        """
        if len(selected_bets) < 2:
            return []

        parlays = []

        # 2-leg parlays from top bets
        top_bets = sorted(selected_bets, key=lambda x: x['roi'], reverse=True)[:10]

        for i, bet1 in enumerate(top_bets[:8]):
            for bet2 in top_bets[i+1:10]:
                # Avoid same player (correlated)
                if bet1.get('player') == bet2.get('player'):
                    continue

                # Calculate parlay probability (assume independence)
                parlay_prob = bet1['model_prob'] * bet2['model_prob']

                # Calculate parlay odds
                dec1 = abs(bet1['american_odds']) / 100 + 1 if bet1['american_odds'] < 0 else bet1['american_odds'] / 100 + 1
                dec2 = abs(bet2['american_odds']) / 100 + 1 if bet2['american_odds'] < 0 else bet2['american_odds'] / 100 + 1
                parlay_decimal = dec1 * dec2

                if parlay_decimal >= 2:
                    parlay_american = int((parlay_decimal - 1) * 100)
                else:
                    parlay_american = int(-100 / (parlay_decimal - 1))

                # Calculate edge
                parlay_market_prob = 1 / parlay_decimal
                parlay_edge = parlay_prob - parlay_market_prob

                # Only include if positive edge
                if parlay_edge > 0.05:  # 5% edge minimum for parlays
                    parlay_size = min(5.0, self.bankroll * 0.05)  # Smaller bet sizes for parlays
                    parlay_ev = self.calculate_expected_value(parlay_size, parlay_american, parlay_prob)
                    parlay_roi = self.calculate_roi(parlay_size, parlay_american, parlay_prob)

                    parlays.append({
                        'type': '2-leg parlay',
                        'legs': [
                            f"{bet1['pick']} ({bet1.get('player', '')})",
                            f"{bet2['pick']} ({bet2.get('player', '')})"
                        ],
                        'american_odds': parlay_american,
                        'model_prob': parlay_prob,
                        'edge': parlay_edge,
                        'bet_size': parlay_size,
                        'roi': parlay_roi,
                        'expected_value': parlay_ev
                    })

                    if len(parlays) >= max_parlays:
                        break

            if len(parlays) >= max_parlays:
                break

        return parlays


def load_all_recommendations() -> pd.DataFrame:
    """Load all recommendations from unified file."""
    unified_path = Path('reports/unified_betting_recommendations.csv')

    if not unified_path.exists():
        print(f"❌ File not found: {unified_path}")
        print("   Run: python scripts/run_complete_pipeline.py 9 aggressive")
        return pd.DataFrame()

    df = pd.read_csv(unified_path)

    # Map column names (handle different naming conventions)
    column_mapping = {
        'odds': 'american_odds',
        'american_odds': 'american_odds',
        'framework_win_pct': 'model_prob',
        'model_prob': 'model_prob',
        'framework_win_percent': 'model_prob',
        'edge_pct': 'edge',
        'edge': 'edge',
        'edge_percent': 'edge'
    }

    # Rename columns if needed
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Handle edge conversion (might be percentage)
    if 'edge' in df.columns:
        if df['edge'].max() > 1.0:  # If edge is in percentage (0-100)
            df['edge'] = df['edge'] / 100.0

    # Handle model_prob conversion (might be percentage)
    if 'model_prob' in df.columns:
        if df['model_prob'].max() > 1.0:  # If probability is in percentage (0-100)
            df['model_prob'] = df['model_prob'] / 100.0

    # Ensure required columns exist
    required_cols = ['pick', 'american_odds', 'model_prob', 'edge']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {df.columns.tolist()}")
        print("\n   Trying to infer columns...")

        # Try to infer american_odds
        if 'american_odds' not in df.columns:
            if 'market_odds' in df.columns:
                df['american_odds'] = df['market_odds']
            elif 'odds' in df.columns:
                df['american_odds'] = df['odds']
            elif 'american_odds_number' in df.columns:
                df['american_odds'] = df['american_odds_number']

        # Try to infer model_prob
        if 'model_prob' not in df.columns:
            if 'our_prob' in df.columns:
                df['model_prob'] = df['our_prob']
            elif 'framework_win_pct' in df.columns:
                df['model_prob'] = df['framework_win_pct'] / 100.0
            elif 'confidence' in df.columns:
                df['model_prob'] = df['confidence'] / 100.0

        # Try to infer edge
        if 'edge' not in df.columns:
            if 'edge_pct' in df.columns:
                df['edge'] = df['edge_pct'] / 100.0
            elif 'framework_edge' in df.columns:
                df['edge'] = df['framework_edge'] / 100.0

        # Check again
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Still missing: {missing_cols}")
            return pd.DataFrame()

    return df


def include_anytime_td_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Add anytime TD markets if available."""
    # Check if anytime TD exists in data
    anytime_td = df[df['market'].str.contains('anytime_td', case=False, na=False)]

    if len(anytime_td) == 0:
        print("⚠️  No anytime TD markets found in recommendations")
        print("   To include: Ensure fetch script includes 'player_anytime_td' market")

    return df


def main():
    """Main execution."""
    print("=" * 100)
    print("🎯 OPTIMAL BET PORTFOLIO OPTIMIZER")
    print("=" * 100)
    print()

    # Load bankroll config
    config_path = Path('configs/bankroll_config.json')
    if config_path.exists():
        with open(config_path) as f:
            bankroll_config = json.load(f)
        bankroll = bankroll_config.get('total_bankroll', 50.0)
    else:
        bankroll = 50.0
        print(f"⚠️  Using default bankroll: ${bankroll:.2f}")

    print(f"💰 Bankroll: ${bankroll:.2f}")
    print()

    # Load all recommendations
    print("📊 Loading recommendations...")
    df = load_all_recommendations()

    if df.empty:
        return

    print(f"✅ Loaded {len(df)} total recommendations")

    # Include anytime TD markets
    df = include_anytime_td_markets(df)

    # Filter for positive edge only
    df = df[df['edge'] > 0].copy()
    print(f"✅ {len(df)} bets with positive edge")
    print()

    # Initialize optimizer
    optimizer = BetPortfolioOptimizer(bankroll=bankroll)

    # Optimize portfolio
    print("🔍 Optimizing portfolio...")
    print("   Strategy: Maximize ROI within bankroll constraint")
    print()

    selected_bets = optimizer.optimize_portfolio_greedy(df, max_bankroll_usage=0.9)

    if not selected_bets:
        print("❌ No bets selected - check bankroll and bet sizes")
        return

    # Calculate portfolio metrics
    total_risk = sum(bet['bet_size'] for bet in selected_bets)
    total_expected_value = sum(bet['expected_value'] for bet in selected_bets)
    weighted_roi = (total_expected_value / total_risk * 100) if total_risk > 0 else 0

    # Display results
    print("=" * 100)
    print("📊 OPTIMAL PORTFOLIO SUMMARY")
    print("=" * 100)
    print()
    print(f"Total Bets Selected: {len(selected_bets)}")
    print(f"Total Risk: ${total_risk:.2f} ({total_risk/bankroll*100:.1f}% of bankroll)")
    print(f"Total Expected Value: ${total_expected_value:.2f}")
    print(f"Portfolio ROI: {weighted_roi:.1f}%")
    print()

    # Display bets
    print("=" * 100)
    print("🎯 SELECTED BETS (Sorted by ROI)")
    print("=" * 100)
    print()
    print(f"{'Player':<20} {'Pick':<25} {'Prob':<8} {'Edge':<8} {'Bet':<8} {'ROI':<8} {'EV':<10}")
    print("-" * 100)

    for bet in sorted(selected_bets, key=lambda x: x['roi'], reverse=True):
        player = str(bet.get('player', ''))[:19]
        pick = str(bet['pick'])[:24]
        prob = f"{bet['model_prob']:.1%}"
        edge = f"{bet['edge']:.1%}"
        bet_size = f"${bet['bet_size']:.2f}"
        roi = f"{bet['roi']:.1f}%"
        ev = f"${bet['expected_value']:.2f}"

        print(f"{player:<20} {pick:<25} {prob:<8} {edge:<8} {bet_size:<8} {roi:<8} {ev:<10}")

    print()

    # Generate parlay recommendations
    print("=" * 100)
    print("🎲 PARLAY RECOMMENDATIONS (Additive - Not in Bankroll)")
    print("=" * 100)
    print()

    parlays = optimizer.generate_parlay_recommendations(selected_bets, max_parlays=10)

    if parlays:
        print(f"{'Type':<15} {'Legs':<50} {'Odds':<10} {'Prob':<8} {'Bet':<8} {'ROI':<8} {'EV':<10}")
        print("-" * 100)

        for parlay in sorted(parlays, key=lambda x: x['roi'], reverse=True)[:10]:
            ptype = parlay['type']
            legs = " + ".join(parlay['legs'])[:48]
            odds = f"{parlay['american_odds']:+.0f}"
            prob = f"{parlay['model_prob']:.1%}"
            bet_size = f"${parlay['bet_size']:.2f}"
            roi = f"{parlay['roi']:.1f}%"
            ev = f"${parlay['expected_value']:.2f}"

            print(f"{ptype:<15} {legs:<50} {odds:<10} {prob:<8} {bet_size:<8} {roi:<8} {ev:<10}")
    else:
        print("No parlay opportunities found")

    print()

    # Save optimized portfolio
    output_file = Path('reports/optimized_bet_portfolio.csv')
    portfolio_df = pd.DataFrame(selected_bets)
    portfolio_df.to_csv(output_file, index=False)
    print(f"💾 Optimized portfolio saved: {output_file}")

    # Save parlays separately
    if parlays:
        parlays_file = Path('reports/parlay_recommendations.csv')
        parlays_df = pd.DataFrame(parlays)
        parlays_df.to_csv(parlays_file, index=False)
        print(f"💾 Parlay recommendations saved: {parlays_file}")

    print()
    print("=" * 100)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 100)
    print()
    print("📌 Key Points:")
    print(f"   - Selected {len(selected_bets)} bets optimizing ROI")
    print(f"   - Total risk: ${total_risk:.2f} ({total_risk/bankroll*100:.1f}% of ${bankroll:.2f} bankroll)")
    print(f"   - Expected ROI: {weighted_roi:.1f}%")
    print(f"   - Parlays: {len(parlays)} additional opportunities (separate from bankroll)")
    print()


if __name__ == '__main__':
    main()
