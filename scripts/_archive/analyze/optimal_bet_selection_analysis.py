#!/usr/bin/env python3
"""
Optimal Bet Selection Analysis
================================

Comprehensive analysis to identify highest-confidence bets from the portfolio.
This script:
1. Stratifies performance by edge threshold
2. Validates confidence tier definitions
3. Computes statistical confidence intervals
4. Identifies position-market combinations with highest reliability
5. Recommends optimal bet filtering strategy

Uses true out-of-sample validation (leave-one-week-out) to avoid overfitting.
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression


class OptimalBetSelector:
    """Analyzes bet performance to identify optimal selection criteria."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.backtest_df = None
        self.results = {}

    def load_data(self):
        """Load backtest data."""
        backtest_file = self.base_dir / "models" / "calibration" / "backtest_2025.csv"
        self.backtest_df = pd.read_csv(backtest_file)
        print(f"Loaded {len(self.backtest_df):,} predictions")
        print(f"Weeks: {sorted(self.backtest_df['week'].unique())}")
        print()

    def train_calibrator_oos(self, train_df):
        """Train isotonic calibrator on training data."""
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(train_df['predicted_prob_over'].values, train_df['hit_over'].values)
        return ir

    def stratified_edge_analysis(self):
        """Analyze performance at different edge thresholds."""
        print("=" * 80)
        print("1. STRATIFIED EDGE THRESHOLD ANALYSIS")
        print("=" * 80)
        print()

        df = self.backtest_df.copy()
        weeks = sorted(df['week'].unique())

        # Edge thresholds to test
        edge_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

        results = []

        for edge_thresh in edge_thresholds:
            total_bets = 0
            total_wins = 0
            total_losses = 0

            # True OOS: leave-one-week-out
            for test_week in weeks:
                train_df = df[df['week'] != test_week]
                test_df = df[df['week'] == test_week].copy()

                # Train calibrator
                calibrator = self.train_calibrator_oos(train_df)

                # Apply calibration
                test_df['calibrated_prob'] = calibrator.predict(test_df['predicted_prob_over'].values)

                # Filter by edge threshold (OVER bets where cal_prob > 0.5 + threshold)
                qualified = test_df[test_df['calibrated_prob'] - 0.5 > edge_thresh]

                total_bets += len(qualified)
                total_wins += qualified['hit_over'].sum()
                total_losses += len(qualified) - qualified['hit_over'].sum()

            if total_bets > 0:
                win_rate = total_wins / total_bets
                profit = (total_wins * 100) - (total_losses * 110)
                roi = profit / (total_bets * 110) * 100
                bets_per_week = total_bets / len(weeks)

                # Calculate confidence interval for win rate (Wilson score)
                z = 1.96  # 95% CI
                n = total_bets
                p_hat = win_rate
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2*n)) / denominator
                margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2))
                ci_lower = center - margin
                ci_upper = center + margin

                results.append({
                    'edge_threshold': edge_thresh,
                    'total_bets': total_bets,
                    'bets_per_week': round(bets_per_week, 1),
                    'wins': total_wins,
                    'losses': total_losses,
                    'win_rate': round(win_rate, 4),
                    'win_rate_ci_lower': round(ci_lower, 4),
                    'win_rate_ci_upper': round(ci_upper, 4),
                    'roi_pct': round(roi, 2),
                    'profit_per_bet': round(profit / total_bets, 2) if total_bets > 0 else 0
                })

        # Display results
        print(f"{'Edge':>8} {'N Bets':>8} {'Bets/Wk':>8} {'Win%':>8} {'95% CI':>16} {'ROI%':>8} {'$/Bet':>8}")
        print("-" * 80)

        for r in results:
            ci_str = f"[{r['win_rate_ci_lower']:.3f}-{r['win_rate_ci_upper']:.3f}]"
            print(f"{r['edge_threshold']*100:>7.0f}% {r['total_bets']:>8,} {r['bets_per_week']:>8.1f} "
                  f"{r['win_rate']*100:>7.1f}% {ci_str:>16} {r['roi_pct']:>7.1f}% ${r['profit_per_bet']:>7.2f}")

        self.results['stratified_edge'] = results
        print()

        # Find optimal threshold
        max_roi_result = max(results, key=lambda x: x['roi_pct'])
        max_sharpe = max(results, key=lambda x: x['win_rate'] * np.sqrt(x['total_bets']))

        print("RECOMMENDATIONS:")
        print(f"  - Maximum ROI: {max_roi_result['edge_threshold']*100:.0f}% edge ({max_roi_result['roi_pct']:.1f}% ROI)")
        print(f"  - Best risk-adjusted: {max_sharpe['edge_threshold']*100:.0f}% edge (high win rate + volume)")
        print()

        return results

    def probability_tier_analysis(self):
        """Analyze performance by calibrated probability tiers."""
        print("=" * 80)
        print("2. CALIBRATED PROBABILITY TIER ANALYSIS")
        print("=" * 80)
        print()

        df = self.backtest_df.copy()
        weeks = sorted(df['week'].unique())

        # Probability tiers (for OVER bets, higher = more confident OVER)
        prob_tiers = [
            ('50-55%', 0.50, 0.55),
            ('55-60%', 0.55, 0.60),
            ('60-65%', 0.60, 0.65),
            ('65-70%', 0.65, 0.70),
            ('70-75%', 0.70, 0.75),
            ('75-80%', 0.75, 0.80),
            ('80-85%', 0.80, 0.85),
            ('85-90%', 0.85, 0.90),
            ('90%+', 0.90, 1.01),
        ]

        results = []

        for tier_name, prob_low, prob_high in prob_tiers:
            total_bets = 0
            total_wins = 0

            # OOS validation
            for test_week in weeks:
                train_df = df[df['week'] != test_week]
                test_df = df[df['week'] == test_week].copy()

                calibrator = self.train_calibrator_oos(train_df)
                test_df['calibrated_prob'] = calibrator.predict(test_df['predicted_prob_over'].values)

                # Filter to probability tier
                qualified = test_df[(test_df['calibrated_prob'] >= prob_low) &
                                   (test_df['calibrated_prob'] < prob_high)]

                total_bets += len(qualified)
                total_wins += qualified['hit_over'].sum()

            if total_bets > 10:  # Minimum sample size
                win_rate = total_wins / total_bets
                expected_win_rate = (prob_low + prob_high) / 2

                # Calibration error
                cal_error = win_rate - expected_win_rate

                results.append({
                    'tier': tier_name,
                    'total_bets': total_bets,
                    'wins': total_wins,
                    'win_rate': round(win_rate, 4),
                    'expected_rate': round(expected_win_rate, 4),
                    'calibration_error': round(cal_error, 4),
                    'is_well_calibrated': abs(cal_error) < 0.05
                })

        # Display
        print(f"{'Prob Tier':>12} {'N Bets':>8} {'Win%':>8} {'Expected':>8} {'Cal Error':>10} {'Status':>12}")
        print("-" * 80)

        for r in results:
            status = "GOOD" if r['is_well_calibrated'] else "OVERCONFIDENT" if r['calibration_error'] < 0 else "UNDERCONFIDENT"
            print(f"{r['tier']:>12} {r['total_bets']:>8,} {r['win_rate']*100:>7.1f}% "
                  f"{r['expected_rate']*100:>7.1f}% {r['calibration_error']*100:>+9.1f}% {status:>12}")

        self.results['probability_tiers'] = results
        print()

        return results

    def position_market_analysis(self):
        """Analyze which position-market combinations are most reliable."""
        print("=" * 80)
        print("3. POSITION-MARKET RELIABILITY ANALYSIS")
        print("=" * 80)
        print()

        df = self.backtest_df.copy()
        weeks = sorted(df['week'].unique())

        # Get unique position-market combinations
        combinations = df.groupby(['position', 'market']).size().reset_index(name='count')
        combinations = combinations[combinations['count'] >= 100]  # Min sample

        results = []

        for _, row in combinations.iterrows():
            pos = row['position']
            market = row['market']

            total_bets = 0
            total_wins = 0
            all_edges = []

            # OOS validation for this position-market
            pos_market_df = df[(df['position'] == pos) & (df['market'] == market)]

            for test_week in weeks:
                train_df = pos_market_df[pos_market_df['week'] != test_week]
                test_df = pos_market_df[pos_market_df['week'] == test_week].copy()

                if len(train_df) < 50 or len(test_df) < 5:
                    continue

                calibrator = self.train_calibrator_oos(train_df)
                test_df['calibrated_prob'] = calibrator.predict(test_df['predicted_prob_over'].values)

                # 10% edge threshold
                qualified = test_df[test_df['calibrated_prob'] - 0.5 > 0.10]

                total_bets += len(qualified)
                total_wins += qualified['hit_over'].sum()
                all_edges.extend((qualified['calibrated_prob'] - 0.5).tolist())

            if total_bets >= 20:
                win_rate = total_wins / total_bets
                profit = (total_wins * 100) - ((total_bets - total_wins) * 110)
                roi = profit / (total_bets * 110) * 100

                # Correlation between prediction and outcome
                corr = pos_market_df['predicted_prob_over'].corr(pos_market_df['hit_over'])

                results.append({
                    'position': pos,
                    'market': market,
                    'total_bets': total_bets,
                    'wins': total_wins,
                    'win_rate': round(win_rate, 4),
                    'roi_pct': round(roi, 2),
                    'correlation': round(corr, 4),
                    'avg_edge': round(np.mean(all_edges) * 100, 2) if all_edges else 0,
                    'reliability_score': round(win_rate * corr * 100, 2)
                })

        # Sort by reliability score
        results = sorted(results, key=lambda x: x['reliability_score'], reverse=True)

        # Display
        print(f"{'Position':>8} {'Market':>20} {'N Bets':>8} {'Win%':>8} {'ROI%':>8} {'Corr':>8} {'Reliability':>12}")
        print("-" * 90)

        for r in results[:15]:  # Top 15
            print(f"{r['position']:>8} {r['market']:>20} {r['total_bets']:>8,} "
                  f"{r['win_rate']*100:>7.1f}% {r['roi_pct']:>7.1f}% {r['correlation']:>7.3f} {r['reliability_score']:>11.1f}")

        self.results['position_market'] = results
        print()

        # Recommendations
        print("HIGHEST RELIABILITY COMBINATIONS:")
        for r in results[:5]:
            print(f"  - {r['position']} {r['market']}: {r['win_rate']*100:.1f}% win rate, {r['roi_pct']:.1f}% ROI")
        print()

        if results:
            print("LOWEST RELIABILITY (AVOID):")
            for r in results[-3:]:
                print(f"  - {r['position']} {r['market']}: {r['win_rate']*100:.1f}% win rate, {r['roi_pct']:.1f}% ROI")
            print()

        return results

    def elite_bet_identification(self):
        """Identify characteristics of elite bets (highest win rate)."""
        print("=" * 80)
        print("4. ELITE BET IDENTIFICATION (Top Tier)")
        print("=" * 80)
        print()

        df = self.backtest_df.copy()
        weeks = sorted(df['week'].unique())

        # Multi-criteria scoring
        all_scored_bets = []

        for test_week in weeks:
            train_df = df[df['week'] != test_week]
            test_df = df[df['week'] == test_week].copy()

            calibrator = self.train_calibrator_oos(train_df)
            test_df['calibrated_prob'] = calibrator.predict(test_df['predicted_prob_over'].values)
            test_df['edge'] = test_df['calibrated_prob'] - 0.5

            # Only consider positive edge bets
            positive_edge = test_df[test_df['edge'] > 0.05].copy()

            if len(positive_edge) == 0:
                continue

            # Score each bet
            for idx, row in positive_edge.iterrows():
                score = 0

                # 1. Edge magnitude (0-30 points)
                edge_score = min(30, row['edge'] * 100)
                score += edge_score

                # 2. Calibrated probability strength (0-30 points)
                prob_score = (row['calibrated_prob'] - 0.5) * 60
                score += prob_score

                # 3. Position reliability (0-20 points)
                pos_reliability = {
                    'RB': 20,   # Best in backtest
                    'TE': 18,
                    'WR': 15,
                    'QB': 12,
                    'FB': 10
                }
                score += pos_reliability.get(row['position'], 10)

                # 4. Market reliability (0-20 points)
                market_reliability = {
                    'receptions': 20,
                    'targets': 18,
                    'carries': 18,
                    'receiving_yards': 15,
                    'rushing_yards': 12
                }
                score += market_reliability.get(row['market'], 10)

                all_scored_bets.append({
                    'week': test_week,
                    'player': row['player_name'],
                    'position': row['position'],
                    'market': row['market'],
                    'line': row['line'],
                    'calibrated_prob': row['calibrated_prob'],
                    'edge': row['edge'],
                    'hit_over': row['hit_over'],
                    'composite_score': score
                })

        scored_df = pd.DataFrame(all_scored_bets)

        # Define tiers based on composite score
        elite_threshold = scored_df['composite_score'].quantile(0.95)  # Top 5%
        high_threshold = scored_df['composite_score'].quantile(0.85)   # Top 15%
        standard_threshold = scored_df['composite_score'].quantile(0.70)  # Top 30%

        print(f"Score thresholds:")
        print(f"  - ELITE (top 5%): Score >= {elite_threshold:.1f}")
        print(f"  - HIGH (top 15%): Score >= {high_threshold:.1f}")
        print(f"  - STANDARD (top 30%): Score >= {standard_threshold:.1f}")
        print()

        # Performance by tier
        tiers = [
            ('ELITE', elite_threshold, 100),
            ('HIGH', high_threshold, elite_threshold),
            ('STANDARD', standard_threshold, high_threshold),
            ('BELOW_STANDARD', 0, standard_threshold)
        ]

        tier_results = []

        print(f"{'Tier':>15} {'N Bets':>8} {'Win%':>8} {'ROI%':>8} {'Avg Score':>10} {'95% CI Win%':>20}")
        print("-" * 80)

        for tier_name, score_low, score_high in tiers:
            tier_bets = scored_df[(scored_df['composite_score'] >= score_low) &
                                  (scored_df['composite_score'] < score_high)]

            if len(tier_bets) > 0:
                wins = tier_bets['hit_over'].sum()
                n = len(tier_bets)
                win_rate = wins / n
                profit = (wins * 100) - ((n - wins) * 110)
                roi = profit / (n * 110) * 100

                # Confidence interval
                z = 1.96
                p_hat = win_rate
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2*n)) / denominator
                margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2))

                ci_str = f"[{(center-margin)*100:.1f}%-{(center+margin)*100:.1f}%]"

                print(f"{tier_name:>15} {n:>8,} {win_rate*100:>7.1f}% {roi:>7.1f}% "
                      f"{tier_bets['composite_score'].mean():>9.1f} {ci_str:>20}")

                tier_results.append({
                    'tier': tier_name,
                    'n_bets': n,
                    'bets_per_week': round(n / len(weeks), 1),
                    'win_rate': round(win_rate, 4),
                    'roi_pct': round(roi, 2),
                    'avg_score': round(tier_bets['composite_score'].mean(), 2),
                    'ci_lower': round((center - margin), 4),
                    'ci_upper': round((center + margin), 4)
                })

        self.results['elite_tiers'] = tier_results
        print()

        # Sample elite bets
        elite_bets = scored_df[scored_df['composite_score'] >= elite_threshold].nlargest(10, 'composite_score')
        print("SAMPLE ELITE BETS (Top 10):")
        for _, bet in elite_bets.iterrows():
            hit_emoji = "✅" if bet['hit_over'] == 1 else "❌"
            print(f"  {hit_emoji} Week {bet['week']}: {bet['player']} {bet['position']} {bet['market']} "
                  f"Over {bet['line']} | Score: {bet['composite_score']:.1f} | Edge: {bet['edge']*100:.1f}%")
        print()

        return tier_results, scored_df

    def compute_confidence_intervals(self):
        """Compute statistical confidence for key metrics."""
        print("=" * 80)
        print("5. STATISTICAL CONFIDENCE ANALYSIS")
        print("=" * 80)
        print()

        df = self.backtest_df.copy()

        # Overall metrics with confidence intervals
        print("A. Overall Model Performance (True OOS)")
        print("-" * 60)

        # Run full OOS validation
        weeks = sorted(df['week'].unique())
        all_cal_probs = []
        all_actuals = []

        for test_week in weeks:
            train_df = df[df['week'] != test_week]
            test_df = df[df['week'] == test_week].copy()

            calibrator = self.train_calibrator_oos(train_df)
            test_df['calibrated_prob'] = calibrator.predict(test_df['predicted_prob_over'].values)

            # 10% edge
            qualified = test_df[test_df['calibrated_prob'] - 0.5 > 0.10]
            all_cal_probs.extend(qualified['calibrated_prob'].tolist())
            all_actuals.extend(qualified['hit_over'].tolist())

        n = len(all_actuals)
        wins = sum(all_actuals)
        win_rate = wins / n

        # Bootstrap confidence interval for win rate
        n_bootstrap = 10000
        bootstrap_win_rates = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            sampled_actuals = [all_actuals[i] for i in indices]
            bootstrap_win_rates.append(sum(sampled_actuals) / n)

        ci_lower = np.percentile(bootstrap_win_rates, 2.5)
        ci_upper = np.percentile(bootstrap_win_rates, 97.5)

        print(f"Total Qualified Bets (10% edge): {n:,}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"95% Bootstrap CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        print(f"Standard Error: {np.std(bootstrap_win_rates)*100:.2f}%")
        print()

        # Test if win rate is significantly > 52.4% (breakeven at -110)
        breakeven = 0.524
        t_stat = (win_rate - breakeven) / (np.std(bootstrap_win_rates))
        p_value = 1 - stats.norm.cdf(t_stat)

        print(f"Hypothesis Test: H0: Win Rate <= 52.4% (breakeven)")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.6f}")
        if p_value < 0.001:
            print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
        elif p_value < 0.01:
            print("Result: SIGNIFICANT (p < 0.01)")
        elif p_value < 0.05:
            print("Result: SIGNIFICANT (p < 0.05)")
        else:
            print("Result: NOT SIGNIFICANT")
        print()

        # Brier Score confidence
        cal_probs_arr = np.array(all_cal_probs)
        actuals_arr = np.array(all_actuals)
        brier_scores = (cal_probs_arr - actuals_arr) ** 2
        brier_mean = brier_scores.mean()
        brier_se = brier_scores.std() / np.sqrt(n)

        print(f"B. Brier Score Analysis")
        print("-" * 60)
        print(f"Mean Brier Score: {brier_mean:.4f}")
        print(f"95% CI: [{brier_mean - 1.96*brier_se:.4f}, {brier_mean + 1.96*brier_se:.4f}]")
        print(f"Interpretation: {(1 - brier_mean)*100:.1f}% of variance explained")
        print()

        self.results['statistical_confidence'] = {
            'n_bets': n,
            'win_rate': round(win_rate, 4),
            'ci_lower': round(ci_lower, 4),
            'ci_upper': round(ci_upper, 4),
            'p_value': float(p_value),
            'brier_score': round(brier_mean, 4)
        }

        return self.results['statistical_confidence']

    def generate_final_recommendations(self):
        """Generate actionable bet selection recommendations."""
        print("=" * 80)
        print("6. FINAL BET SELECTION RECOMMENDATIONS")
        print("=" * 80)
        print()

        # Based on all analyses
        print("A. OPTIMAL EDGE THRESHOLDS")
        print("-" * 60)

        if 'stratified_edge' in self.results:
            best_roi = max(self.results['stratified_edge'], key=lambda x: x['roi_pct'])
            best_volume = [r for r in self.results['stratified_edge'] if r['total_bets'] > 500]
            if best_volume:
                best_balanced = max(best_volume, key=lambda x: x['roi_pct'])

            print(f"Maximum ROI Strategy:")
            print(f"  - Edge Threshold: {best_roi['edge_threshold']*100:.0f}%")
            print(f"  - Expected Win Rate: {best_roi['win_rate']*100:.1f}%")
            print(f"  - Expected ROI: {best_roi['roi_pct']:.1f}%")
            print(f"  - Bets per Week: {best_roi['bets_per_week']:.0f}")
            print()

            if best_volume:
                print(f"Balanced Volume Strategy:")
                print(f"  - Edge Threshold: {best_balanced['edge_threshold']*100:.0f}%")
                print(f"  - Expected Win Rate: {best_balanced['win_rate']*100:.1f}%")
                print(f"  - Expected ROI: {best_balanced['roi_pct']:.1f}%")
                print(f"  - Bets per Week: {best_balanced['bets_per_week']:.0f}")
                print()

        print("B. RECOMMENDED TIER SYSTEM")
        print("-" * 60)

        if 'elite_tiers' in self.results:
            for tier in self.results['elite_tiers']:
                print(f"{tier['tier']}:")
                print(f"  - Bets per Week: {tier['bets_per_week']:.0f}")
                print(f"  - Win Rate: {tier['win_rate']*100:.1f}% (95% CI: [{tier['ci_lower']*100:.1f}%, {tier['ci_upper']*100:.1f}%])")
                print(f"  - ROI: {tier['roi_pct']:.1f}%")
                print()

        print("C. POSITION-MARKET PRIORITIES")
        print("-" * 60)
        if 'position_market' in self.results and self.results['position_market']:
            print("PRIORITIZE (highest reliability):")
            for r in self.results['position_market'][:5]:
                print(f"  - {r['position']} {r['market']}")

            print("\nAVOID (lowest reliability):")
            for r in self.results['position_market'][-3:]:
                print(f"  - {r['position']} {r['market']}")
        print()

        print("D. FINAL BETTING STRATEGY")
        print("-" * 60)
        print("1. Filter all predictions to >=20% edge for ELITE tier")
        print("2. Apply composite scoring (edge + probability + position + market reliability)")
        print("3. Bet only top 5% of scored predictions (ELITE tier)")
        print("4. Use fractional Kelly (25%) for sizing")
        print("5. Cap exposure per position-market combination")
        print("6. Track actual performance weekly to adjust thresholds")
        print()

    def save_results(self):
        """Save all analysis results to JSON."""
        output_file = self.base_dir / "reports" / "optimal_bet_selection_analysis.json"

        # Convert any non-serializable types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=convert)

        print(f"Analysis results saved to: {output_file}")

    def run_full_analysis(self):
        """Run complete optimal bet selection analysis."""
        self.load_data()
        self.stratified_edge_analysis()
        self.probability_tier_analysis()
        self.position_market_analysis()
        self.elite_bet_identification()
        self.compute_confidence_intervals()
        self.generate_final_recommendations()
        self.save_results()


if __name__ == '__main__':
    analyzer = OptimalBetSelector()
    analyzer.run_full_analysis()
