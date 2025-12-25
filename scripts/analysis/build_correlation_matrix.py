"""
Build Empirical Correlation Matrix
===================================

Computes correlations between betting outcomes from historical data.
Uses the 37,462 bet outcomes in combined_odds_actuals_ENRICHED.csv.

Correlation Types:
1. Same-player: receptions ↔ rec_yds, rush_yds ↔ rush_tds
2. Same-team: QB pass_yds ↔ WR rec_yds (same game)
3. Same-game: player props ↔ game script
4. Cross-game: Should be ~0 (independence verification)

Output: data/correlations/empirical_matrix.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class CorrelationEntry:
    """Single correlation entry."""
    market1: str
    market2: str
    relationship: str  # same_player, same_team, same_game, cross_game
    correlation: float
    sample_size: int
    p_value: float
    confidence_interval_low: float
    confidence_interval_high: float


class CorrelationMatrixBuilder:
    """Build empirical correlation matrix from historical betting data."""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or Path("data/backtest/combined_odds_actuals_ENRICHED.csv")
        self.df: Optional[pd.DataFrame] = None
        self.correlations: Dict[str, Dict[str, CorrelationEntry]] = {}

    def load_data(self) -> pd.DataFrame:
        """Load historical betting data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)

        # Ensure required columns exist
        required_cols = ['season', 'week', 'player', 'market', 'over_hit', 'under_hit']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"Loaded {len(self.df):,} betting records")
        return self.df

    def _calculate_correlation(
        self,
        outcomes1: np.ndarray,
        outcomes2: np.ndarray
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate Pearson correlation with confidence interval.

        Returns:
            (correlation, p_value, (ci_low, ci_high))
        """
        if len(outcomes1) < 10:
            return 0.0, 1.0, (0.0, 0.0)

        # Pearson correlation
        corr, p_value = stats.pearsonr(outcomes1, outcomes2)

        # Fisher z-transform for confidence interval
        n = len(outcomes1)
        z = np.arctanh(corr)
        se = 1 / np.sqrt(n - 3)
        z_crit = 1.96  # 95% CI

        ci_low = np.tanh(z - z_crit * se)
        ci_high = np.tanh(z + z_crit * se)

        return float(corr), float(p_value), (float(ci_low), float(ci_high))

    def compute_same_player_correlations(self) -> Dict[Tuple[str, str], CorrelationEntry]:
        """
        Compute correlations between different markets for the same player.
        Example: player_receptions ↔ player_reception_yds
        """
        print("\nComputing same-player correlations...")
        correlations = {}

        # Group by player-game
        grouped = self.df.groupby(['season', 'week', 'player'])

        # Market pairs to check
        market_pairs = [
            ('player_receptions', 'player_reception_yds'),
            ('player_rush_yds', 'player_rush_attempts'),
            ('player_pass_yds', 'player_pass_tds'),
            ('player_rush_yds', 'player_anytime_td'),
            ('player_reception_yds', 'player_anytime_td'),
        ]

        for market1, market2 in market_pairs:
            outcomes1 = []
            outcomes2 = []

            for (season, week, player), group in grouped:
                m1_rows = group[group['market'] == market1]
                m2_rows = group[group['market'] == market2]

                if len(m1_rows) == 1 and len(m2_rows) == 1:
                    outcomes1.append(m1_rows['over_hit'].iloc[0])
                    outcomes2.append(m2_rows['over_hit'].iloc[0])

            if len(outcomes1) >= 30:
                outcomes1 = np.array(outcomes1)
                outcomes2 = np.array(outcomes2)

                corr, p_val, (ci_low, ci_high) = self._calculate_correlation(outcomes1, outcomes2)

                entry = CorrelationEntry(
                    market1=market1,
                    market2=market2,
                    relationship='same_player',
                    correlation=round(corr, 4),
                    sample_size=len(outcomes1),
                    p_value=round(p_val, 6),
                    confidence_interval_low=round(ci_low, 4),
                    confidence_interval_high=round(ci_high, 4)
                )
                correlations[(market1, market2)] = entry
                print(f"  {market1} ↔ {market2}: r={corr:.3f} (n={len(outcomes1)})")

        return correlations

    def compute_same_team_correlations(self) -> Dict[Tuple[str, str], CorrelationEntry]:
        """
        Compute correlations between different players on the same team.
        Example: QB pass_yds ↔ WR rec_yds
        """
        print("\nComputing same-team correlations...")
        correlations = {}

        # Need team info - check if available
        if 'team' not in self.df.columns and 'player_team' not in self.df.columns:
            print("  Warning: No team column found, skipping same-team correlations")
            return correlations

        team_col = 'team' if 'team' in self.df.columns else 'player_team'

        # Group by team-game
        grouped = self.df.groupby(['season', 'week', team_col])

        # Track QB pass_yds vs WR rec_yds
        qb_wr_outcomes = defaultdict(list)

        for (season, week, team), group in grouped:
            # Get QB passing (player_pass_yds)
            qb_rows = group[group['market'] == 'player_pass_yds']

            # Get WR receiving (player_reception_yds)
            wr_rows = group[group['market'] == 'player_reception_yds']

            for _, qb_row in qb_rows.iterrows():
                for _, wr_row in wr_rows.iterrows():
                    if qb_row['player'] != wr_row['player']:  # Different players
                        qb_wr_outcomes['qb'].append(qb_row['over_hit'])
                        qb_wr_outcomes['wr'].append(wr_row['over_hit'])

        if len(qb_wr_outcomes['qb']) >= 30:
            outcomes1 = np.array(qb_wr_outcomes['qb'])
            outcomes2 = np.array(qb_wr_outcomes['wr'])

            corr, p_val, (ci_low, ci_high) = self._calculate_correlation(outcomes1, outcomes2)

            entry = CorrelationEntry(
                market1='player_pass_yds',
                market2='player_reception_yds',
                relationship='same_team',
                correlation=round(corr, 4),
                sample_size=len(outcomes1),
                p_value=round(p_val, 6),
                confidence_interval_low=round(ci_low, 4),
                confidence_interval_high=round(ci_high, 4)
            )
            correlations[('player_pass_yds', 'player_reception_yds')] = entry
            print(f"  QB pass_yds ↔ WR rec_yds: r={corr:.3f} (n={len(outcomes1)})")

        return correlations

    def compute_same_game_correlations(self) -> Dict[Tuple[str, str], CorrelationEntry]:
        """
        Compute correlations between props in the same game (different teams).
        Should be weak positive due to game script effects.
        """
        print("\nComputing same-game correlations...")
        correlations = {}

        # Check if we have game identifier
        if 'home_team' not in self.df.columns or 'away_team' not in self.df.columns:
            # Try to derive from other columns
            print("  Warning: No home_team/away_team columns, skipping same-game correlations")
            return correlations

        # Group by game
        self.df['game_id'] = (
            self.df['season'].astype(str) + '_' +
            self.df['week'].astype(str) + '_' +
            self.df['home_team'].astype(str) + '_' +
            self.df['away_team'].astype(str)
        )

        grouped = self.df.groupby('game_id')

        # Track within-game correlations for each market
        market_pairs = [
            ('player_pass_yds', 'player_rush_yds'),
            ('player_receptions', 'player_rush_yds'),
        ]

        for market1, market2 in market_pairs:
            outcomes1 = []
            outcomes2 = []

            for game_id, group in grouped:
                m1_rows = group[group['market'] == market1]
                m2_rows = group[group['market'] == market2]

                # Take all pairwise combinations of different players
                for _, row1 in m1_rows.iterrows():
                    for _, row2 in m2_rows.iterrows():
                        if row1['player'] != row2['player']:
                            outcomes1.append(row1['over_hit'])
                            outcomes2.append(row2['over_hit'])

            if len(outcomes1) >= 30:
                outcomes1 = np.array(outcomes1)
                outcomes2 = np.array(outcomes2)

                corr, p_val, (ci_low, ci_high) = self._calculate_correlation(outcomes1, outcomes2)

                entry = CorrelationEntry(
                    market1=market1,
                    market2=market2,
                    relationship='same_game',
                    correlation=round(corr, 4),
                    sample_size=len(outcomes1),
                    p_value=round(p_val, 6),
                    confidence_interval_low=round(ci_low, 4),
                    confidence_interval_high=round(ci_high, 4)
                )
                correlations[(market1, market2)] = entry
                print(f"  {market1} ↔ {market2} (same game): r={corr:.3f} (n={len(outcomes1)})")

        return correlations

    def compute_cross_game_correlations(self) -> Dict[Tuple[str, str], CorrelationEntry]:
        """
        Verify that cross-game correlations are approximately zero.
        This is a sanity check - props from different games should be independent.
        """
        print("\nVerifying cross-game independence...")
        correlations = {}

        # Sample random pairs from different games
        sample_size = min(5000, len(self.df))
        sampled = self.df.sample(n=sample_size, random_state=42)

        # Create game identifier
        if 'home_team' in sampled.columns and 'away_team' in sampled.columns:
            sampled['game_id'] = (
                sampled['season'].astype(str) + '_' +
                sampled['week'].astype(str) + '_' +
                sampled['home_team'].astype(str) + '_' +
                sampled['away_team'].astype(str)
            )

            # Get pairs from different games
            outcomes1 = []
            outcomes2 = []

            game_ids = sampled['game_id'].unique()
            for i, gid1 in enumerate(game_ids[:50]):  # Limit to 50 games
                for gid2 in game_ids[i+1:i+10]:  # Compare with next 10 games
                    g1 = sampled[sampled['game_id'] == gid1]
                    g2 = sampled[sampled['game_id'] == gid2]

                    if len(g1) > 0 and len(g2) > 0:
                        outcomes1.append(g1['over_hit'].iloc[0])
                        outcomes2.append(g2['over_hit'].iloc[0])

            if len(outcomes1) >= 30:
                outcomes1 = np.array(outcomes1)
                outcomes2 = np.array(outcomes2)

                corr, p_val, (ci_low, ci_high) = self._calculate_correlation(outcomes1, outcomes2)

                entry = CorrelationEntry(
                    market1='any',
                    market2='any',
                    relationship='cross_game',
                    correlation=round(corr, 4),
                    sample_size=len(outcomes1),
                    p_value=round(p_val, 6),
                    confidence_interval_low=round(ci_low, 4),
                    confidence_interval_high=round(ci_high, 4)
                )
                correlations[('any', 'any')] = entry
                print(f"  Cross-game baseline: r={corr:.3f} (should be ~0)")

        return correlations

    def build_matrix(self) -> Dict:
        """Build the complete correlation matrix."""
        if self.df is None:
            self.load_data()

        all_correlations = {}

        # Compute each type
        same_player = self.compute_same_player_correlations()
        same_team = self.compute_same_team_correlations()
        same_game = self.compute_same_game_correlations()
        cross_game = self.compute_cross_game_correlations()

        # Combine
        all_correlations['same_player'] = {
            f"{k[0]}|{k[1]}": asdict(v) for k, v in same_player.items()
        }
        all_correlations['same_team'] = {
            f"{k[0]}|{k[1]}": asdict(v) for k, v in same_team.items()
        }
        all_correlations['same_game'] = {
            f"{k[0]}|{k[1]}": asdict(v) for k, v in same_game.items()
        }
        all_correlations['cross_game'] = {
            f"{k[0]}|{k[1]}": asdict(v) for k, v in cross_game.items()
        }

        # Add metadata
        all_correlations['metadata'] = {
            'source_file': str(self.data_path),
            'total_records': len(self.df),
            'date_generated': pd.Timestamp.now().isoformat()
        }

        return all_correlations

    def save_matrix(self, output_path: Path = None):
        """Save the correlation matrix to JSON."""
        output_path = output_path or Path("data/correlations/empirical_matrix.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrix = self.build_matrix()

        with open(output_path, 'w') as f:
            json.dump(matrix, f, indent=2)

        print(f"\nCorrelation matrix saved to {output_path}")
        return matrix


def main():
    """Build and save the empirical correlation matrix."""
    builder = CorrelationMatrixBuilder()
    matrix = builder.save_matrix()

    # Print summary
    print("\n" + "=" * 60)
    print("CORRELATION MATRIX SUMMARY")
    print("=" * 60)

    for relationship in ['same_player', 'same_team', 'same_game', 'cross_game']:
        entries = matrix.get(relationship, {})
        if entries:
            print(f"\n{relationship.upper().replace('_', ' ')} ({len(entries)} pairs):")
            for key, entry in entries.items():
                corr = entry['correlation']
                n = entry['sample_size']
                print(f"  {key}: r={corr:.3f} (n={n})")


if __name__ == "__main__":
    main()
