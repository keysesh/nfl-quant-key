"""
CLV (Closing Line Value) calculator utilities.

Calculates how much better our bet was compared to the closing line.
This is the gold standard for validating betting edge.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def calculate_clv_spread(bet_line: float, closing_line: float, bet_side: str) -> float:
    """
    Calculate CLV for spread bets.

    Args:
        bet_line: Line we bet at (e.g., -6.5)
        closing_line: Closing line (e.g., -7.5)
        bet_side: 'favorite' or 'underdog'

    Returns:
        CLV in points (positive = beat closing)
    """
    if bet_side == 'favorite':
        # We got -6.5, closing at -7.5 → +1 point of value
        return abs(closing_line) - abs(bet_line)
    else:
        # We got +3.5, closing at +3.0 → -0.5 points (bad)
        return bet_line - closing_line


def calculate_clv_total(bet_line: float, closing_line: float, bet_side: str) -> float:
    """
    Calculate CLV for total bets.

    Args:
        bet_line: Total we bet (e.g., 47.5)
        closing_line: Closing total (e.g., 49.0)
        bet_side: 'over' or 'under'

    Returns:
        CLV in points
    """
    if bet_side.lower() == 'over':
        # We got Over 47.5, closing at 49.0 → +1.5 points of value
        return closing_line - bet_line
    else:
        # We got Under 47.5, closing at 49.0 → -1.5 points (bad)
        return bet_line - closing_line


def calculate_clv_no_vig(
    bet_odds: int,
    closing_odds: int,
    closing_opposite_odds: int
) -> float:
    """
    Calculate no-vig CLV (true probability-based).

    This is THE KEY METRIC for CLV analysis.

    Args:
        bet_odds: Odds we bet at (e.g., -110)
        closing_odds: Closing odds our side (e.g., -115)
        closing_opposite_odds: Closing odds opposite side (e.g., -105)

    Returns:
        CLV as probability difference (e.g., 0.02 = 2% edge)
    """
    # Our bet implied probability
    bet_prob = american_to_prob(bet_odds)

    # Closing implied probabilities
    closing_prob = american_to_prob(closing_odds)
    closing_opposite_prob = american_to_prob(closing_opposite_odds)

    # Remove vig (Pinnacle method - most accurate)
    total_prob = closing_prob + closing_opposite_prob
    no_vig_prob = closing_prob / total_prob

    # CLV = difference between our odds and no-vig closing
    clv_prob = bet_prob - no_vig_prob

    return clv_prob


def calculate_clv_percentage(
    bet_odds: int,
    closing_odds: int,
    closing_opposite_odds: int
) -> float:
    """
    Calculate CLV as percentage.

    Returns:
        CLV as percentage (e.g., 2.5 = 2.5% edge)
    """
    clv_prob = calculate_clv_no_vig(bet_odds, closing_odds, closing_opposite_odds)
    return clv_prob * 100


def analyze_clv_portfolio(bets_df: pd.DataFrame) -> Dict:
    """
    Analyze CLV across a portfolio of bets.

    Args:
        bets_df: DataFrame with columns: clv_no_vig_prob, our_probability, outcome, market_type

    Returns:
        Dictionary with portfolio CLV metrics
    """
    if len(bets_df) == 0:
        return {}

    results = {
        'total_bets': len(bets_df),
        'avg_clv': bets_df['clv_no_vig_prob'].mean() * 100,  # As percentage
        'median_clv': bets_df['clv_no_vig_prob'].median() * 100,
        'positive_clv_rate': (bets_df['clv_no_vig_prob'] > 0).mean() * 100,
        'avg_positive_clv': bets_df[bets_df['clv_no_vig_prob'] > 0]['clv_no_vig_prob'].mean() * 100,
        'avg_negative_clv': bets_df[bets_df['clv_no_vig_prob'] < 0]['clv_no_vig_prob'].mean() * 100,
    }

    # CLV by market type
    if 'market_type' in bets_df.columns:
        results['clv_by_market'] = bets_df.groupby('market_type')['clv_no_vig_prob'].mean() * 100

    # CLV by confidence level
    if 'our_probability' in bets_df.columns:
        bets_df['confidence_tier'] = pd.cut(
            bets_df['our_probability'],
            bins=[0, 0.55, 0.60, 0.70, 1.0],
            labels=['Low (50-55%)', 'Medium (55-60%)', 'High (60-70%)', 'Very High (70%+)']
        )
        results['clv_by_confidence'] = bets_df.groupby('confidence_tier')['clv_no_vig_prob'].mean() * 100

    # Win rate vs CLV correlation
    if 'outcome' in bets_df.columns:
        # Separate bets by CLV
        positive_clv = bets_df[bets_df['clv_no_vig_prob'] > 0]
        negative_clv = bets_df[bets_df['clv_no_vig_prob'] <= 0]

        if len(positive_clv) > 0:
            results['win_rate_positive_clv'] = positive_clv['outcome'].map({'win': 1, 'loss': 0, 'push': 0.5}).mean() * 100

        if len(negative_clv) > 0:
            results['win_rate_negative_clv'] = negative_clv['outcome'].map({'win': 1, 'loss': 0, 'push': 0.5}).mean() * 100

    return results


def is_clv_healthy(avg_clv: float, positive_rate: float) -> tuple[bool, str]:
    """
    Determine if CLV performance is healthy.

    Args:
        avg_clv: Average CLV percentage
        positive_rate: Percentage of bets with positive CLV

    Returns:
        (is_healthy, message)
    """
    if avg_clv >= 2.0 and positive_rate >= 60:
        return True, "✅ EXCELLENT - Sharp bettor performance"
    elif avg_clv >= 1.0 and positive_rate >= 55:
        return True, "✅ GOOD - Beating the market consistently"
    elif avg_clv >= 0 and positive_rate >= 50:
        return True, "⚠️  OK - Slight edge, room for improvement"
    elif avg_clv >= -1.0:
        return False, "⚠️  WARNING - Marginal performance, review strategy"
    else:
        return False, "❌ CRITICAL - Losing to closing line, stop betting!"


class CLVTracker:
    """Track and analyze CLV for a betting portfolio."""

    def __init__(self, db_path: str = 'data/clv_tracking.db'):
        """Initialize CLV tracker with database connection."""
        import sqlite3
        self.conn = sqlite3.connect(db_path)

    def log_bet(
        self,
        game_id: str,
        market_type: str,
        bet_side: str,
        odds_at_bet: int,
        bet_size: float,
        our_probability: float,
        **kwargs
    ):
        """Log a bet for later CLV calculation."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO our_bets (
                game_id, market_type, bet_side, odds_at_bet,
                bet_size, our_probability, bet_timestamp,
                market_key, player_name, line_at_bet, our_edge, week, season
            ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            market_type,
            bet_side,
            odds_at_bet,
            bet_size,
            our_probability,
            kwargs.get('market_key'),
            kwargs.get('player_name'),
            kwargs.get('line_at_bet'),
            kwargs.get('our_edge'),
            kwargs.get('week'),
            kwargs.get('season', 2025)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def update_bet_clv(
        self,
        bet_id: int,
        closing_line: float,
        closing_odds: int,
        closing_opposite_odds: int
    ):
        """Update bet with closing line for CLV calculation."""
        cursor = self.conn.cursor()

        # Get bet details
        cursor.execute('SELECT odds_at_bet FROM our_bets WHERE id = ?', (bet_id,))
        row = cursor.fetchone()

        if not row:
            return

        bet_odds = row[0]

        # Calculate CLV
        clv_prob = calculate_clv_no_vig(bet_odds, closing_odds, closing_opposite_odds)
        clv_pct = clv_prob * 100

        # Update bet record
        cursor.execute("""
            UPDATE our_bets
            SET closing_line = ?,
                closing_odds = ?,
                closing_opposite_odds = ?,
                clv_no_vig_prob = ?,
                clv_percentage = ?
            WHERE id = ?
        """, (closing_line, closing_odds, closing_opposite_odds, clv_prob, clv_pct, bet_id))

        self.conn.commit()

    def get_weekly_clv_report(self, week: int, season: int = 2025) -> Dict:
        """Generate CLV report for a specific week."""
        query = """
            SELECT * FROM our_bets
            WHERE week = ? AND season = ?
            AND clv_no_vig_prob IS NOT NULL
        """

        bets_df = pd.read_sql_query(query, self.conn, params=(week, season))

        if len(bets_df) == 0:
            return {'week': week, 'total_bets': 0}

        report = analyze_clv_portfolio(bets_df)
        report['week'] = week
        report['season'] = season

        return report

    def close(self):
        """Close database connection."""
        self.conn.close()
