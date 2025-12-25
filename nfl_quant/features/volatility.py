"""
Player Volatility Features

Player standard deviation over trailing weeks predicts whether the line
is beatable at all. High-variance players should require higher confidence
thresholds.

Key insight: Stable players (low CV) are more predictable and thus
more profitable to bet on.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List


def calculate_trailing_volatility(
    values: List[float],
    min_samples: int = 3
) -> Optional[Dict[str, float]]:
    """
    Calculate volatility metrics from trailing values.

    Args:
        values: List of recent stat values
        min_samples: Minimum samples required

    Returns:
        Dict with 'std', 'cv', 'range' or None if insufficient data
    """
    if len(values) < min_samples:
        return None

    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr)

    if mean <= 0:
        return None

    return {
        'mean': mean,
        'std': std,
        'cv': std / mean,  # Coefficient of variation
        'range': np.max(arr) - np.min(arr),
        'min': np.min(arr),
        'max': np.max(arr),
    }


def calculate_player_volatility_from_stats(
    player_stats: pd.DataFrame,
    stat_column: str,
    window: int = 4
) -> Optional[Dict[str, float]]:
    """
    Calculate volatility for a player from their stats DataFrame.

    Args:
        player_stats: DataFrame with player's historical stats
        stat_column: Column to calculate volatility for (e.g., 'receptions')
        window: Number of trailing weeks to use

    Returns:
        Dict with volatility metrics or None
    """
    if len(player_stats) < 3:
        return None

    # Get most recent values
    recent = player_stats.tail(window)

    if stat_column not in recent.columns:
        return None

    values = recent[stat_column].dropna().tolist()
    return calculate_trailing_volatility(values)


def get_volatility_bucket(cv: float) -> str:
    """
    Categorize player volatility into buckets.

    Args:
        cv: Coefficient of variation

    Returns:
        Bucket label: 'low', 'medium', 'high', 'very_high'
    """
    if cv < 0.25:
        return 'low'
    elif cv < 0.40:
        return 'medium'
    elif cv < 0.55:
        return 'high'
    else:
        return 'very_high'


def adjust_confidence_for_volatility(
    base_confidence: float,
    cv: float,
    base_cv: float = 0.30
) -> float:
    """
    Adjust required confidence threshold based on player volatility.

    High-variance players require higher confidence to bet.

    Args:
        base_confidence: Base confidence threshold (e.g., 0.60)
        cv: Player's coefficient of variation
        base_cv: CV at which no adjustment is made (default 0.30)

    Returns:
        Adjusted confidence threshold
    """
    # Linear adjustment: +10% threshold for each 0.1 CV above base
    cv_adjustment = max(0, (cv - base_cv)) * 1.0
    adjusted = base_confidence + cv_adjustment

    # Cap at 80%
    return min(0.80, adjusted)


def calculate_consistency_score(
    values: List[float],
    line: float
) -> Optional[float]:
    """
    Calculate how consistently a player hits over/under a line.

    Args:
        values: Recent stat values
        line: The betting line

    Returns:
        Score from 0-1 where 1 = perfectly consistent
    """
    if len(values) < 3:
        return None

    arr = np.array(values)

    # Count how many times they went over vs under
    over_count = np.sum(arr > line)
    under_count = np.sum(arr <= line)

    # Consistency = how lopsided the distribution is
    total = len(arr)
    max_count = max(over_count, under_count)

    return max_count / total


def get_volatility_features(
    player_stats: pd.DataFrame,
    stat_columns: List[str] = None
) -> Dict[str, float]:
    """
    Get all volatility features for a player.

    Args:
        player_stats: DataFrame with player's historical stats
        stat_columns: Columns to analyze (default: common stat columns)

    Returns:
        Dict with volatility features
    """
    if stat_columns is None:
        stat_columns = ['receptions', 'receiving_yards', 'rushing_yards',
                        'targets', 'carries']

    features = {}

    for col in stat_columns:
        vol = calculate_player_volatility_from_stats(player_stats, col)
        if vol:
            features[f'{col}_cv'] = vol['cv']
            features[f'{col}_std'] = vol['std']
            features[f'{col}_range'] = vol['range']

    # Overall volatility (average CV across stats)
    cvs = [v for k, v in features.items() if k.endswith('_cv')]
    if cvs:
        features['avg_cv'] = np.mean(cvs)
        features['volatility_bucket'] = get_volatility_bucket(features['avg_cv'])

    return features
