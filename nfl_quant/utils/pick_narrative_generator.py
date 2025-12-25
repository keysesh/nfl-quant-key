"""
Generate human-readable narratives explaining the logic behind each betting recommendation.

This module creates automated explanations for why the model recommends each pick,
based on the underlying model calculations, historical data, and opponent analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from nfl_quant.utils.player_names import normalize_player_name


def generate_pick_narrative(
    pick_row: pd.Series,
    historical_avg: Optional[float] = None,
    recent_trend: Optional[str] = None
) -> str:
    """
    Generate a human-readable narrative explaining why the model likes this pick.

    Args:
        pick_row: Row from recommendations CSV with all pick data
        historical_avg: Historical average for this stat (optional)
        recent_trend: 'up', 'down', or 'stable' (optional)

    Returns:
        String narrative explaining the pick logic
    """

    # Extract key metrics
    player = pick_row['player']
    market = pick_row['market']
    line = pick_row['line']
    pick = pick_row['pick']  # 'Over' or 'Under'
    projection = pick_row['model_projection']
    std = pick_row['model_std']
    edge = pick_row['edge_pct']
    prob = pick_row['model_prob']
    opp_epa = pick_row['opponent_def_epa']
    snap_share = pick_row['snap_share']

    # Calculate z-score
    z_score = (line - projection) / std if std > 0 else 0

    # Start building narrative
    narrative_parts = []

    # Part 1: Projection vs Line (core logic)
    if pick == 'Under':
        if abs(z_score) > 2.0:
            narrative_parts.append(
                f"Model projects {projection:.1f} (line {line} is {abs(z_score):.1f}σ above, highly favorable UNDER)"
            )
        elif abs(z_score) > 1.5:
            narrative_parts.append(
                f"Model projects {projection:.1f} vs {line} line ({abs(z_score):.1f}σ above, strong UNDER)"
            )
        else:
            narrative_parts.append(
                f"Model projects {projection:.1f}, below {line} line ({abs(z_score):.1f}σ separation)"
            )
    else:  # Over
        if abs(z_score) > 2.0:
            narrative_parts.append(
                f"Model projects {projection:.1f} (line {line} is {abs(z_score):.1f}σ below, highly favorable OVER)"
            )
        elif abs(z_score) > 1.5:
            narrative_parts.append(
                f"Model projects {projection:.1f} vs {line} line ({abs(z_score):.1f}σ below, strong OVER)"
            )
        else:
            narrative_parts.append(
                f"Model projects {projection:.1f}, above {line} line ({abs(z_score):.1f}σ separation)"
            )

    # Part 2: Historical context (if available)
    if historical_avg is not None:
        diff_from_hist = ((projection - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0

        if abs(diff_from_hist) < 10:
            narrative_parts.append(f"aligns with historical avg ({historical_avg:.1f})")
        elif diff_from_hist > 0:
            narrative_parts.append(f"vs {historical_avg:.1f} historical avg (+{diff_from_hist:.0f}% boost)")
        else:
            narrative_parts.append(f"vs {historical_avg:.1f} historical avg ({diff_from_hist:.0f}% decline)")

    # Part 3: Recent trend (if available)
    if recent_trend == 'down':
        narrative_parts.append("usage trending down")
    elif recent_trend == 'up':
        narrative_parts.append("usage trending up")

    # Part 4: Opponent strength
    if opp_epa < -0.05:
        if pick == 'Under':
            narrative_parts.append(f"strong defense (EPA {opp_epa:.3f}, supports UNDER)")
        else:
            narrative_parts.append(f"tough matchup (EPA {opp_epa:.3f})")
    elif opp_epa > 0.05:
        if pick == 'Over':
            narrative_parts.append(f"weak defense (EPA {opp_epa:.3f}, supports OVER)")
        else:
            narrative_parts.append(f"favorable matchup (EPA {opp_epa:.3f})")

    # Part 5: Role/Snap share context
    if snap_share < 0.30:
        narrative_parts.append(f"backup role ({snap_share:.0%} snaps)")
    elif snap_share > 0.75:
        narrative_parts.append(f"bell-cow ({snap_share:.0%} snaps)")

    # Part 6: Edge magnitude
    if edge >= 25:
        narrative_parts.append(f"exceptional {edge:.0f}% edge")
    elif edge >= 15:
        narrative_parts.append(f"strong {edge:.0f}% edge")
    elif edge >= 10:
        narrative_parts.append(f"{edge:.0f}% edge")

    # Part 7: Confidence
    if prob >= 0.80:
        narrative_parts.append(f"{prob:.0%} confidence")
    elif prob >= 0.70:
        narrative_parts.append(f"{prob:.0%} prob")

    # Combine all parts
    narrative = "; ".join(narrative_parts) + "."

    # Capitalize first letter
    narrative = narrative[0].upper() + narrative[1:]

    return narrative


def generate_short_logic(pick_row: pd.Series) -> str:
    """
    Generate a concise 1-sentence logic summary.

    Args:
        pick_row: Row from recommendations CSV

    Returns:
        Short string explaining core logic
    """

    projection = pick_row['model_projection']
    line = pick_row['line']
    pick = pick_row['pick']
    std = pick_row['model_std']
    edge = pick_row['edge_pct']

    z_score = (line - projection) / std if std > 0 else 0

    if pick == 'Under':
        return f"Projects {projection:.1f} vs {line} line ({abs(z_score):.1f}σ above), {edge:.0%} edge"
    else:
        return f"Projects {projection:.1f} vs {line} line ({abs(z_score):.1f}σ below), {edge:.0%} edge"


def add_narratives_to_recommendations(
    recommendations_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame] = None,
    weekly_stats_df: Optional[pd.DataFrame] = None,
    narrative_type: str = 'full'
) -> pd.DataFrame:
    """
    Add narrative explanations to all recommendations.

    Args:
        recommendations_df: DataFrame with recommendations
        predictions_df: DataFrame with model predictions (optional, for historical context)
        weekly_stats_df: DataFrame with weekly stats (optional, for trends)
        narrative_type: 'full' (detailed) or 'short' (concise)

    Returns:
        DataFrame with added 'logic' column
    """

    df = recommendations_df.copy()

    narratives = []

    for idx, row in df.iterrows():
        if narrative_type == 'short':
            narrative = generate_short_logic(row)
        else:
            # Calculate historical average if data available
            historical_avg = None
            recent_trend = None

            if predictions_df is not None:
                # Try to get historical context
                # This would require matching player and calculating averages
                # For now, use None (can be enhanced later)
                pass

            narrative = generate_pick_narrative(
                row,
                historical_avg=historical_avg,
                recent_trend=recent_trend
            )

        narratives.append(narrative)

    df['logic'] = narratives

    return df


def generate_narrative_with_historical_data(
    pick_row: pd.Series,
    weekly_stats: pd.DataFrame,
    market_stat_map: Optional[Dict[str, str]] = None
) -> str:
    """
    Generate narrative with historical data lookup.

    Args:
        pick_row: Row from recommendations
        weekly_stats: Full weekly stats DataFrame
        market_stat_map: Mapping of market names to stat columns

    Returns:
        Narrative string with historical context
    """

    if market_stat_map is None:
        market_stat_map = {
            'player_receptions': 'receptions',
            'player_receiving_yards': 'receiving_yards',
            'player_reception_yds': 'receiving_yards',
            'player_rush_attempts': 'carries',
            'player_rush_yds': 'rushing_yards',
            'player_pass_yds': 'passing_yards',
            'player_pass_attempts': 'attempts',
            'player_pass_tds': 'passing_tds'
        }

    # Get historical data for this player
    player_name = pick_row['player']
    team = pick_row['team']
    market = pick_row['market']

    # Normalize player name for matching using canonical function
    player_norm = normalize_player_name(player_name)

    # Get stat column
    stat_col = market_stat_map.get(market)

    historical_avg = None
    recent_trend = None

    if stat_col and stat_col in weekly_stats.columns:
        # Filter to this player, recent weeks
        player_stats = weekly_stats[
            (weekly_stats['player_display_name'].apply(normalize_player_name) == player_norm) &
            (weekly_stats['team'] == team) &
            (weekly_stats['season'] == 2025) &
            (weekly_stats['week'] >= 8) &
            (weekly_stats['week'] <= 11)
        ]

        if len(player_stats) > 0:
            historical_avg = player_stats[stat_col].mean()

            # Calculate trend (last 2 vs first 2 games)
            if len(player_stats) >= 4:
                first_half = player_stats.iloc[:2][stat_col].mean()
                second_half = player_stats.iloc[2:][stat_col].mean()

                if second_half > first_half * 1.1:
                    recent_trend = 'up'
                elif second_half < first_half * 0.9:
                    recent_trend = 'down'
                else:
                    recent_trend = 'stable'

    return generate_pick_narrative(
        pick_row,
        historical_avg=historical_avg,
        recent_trend=recent_trend
    )


if __name__ == "__main__":
    # Test with sample data
    sample_pick = pd.Series({
        'player': 'Greg Dortch',
        'market': 'player_receptions',
        'line': 3.5,
        'pick': 'Under',
        'model_projection': 1.85,
        'model_std': 1.36,
        'edge_pct': 39.0,
        'model_prob': 0.837,
        'opponent_def_epa': 0.033,
        'snap_share': 0.44
    })

    narrative = generate_pick_narrative(sample_pick, historical_avg=2.0, recent_trend='down')
    print("Full narrative:")
    print(narrative)
    print()

    short = generate_short_logic(sample_pick)
    print("Short logic:")
    print(short)
