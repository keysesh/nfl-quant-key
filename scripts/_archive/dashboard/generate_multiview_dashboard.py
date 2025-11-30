#!/usr/bin/env python3
"""
Generate Multi-View Interactive Dashboard for NFL QUANT Recommendations

Features:
1. Top Picks (at a glance)
2. By Game (all picks grouped by matchup) - COLLAPSIBLE
3. By Prop Type (all receptions together, all yards together, etc.) - COLLAPSIBLE
4. By Player (all props for each player) - COLLAPSIBLE
5. Game Lines (spreads, totals, moneylines)
6. Live Edges (real-time DraftKings comparison) - NEW!

Usage:
    python scripts/dashboard/generate_multiview_dashboard.py
    python scripts/dashboard/generate_multiview_dashboard.py --fetch-live  # Fetch live odds
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User's max bet per pick
MAX_BET_DOLLARS = 5.0


def load_schedule(week: int = None) -> pd.DataFrame:
    """Load schedule with game times."""
    schedule_path = Path('data/nflverse/schedules_2024_2025.csv')
    if not schedule_path.exists():
        logger.warning("Schedule file not found")
        return pd.DataFrame()

    schedule = pd.read_csv(schedule_path)
    # Filter for current season
    schedule = schedule[schedule['season'] == 2025].copy()

    # If week specified, filter to that week
    if week is not None:
        schedule = schedule[schedule['week'] == week].copy()

    logger.info(f"Loaded {len(schedule)} scheduled games")
    return schedule


def load_recommendations() -> pd.DataFrame:
    """Load current week recommendations."""
    csv_path = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Recommendations not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Calculate effective confidence based on V12 probability for supported markets
    # For V12 markets (receptions, rush_yds, reception_yds, pass_yds), use v12_p_under
    # For other markets, use model_prob
    v12_markets = ['player_receptions', 'player_rush_yds', 'player_reception_yds', 'player_pass_yds']

    def get_effective_confidence(row):
        """Get the best confidence measure for this pick."""
        v12_p = row.get('v12_p_under', 0.5)
        model_p = row.get('model_prob', 0.5)
        market = row.get('market', '')
        pick = str(row.get('pick', '')).lower()

        # For V12 markets, use V12 probability (adjusted for direction)
        if market in v12_markets and v12_p != 0.5:
            if 'under' in pick:
                return v12_p
            else:
                return 1 - v12_p
        # For other markets, use model_prob
        return model_p

    def get_confidence_tier(prob):
        """Convert probability to tier."""
        if prob >= 0.70:
            return 'ELITE'
        elif prob >= 0.60:
            return 'HIGH'
        elif prob >= 0.55:
            return 'STANDARD'
        else:
            return 'LOW'

    df['effective_confidence'] = df.apply(get_effective_confidence, axis=1)
    df['effective_tier'] = df['effective_confidence'].apply(get_confidence_tier)

    # Update confidence column to use effective tier
    df['confidence'] = df['effective_tier']
    df['model_prob'] = df['effective_confidence']  # Use effective confidence for sorting

    logger.info(f"Loaded {len(df)} player prop recommendations")
    return df


def load_game_lines() -> pd.DataFrame:
    """Load game line recommendations - auto-detect most recent week."""
    # Find most recent game line recommendations file
    reports_dir = Path('reports')
    game_line_files = sorted(reports_dir.glob('WEEK*_GAME_LINE_RECOMMENDATIONS.csv'), reverse=True)

    if not game_line_files:
        logger.warning("No game line recommendations files found")
        return pd.DataFrame()

    # Use most recent file
    csv_path = game_line_files[0]
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} game line recommendations from {csv_path.name}")
    return df


def load_live_edges() -> pd.DataFrame:
    """Load most recent live edges from DraftKings comparison."""
    # Find most recent live edges file
    reports_dir = Path('reports')
    edge_files = sorted(reports_dir.glob('live_prop_edges_week*.csv'), reverse=True)

    if not edge_files:
        logger.warning("No live edges files found")
        return pd.DataFrame()

    # Load most recent
    latest_file = edge_files[0]
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} live edges from {latest_file.name}")
    return df


def fetch_live_edges(week: int) -> pd.DataFrame:
    """Fetch fresh live edges from DraftKings API."""
    try:
        from nfl_quant.data.draftkings_client import DKClient, find_edges
        from scripts.fetch.fetch_player_props_live import load_model_predictions

        logger.info("Fetching live odds from DraftKings...")
        model_preds = load_model_predictions(week)
        edges = find_edges(model_preds, min_edge=0.03, min_ev=0.02, variance_inflation=1.25)

        if not edges.empty:
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f'reports/live_prop_edges_week{week}_{timestamp}.csv')
            edges.to_csv(output_path, index=False)
            logger.info(f"Saved {len(edges)} live edges to {output_path}")

        return edges
    except Exception as e:
        logger.error(f"Failed to fetch live edges: {e}")
        return pd.DataFrame()


def clean_market_name(market: str) -> str:
    """Convert market to readable name."""
    mapping = {
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Receiving Yards',
        'player_receiving_yds': 'Receiving Yards',
        'player_rush_yds': 'Rushing Yards',
        'player_rush_attempts': 'Rush Attempts',
        'player_rush_reception_yds': 'Rush+Rec Yards',
        'player_pass_yds': 'Passing Yards',
        'player_pass_tds': 'Passing TDs',
        'player_pass_attempts': 'Pass Attempts',
    }
    return mapping.get(market, market)


def format_pick_row(row: pd.Series, columns: list, include_logic: bool = True) -> str:
    """Format a single pick as HTML row with expandable logic."""
    import hashlib

    conf_colors = {
        'ELITE': '#10b981',
        'HIGH': '#3b82f6',
        'STANDARD': '#6b7280',
        'LOW': '#9ca3af'
    }
    conf_color = conf_colors.get(row['confidence'], '#6b7280')

    edge = row['edge_pct']
    if edge >= 30:
        edge_color = '#10b981'
    elif edge >= 20:
        edge_color = '#3b82f6'
    elif edge >= 10:
        edge_color = '#f59e0b'
    else:
        edge_color = '#6b7280'

    # Get logic if available (check both 'model_reasoning' and 'logic' columns)
    logic_text = row.get('model_reasoning', row.get('logic', ''))
    has_logic = bool(logic_text) and include_logic

    # Create unique ID for this row
    row_id = hashlib.md5(f"{row['player']}{row['market']}{row['line']}".encode()).hexdigest()[:8]

    cells = []
    for col in columns:
        if col == 'player':
            # Add info icon if logic available
            player_cell = f'<td>{row["player"]}'
            if has_logic:
                player_cell += ' <span style="font-size: 11px; color: #667eea; cursor: help;" title="Click row for pick logic">ℹ️</span>'
            player_cell += '</td>'
            cells.append(player_cell)
        elif col == 'position':
            cells.append(f'<td>{row["position"]}</td>')
        elif col == 'team':
            cells.append(f'<td>{row["team"]}</td>')
        elif col == 'game':
            cells.append(f'<td style="font-size: 12px;">{row["game"]}</td>')
        elif col == 'market':
            cells.append(f'<td>{clean_market_name(row["market"])}</td>')
        elif col == 'pick':
            cells.append(
                f'<td><strong>{row["pick"]}</strong> {row["line"]}</td>'
            )
        elif col == 'projection':
            cells.append(f'<td>{row["model_projection"]:.1f}</td>')
        elif col == 'model_prob':
            prob = row['model_prob']
            # Color based on confidence level
            if prob >= 0.80:
                prob_color = '#10b981'  # Green for very high confidence
            elif prob >= 0.70:
                prob_color = '#3b82f6'  # Blue for high confidence
            elif prob >= 0.60:
                prob_color = '#f59e0b'  # Yellow for moderate
            else:
                prob_color = '#6b7280'  # Gray for lower
            prob_badge = f'<span style="background: {prob_color}; color: white; '
            prob_badge += f'padding: 2px 8px; border-radius: 4px; font-weight: bold;">{prob:.1%}</span>'
            cells.append(f'<td data-sort="{prob}">{prob_badge}</td>')
        elif col == 'edge':
            badge = f'<span style="background: {edge_color}; color: white; '
            badge += 'padding: 2px 8px; border-radius: 4px; '
            badge += f'font-weight: bold;">{edge:.1f}%</span>'
            cells.append(f'<td data-sort="{edge}">{badge}</td>')
        elif col == 'confidence':
            badge = f'<span style="background: {conf_color}; color: white; '
            badge += 'padding: 2px 8px; border-radius: 4px; '
            badge += f'font-size: 11px;">{row["confidence"]}</span>'
            cells.append(f'<td>{badge}</td>')
        elif col == 'kelly':
            # Convert kelly units to dollar amount (max $5 bet)
            # kelly_units is on a 0-5 scale roughly, normalize to max bet
            kelly_raw = row.get('kelly_units', 0) or 0
            # Scale: 5 units = max bet ($5), proportional below
            bet_dollars = min(kelly_raw, 5.0) / 5.0 * MAX_BET_DOLLARS
            bet_dollars = max(0.50, bet_dollars)  # Minimum $0.50 bet
            cells.append(f'<td data-sort="{bet_dollars}">${bet_dollars:.2f}</td>')

    # Main row (clickable if has logic)
    row_class = 'expandable-row' if has_logic else ''
    row_style = 'cursor: pointer;' if has_logic else ''
    onclick = f"toggleLogic('{row_id}')" if has_logic else ''

    main_row = f'<tr class="{row_class}" style="{row_style}" onclick="{onclick}">{"".join(cells)}</tr>'

    # Logic detail row (hidden by default)
    if has_logic:
        logic_row = f'''
        <tr class="logic-row" id="logic-{row_id}" style="display: none;">
            <td colspan="{len(columns)}" style="background: #f9fafb; padding: 12px 20px; border-left: 4px solid {conf_color};">
                <div style="font-size: 13px; color: #374151; line-height: 1.6;">
                    <strong style="color: #667eea;">📊 Pick Logic:</strong> {logic_text}
                </div>
            </td>
        </tr>
        '''
        return main_row + logic_row

    return main_row


def generate_top_picks_section(df: pd.DataFrame, n: int = 20) -> str:
    """Generate Top N picks section - sorted by confidence (model_prob)."""
    # Sort by model_prob (confidence percentage) descending
    top_picks = df.nlargest(n, 'model_prob')

    columns = ['player', 'position', 'game', 'market', 'pick', 'projection',
               'model_prob', 'edge', 'confidence', 'kelly']
    rows = ''.join([
        format_pick_row(row, columns)
        for _, row in top_picks.iterrows()
    ])

    return f"""
    <div class="view-section active" id="top-picks">
        <h2>🎯 Top {n} Picks (Highest Confidence)</h2>
        <p style="color: #6b7280; margin-bottom: 15px; font-size: 13px;">
            Sorted by model probability (confidence %). Click column headers to re-sort.
        </p>
        <table class="sortable-table">
            <thead>
                <tr>
                    <th onclick="sortTable(this, 0, 'string')">Player ⇅</th>
                    <th onclick="sortTable(this, 1, 'string')">Pos ⇅</th>
                    <th onclick="sortTable(this, 2, 'string')">Game ⇅</th>
                    <th onclick="sortTable(this, 3, 'string')">Prop ⇅</th>
                    <th onclick="sortTable(this, 4, 'string')">Pick ⇅</th>
                    <th onclick="sortTable(this, 5, 'number')">Projection ⇅</th>
                    <th onclick="sortTable(this, 6, 'number')" style="background: #eef2ff;">Confidence % ⇅</th>
                    <th onclick="sortTable(this, 7, 'number')">Edge % ⇅</th>
                    <th onclick="sortTable(this, 8, 'string')">Tier ⇅</th>
                    <th onclick="sortTable(this, 9, 'number')">Bet $ ⇅</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """


def generate_by_game_section(df: pd.DataFrame, schedule_df: pd.DataFrame = None) -> str:
    """Generate picks grouped by game (collapsible), sorted by game time."""
    games = df['game'].unique().tolist()

    # Build game info dict from schedule
    game_info = {}
    if schedule_df is not None and not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            game_key = f"{row['away_team']} @ {row['home_team']}"
            game_info[game_key] = {
                'gametime': row.get('gametime', ''),
                'weekday': row.get('weekday', ''),
                'gameday': row.get('gameday', ''),
            }

    # Sort games by gametime
    def get_game_sort_key(game):
        info = game_info.get(game, {})
        gametime = info.get('gametime', '99:99')
        gameday = info.get('gameday', '9999-99-99')
        return (gameday, gametime)

    games_sorted = sorted(games, key=get_game_sort_key)

    html_parts = [
        '<div class="view-section" id="by-game">',
        '<h2>🏈 Picks by Game</h2>',
        '<p style="color: #6b7280; margin-bottom: 15px; font-size: 13px;">',
        'Games sorted by kickoff time. Click column headers to sort within each game.',
        '</p>',
        '<p class="expand-collapse-controls">',
        '<button onclick="expandAllSections(\'game\')">Expand All</button>',
        '<button onclick="collapseAllSections(\'game\')">Collapse All</button>',
        '</p>'
    ]

    columns = ['player', 'position', 'market', 'pick',
               'projection', 'edge', 'confidence', 'kelly']

    for idx, game in enumerate(games_sorted):
        game_picks = df[df['game'] == game].sort_values(
            'edge_pct', ascending=False
        )

        avg_edge = game_picks['edge_pct'].mean()
        max_edge = game_picks['edge_pct'].max()
        elite_count = len(game_picks[game_picks['confidence'] == 'ELITE'])

        # Get game time info
        info = game_info.get(game, {})
        gametime = info.get('gametime', '')
        weekday = info.get('weekday', '')
        time_display = f"{weekday} {gametime}" if weekday and gametime else ""

        rows = ''.join([
            format_pick_row(row, columns)
            for _, row in game_picks.iterrows()
        ])

        html_parts.append(f"""
        <div class="collapsible-container">
            <h3 class="collapsible-header" onclick="toggleSection('game-{idx}')">
                <span class="toggle-icon" id="game-{idx}-icon">▼</span>
                {game}
                <span class="game-time-badge" style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;">{time_display}</span>
                <span class="header-stats">
                    {len(game_picks)} picks | Avg: {avg_edge:.1f}% |
                    Max: {max_edge:.1f}% | {elite_count} ELITE
                </span>
            </h3>
            <div class="collapsible-content" id="game-{idx}">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(this, 0, 'string')">Player ⇅</th>
                            <th onclick="sortTable(this, 1, 'string')">Pos ⇅</th>
                            <th onclick="sortTable(this, 2, 'string')">Market ⇅</th>
                            <th onclick="sortTable(this, 3, 'string')">Pick ⇅</th>
                            <th onclick="sortTable(this, 4, 'number')">Projection ⇅</th>
                            <th onclick="sortTable(this, 5, 'number')">Edge ⇅</th>
                            <th onclick="sortTable(this, 6, 'string')">Confidence ⇅</th>
                            <th onclick="sortTable(this, 7, 'number')">Bet $ ⇅</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """)

    html_parts.append('</div>')
    return '\n'.join(html_parts)


def generate_by_prop_section(df: pd.DataFrame) -> str:
    """Generate picks grouped by prop type (collapsible)."""
    market_counts = df.groupby('market').size().to_dict()
    markets = sorted(
        df['market'].unique(),
        key=lambda m: market_counts[m],
        reverse=True
    )

    html_parts = [
        '<div class="view-section" id="by-prop">',
        '<h2>📊 Picks by Prop Type</h2>',
        '<p style="color: #6b7280; margin-bottom: 15px; font-size: 13px;">',
        'Grouped by prop type. Click column headers to sort within each section.',
        '</p>',
        '<p class="expand-collapse-controls">',
        '<button onclick="expandAllSections(\'prop\')">Expand All</button>',
        '<button onclick="collapseAllSections(\'prop\')">Collapse All</button>',
        '</p>'
    ]

    columns = ['player', 'position', 'team', 'game', 'pick',
               'projection', 'edge', 'confidence', 'kelly']

    for idx, market in enumerate(markets):
        market_picks = df[df['market'] == market].sort_values(
            'edge_pct', ascending=False
        )
        market_clean = clean_market_name(market)

        avg_edge = market_picks['edge_pct'].mean()
        max_edge = market_picks['edge_pct'].max()
        elite_count = len(market_picks[market_picks['confidence'] == 'ELITE'])

        rows = ''.join([
            format_pick_row(row, columns)
            for _, row in market_picks.iterrows()
        ])

        html_parts.append(f"""
        <div class="collapsible-container">
            <h3 class="collapsible-header" onclick="toggleSection('prop-{idx}')">
                <span class="toggle-icon" id="prop-{idx}-icon">▼</span>
                {market_clean}
                <span class="header-stats">
                    {len(market_picks)} picks | Avg: {avg_edge:.1f}% |
                    Max: {max_edge:.1f}% | {elite_count} ELITE
                </span>
            </h3>
            <div class="collapsible-content" id="prop-{idx}">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(this, 0, 'string')">Player ⇅</th>
                            <th onclick="sortTable(this, 1, 'string')">Pos ⇅</th>
                            <th onclick="sortTable(this, 2, 'string')">Team ⇅</th>
                            <th onclick="sortTable(this, 3, 'string')">Game ⇅</th>
                            <th onclick="sortTable(this, 4, 'string')">Pick ⇅</th>
                            <th onclick="sortTable(this, 5, 'number')">Projection ⇅</th>
                            <th onclick="sortTable(this, 6, 'number')">Edge ⇅</th>
                            <th onclick="sortTable(this, 7, 'string')">Confidence ⇅</th>
                            <th onclick="sortTable(this, 8, 'number')">Bet $ ⇅</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """)

    html_parts.append('</div>')
    return '\n'.join(html_parts)


def generate_by_player_section(df: pd.DataFrame) -> str:
    """Generate picks grouped by player (collapsible, 2+ props only)."""
    player_counts = df.groupby('player').size()
    multi_prop_players = player_counts[player_counts >= 2].index.tolist()

    player_max_edge = df.groupby('player')['edge_pct'].max()
    players_sorted = sorted(
        multi_prop_players,
        key=lambda p: player_max_edge[p],
        reverse=True
    )

    html_parts = [
        '<div class="view-section" id="by-player">',
        '<h2>👤 Multi-Prop Players</h2>',
        '<p style="color: #6b7280; margin-bottom: 20px;">',
        'Players with multiple recommended props. Click column headers to sort.</p>',
        '<p class="expand-collapse-controls">',
        '<button onclick="expandAllSections(\'player\')">Expand All</button>',
        '<button onclick="collapseAllSections(\'player\')">Collapse All</button>',
        '</p>'
    ]

    columns = ['market', 'pick', 'projection',
               'model_prob', 'edge', 'confidence', 'kelly']

    for idx, player in enumerate(players_sorted):
        player_picks = df[df['player'] == player].sort_values(
            'edge_pct', ascending=False
        )

        first_row = player_picks.iloc[0]
        avg_edge = player_picks['edge_pct'].mean()
        max_edge = player_picks['edge_pct'].max()

        rows = ''.join([
            format_pick_row(row, columns)
            for _, row in player_picks.iterrows()
        ])

        html_parts.append(f"""
        <div class="collapsible-container">
            <h3 class="collapsible-header" onclick="toggleSection('player-{idx}')">
                <span class="toggle-icon" id="player-{idx}-icon">▼</span>
                {player}
                <span class="header-stats">
                    {first_row['position']} - {first_row['team']} -
                    {first_row['game']} | {len(player_picks)} picks |
                    Avg: {avg_edge:.1f}% | Max: {max_edge:.1f}%
                </span>
            </h3>
            <div class="collapsible-content" id="player-{idx}">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(this, 0, 'string')">Market ⇅</th>
                            <th onclick="sortTable(this, 1, 'string')">Pick ⇅</th>
                            <th onclick="sortTable(this, 2, 'number')">Projection ⇅</th>
                            <th onclick="sortTable(this, 3, 'number')">Model Prob ⇅</th>
                            <th onclick="sortTable(this, 4, 'number')">Edge ⇅</th>
                            <th onclick="sortTable(this, 5, 'string')">Confidence ⇅</th>
                            <th onclick="sortTable(this, 6, 'number')">Bet $ ⇅</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """)

    html_parts.append('</div>')
    return '\n'.join(html_parts)


def generate_stats_summary(df: pd.DataFrame) -> str:
    """Generate summary statistics."""
    total_picks = len(df)
    avg_edge = df['edge_pct'].mean()

    # Calculate total bet in dollars (each pick max $5)
    total_bet_dollars = total_picks * MAX_BET_DOLLARS

    conf_counts = df['confidence'].value_counts().to_dict()

    games = df['game'].nunique()
    markets = df['market'].nunique()
    players = df['player'].nunique()

    return f"""
    <div class="stats-summary">
        <div class="stat-card">
            <div class="stat-value">{total_picks}</div>
            <div class="stat-label">Total Picks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_edge:.1f}%</div>
            <div class="stat-label">Avg Edge</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">${total_bet_dollars:.0f}</div>
            <div class="stat-label">Max Total Bet</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{conf_counts.get('ELITE', 0)}</div>
            <div class="stat-label">ELITE Picks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{conf_counts.get('HIGH', 0)}</div>
            <div class="stat-label">HIGH Picks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{games}</div>
            <div class="stat-label">Games</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{markets}</div>
            <div class="stat-label">Prop Types</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{players}</div>
            <div class="stat-label">Players</div>
        </div>
    </div>
    """


def generate_game_lines_section(game_lines_df: pd.DataFrame, schedule_df: pd.DataFrame = None) -> str:
    """Generate game lines section with collapsible bet type sections (spreads, totals, moneylines)."""
    if game_lines_df.empty:
        return """
        <div class="view-section" id="game-lines">
            <h2>🎯 Game Lines</h2>
            <p style="color: #6b7280; font-style: italic;">No game line recommendations available</p>
        </div>
        """

    # Build game info dict from schedule for game times
    game_info = {}
    if schedule_df is not None and not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            game_key = f"{row['away_team']} @ {row['home_team']}"
            game_info[game_key] = {
                'gametime': row.get('gametime', ''),
                'weekday': row.get('weekday', ''),
                'gameday': row.get('gameday', ''),
            }

    html_parts = [
        '<div class="view-section" id="game-lines">',
        '<h2>🎯 Game Lines (Spreads, Totals, Moneylines)</h2>',
    ]

    # Overall summary stats
    total_lines = len(game_lines_df)
    avg_edge = game_lines_df['edge_pct'].mean()
    max_edge = game_lines_df['edge_pct'].max()
    elite_count = len(game_lines_df[game_lines_df['confidence_tier'] == 'ELITE'])

    html_parts.append(f"""
    <div class="stats-summary" style="margin-bottom: 30px;">
        <div class="stat-card">
            <div class="stat-value">{total_lines}</div>
            <div class="stat-label">Total Lines</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_edge:.1f}%</div>
            <div class="stat-label">Avg Edge</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{max_edge:.1f}%</div>
            <div class="stat-label">Max Edge</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{elite_count}</div>
            <div class="stat-label">ELITE Picks</div>
        </div>
    </div>
    """)

    # Expand/Collapse controls
    html_parts.append("""
    <div class="expand-collapse-controls">
        <button onclick="expandAllSections('gamelines')">Expand All</button>
        <button onclick="collapseAllSections('gamelines')">Collapse All</button>
    </div>
    """)

    # Group by bet_type and create collapsible sections
    bet_types = [
        ('spread', 'Spreads', '📈'),
        ('total', 'Totals', '🎯'),
        ('moneyline', 'Moneylines', '💰')
    ]

    for bet_type, bet_name, emoji in bet_types:
        # Filter data for this bet type
        type_df = game_lines_df[game_lines_df['bet_type'] == bet_type].copy()

        if type_df.empty:
            continue

        # Stats for this bet type
        count = len(type_df)
        avg_edge_type = type_df['edge_pct'].mean()
        max_edge_type = type_df['edge_pct'].max()
        elite_count_type = len(type_df[type_df['confidence_tier'] == 'ELITE'])

        # Collapsible container
        section_id = f'gamelines-{bet_type}'
        html_parts.append(f"""
        <div class="collapsible-container">
            <div class="collapsible-header" onclick="toggleSection('{section_id}')">
                <span class="toggle-icon" id="{section_id}-icon">▼</span>
                <h3 style="margin: 0;">{emoji} {bet_name}</h3>
                <span class="header-stats">
                    {count} picks • Avg Edge: {avg_edge_type:.1f}% • Max: {max_edge_type:.1f}%
                    {f' • {elite_count_type} ELITE' if elite_count_type > 0 else ''}
                </span>
            </div>
            <div class="collapsible-content" id="{section_id}">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(this, 0, 'string')">Game ⇅</th>
                            <th onclick="sortTable(this, 1, 'string')">Time ⇅</th>
                            <th onclick="sortTable(this, 2, 'string')">Pick ⇅</th>
                            <th onclick="sortTable(this, 3, 'number')">Line ⇅</th>
                            <th onclick="sortTable(this, 4, 'number')">Model Fair ⇅</th>
                            <th onclick="sortTable(this, 5, 'number')">Win Prob ⇅</th>
                            <th onclick="sortTable(this, 6, 'number')">Edge ⇅</th>
                            <th onclick="sortTable(this, 7, 'number')">Kelly ⇅</th>
                            <th onclick="sortTable(this, 8, 'string')">Conf ⇅</th>
                        </tr>
                    </thead>
                    <tbody>
        """)

        # Sort by game time, then by edge descending
        def get_game_sort_key(game):
            info = game_info.get(game, {})
            gametime = info.get('gametime', '99:99')
            gameday = info.get('gameday', '9999-99-99')
            return (gameday, gametime)

        type_df['sort_key'] = type_df['game'].apply(get_game_sort_key)
        sorted_df = type_df.sort_values(['sort_key', 'edge_pct'], ascending=[True, False])

        for _, row in sorted_df.iterrows():
            # Get game time info
            info = game_info.get(row['game'], {})
            gametime = info.get('gametime', '')
            weekday = info.get('weekday', '')
            time_display = f"{weekday[:3]} {gametime}" if weekday and gametime else "-"

            # Color coding
            conf_colors = {
                'ELITE': '#10b981',
                'HIGH': '#3b82f6',
                'STANDARD': '#6b7280',
                'LOW': '#9ca3af'
            }
            conf_color = conf_colors.get(row['confidence_tier'], '#6b7280')

            edge = row['edge_pct']
            if edge >= 30:
                edge_color = '#10b981'
            elif edge >= 20:
                edge_color = '#3b82f6'
            elif edge >= 10:
                edge_color = '#f59e0b'
            else:
                edge_color = '#6b7280'

            edge_badge = f'<span style="background: {edge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{edge:.1f}%</span>'
            conf_badge = f'<span style="background: {conf_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{row["confidence_tier"]}</span>'

            # Handle display values for moneylines (no line value)
            line_display = '-' if pd.isna(row['market_line']) else row['market_line']
            line_sort = 0 if pd.isna(row['market_line']) else row['market_line']
            fair_line_display = '-' if pd.isna(row['model_fair_line']) else f"{row['model_fair_line']:.1f}"
            fair_line_sort = 0 if pd.isna(row['model_fair_line']) else row['model_fair_line']

            html_parts.append(f"""
            <tr>
                <td style="font-size: 12px;">{row['game']}</td>
                <td style="font-size: 11px; color: #667eea;">{time_display}</td>
                <td><strong>{row['pick']}</strong></td>
                <td data-sort="{line_sort}">{line_display}</td>
                <td data-sort="{fair_line_sort}">{fair_line_display}</td>
                <td data-sort="{row['model_prob']}">{row['model_prob']:.1%}</td>
                <td data-sort="{edge}">{edge_badge}</td>
                <td data-sort="{row['recommended_units']}">{row['recommended_units']:.1f}u</td>
                <td data-sort="{row['confidence_tier']}">{conf_badge}</td>
            </tr>
            """)

        html_parts.append("""
                    </tbody>
                </table>
            </div>
        </div>
        """)

    html_parts.append('</div>')  # Close view-section

    return '\n'.join(html_parts)


def generate_live_edges_section(live_edges_df: pd.DataFrame) -> str:
    """Generate live edges section showing real-time DraftKings comparison."""
    if live_edges_df.empty:
        return """
        <div class="view-section" id="live-edges">
            <h2>⚡ Live Edges (DraftKings)</h2>
            <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <p style="color: #92400e; margin: 0;">
                    <strong>No live edges available.</strong><br><br>
                    Run with <code>--fetch-live</code> to fetch real-time odds from DraftKings:<br>
                    <code style="background: #1f2937; color: #10b981; padding: 4px 8px; border-radius: 4px;">
                    python scripts/dashboard/generate_multiview_dashboard.py --fetch-live
                    </code>
                </p>
            </div>
        </div>
        """

    html_parts = [
        '<div class="view-section" id="live-edges">',
        '<h2>⚡ Live Edges vs DraftKings</h2>',
    ]

    # Summary stats
    total_edges = len(live_edges_df)
    elite_edges = len(live_edges_df[live_edges_df['edge'] >= 0.10])
    high_edges = len(live_edges_df[(live_edges_df['edge'] >= 0.05) & (live_edges_df['edge'] < 0.10)])
    avg_edge = live_edges_df['edge'].mean() * 100
    total_kelly = live_edges_df['kelly'].sum() * 100
    best_ev = live_edges_df['ev'].max() * 100

    html_parts.append(f"""
    <div class="stats-summary" style="margin-bottom: 30px;">
        <div class="stat-card" style="background: linear-gradient(135deg, #10b981, #059669);">
            <div class="stat-value" style="color: white;">{total_edges}</div>
            <div class="stat-label" style="color: rgba(255,255,255,0.8);">Live Edges</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">{elite_edges}</div>
            <div class="stat-label">Elite (10%+)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #3b82f6;">{high_edges}</div>
            <div class="stat-label">High (5-10%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_edge:.1f}%</div>
            <div class="stat-label">Avg Edge</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_ev:.1f}%</div>
            <div class="stat-label">Best EV</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_kelly:.1f}%</div>
            <div class="stat-label">Total Kelly</div>
        </div>
    </div>
    """)

    # Info banner
    html_parts.append("""
    <div style="background: #eff6ff; border: 1px solid #3b82f6; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
        <p style="color: #1e40af; margin: 0; font-size: 13px;">
            <strong>📊 Live DraftKings Comparison</strong> - These edges are calculated by comparing
            NFL QUANT model predictions to real-time DraftKings player prop lines.
            Edges reflect the difference between model probability and market implied probability (vig removed).
        </p>
    </div>
    """)

    # Elite edges section
    elite_df = live_edges_df[live_edges_df['edge'] >= 0.10].sort_values('ev', ascending=False)
    if not elite_df.empty:
        html_parts.append("""
        <div class="collapsible-container">
            <div class="collapsible-header" onclick="toggleSection('live-elite')">
                <span class="toggle-icon" id="live-elite-icon">▼</span>
                <h3 style="margin: 0;">🔥 Elite Edges (10%+)</h3>
                <span class="header-stats">""" + f"{len(elite_df)} picks" + """</span>
            </div>
            <div class="collapsible-content" id="live-elite">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th onclick="sortTable(this, 0, 'string')">Player ⇅</th>
                            <th onclick="sortTable(this, 1, 'string')">Market ⇅</th>
                            <th onclick="sortTable(this, 2, 'string')">Pick ⇅</th>
                            <th onclick="sortTable(this, 3, 'number')">Line ⇅</th>
                            <th onclick="sortTable(this, 4, 'number')">Model Prob ⇅</th>
                            <th onclick="sortTable(this, 5, 'number')">DK Prob ⇅</th>
                            <th onclick="sortTable(this, 6, 'number')">Edge ⇅</th>
                            <th onclick="sortTable(this, 7, 'number')">EV ⇅</th>
                            <th onclick="sortTable(this, 8, 'number')">Kelly ⇅</th>
                            <th onclick="sortTable(this, 9, 'string')">Matchup ⇅</th>
                        </tr>
                    </thead>
                    <tbody>
        """)

        for _, row in elite_df.iterrows():
            edge_pct = row['edge'] * 100
            ev_pct = row['ev'] * 100
            kelly_pct = row['kelly'] * 100
            market_clean = row['market'].replace('player_', '').replace('_', ' ').title()

            edge_badge = f'<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold;">{edge_pct:.1f}%</span>'
            ev_badge = f'<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px;">{ev_pct:.1f}%</span>'

            html_parts.append(f"""
                <tr>
                    <td><strong>{row['player']}</strong></td>
                    <td>{market_clean}</td>
                    <td><strong>{row['direction']}</strong></td>
                    <td>{row['line']}</td>
                    <td>{row['model_prob']*100:.1f}%</td>
                    <td>{row['dk_prob']*100:.1f}%</td>
                    <td data-sort="{edge_pct}">{edge_badge}</td>
                    <td data-sort="{ev_pct}">{ev_badge}</td>
                    <td>{kelly_pct:.2f}%</td>
                    <td style="font-size: 12px;">{row['matchup']}</td>
                </tr>
            """)

        html_parts.append("""
                    </tbody>
                </table>
            </div>
        </div>
        """)

    # High edges section
    high_df = live_edges_df[(live_edges_df['edge'] >= 0.05) & (live_edges_df['edge'] < 0.10)].sort_values('ev', ascending=False)
    if not high_df.empty:
        html_parts.append("""
        <div class="collapsible-container">
            <div class="collapsible-header" onclick="toggleSection('live-high')">
                <span class="toggle-icon" id="live-high-icon">▼</span>
                <h3 style="margin: 0;">✅ High Edges (5-10%)</h3>
                <span class="header-stats">""" + f"{len(high_df)} picks" + """</span>
            </div>
            <div class="collapsible-content" id="live-high">
                <table class="sortable-table">
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Market</th>
                            <th>Pick</th>
                            <th>Line</th>
                            <th>Model</th>
                            <th>DK</th>
                            <th>Edge</th>
                            <th>EV</th>
                            <th>Kelly</th>
                        </tr>
                    </thead>
                    <tbody>
        """)

        for _, row in high_df.head(20).iterrows():
            edge_pct = row['edge'] * 100
            ev_pct = row['ev'] * 100
            kelly_pct = row['kelly'] * 100
            market_clean = row['market'].replace('player_', '').replace('_', ' ').title()

            edge_badge = f'<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px;">{edge_pct:.1f}%</span>'

            html_parts.append(f"""
                <tr>
                    <td>{row['player']}</td>
                    <td>{market_clean}</td>
                    <td><strong>{row['direction']}</strong></td>
                    <td>{row['line']}</td>
                    <td>{row['model_prob']*100:.1f}%</td>
                    <td>{row['dk_prob']*100:.1f}%</td>
                    <td data-sort="{edge_pct}">{edge_badge}</td>
                    <td>{ev_pct:.1f}%</td>
                    <td>{kelly_pct:.2f}%</td>
                </tr>
            """)

        html_parts.append("""
                    </tbody>
                </table>
            </div>
        </div>
        """)

    # Standard edges section (collapsed by default)
    standard_df = live_edges_df[(live_edges_df['edge'] >= 0.03) & (live_edges_df['edge'] < 0.05)].sort_values('ev', ascending=False)
    if not standard_df.empty:
        html_parts.append("""
        <div class="collapsible-container">
            <div class="collapsible-header" onclick="toggleSection('live-standard')">
                <span class="toggle-icon collapsed" id="live-standard-icon">▼</span>
                <h3 style="margin: 0;">📊 Standard Edges (3-5%)</h3>
                <span class="header-stats">""" + f"{len(standard_df)} picks" + """</span>
            </div>
            <div class="collapsible-content collapsed" id="live-standard">
                <table>
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Market</th>
                            <th>Pick</th>
                            <th>Line</th>
                            <th>Edge</th>
                            <th>EV</th>
                        </tr>
                    </thead>
                    <tbody>
        """)

        for _, row in standard_df.head(30).iterrows():
            edge_pct = row['edge'] * 100
            ev_pct = row['ev'] * 100
            market_clean = row['market'].replace('player_', '').replace('_', ' ').title()

            html_parts.append(f"""
                <tr>
                    <td>{row['player']}</td>
                    <td>{market_clean}</td>
                    <td>{row['direction']}</td>
                    <td>{row['line']}</td>
                    <td>{edge_pct:.1f}%</td>
                    <td>{ev_pct:.1f}%</td>
                </tr>
            """)

        html_parts.append("""
                    </tbody>
                </table>
            </div>
        </div>
        """)

    html_parts.append('</div>')  # Close view-section
    return '\n'.join(html_parts)


def generate_html(df: pd.DataFrame, game_lines_df: pd.DataFrame = None, live_edges_df: pd.DataFrame = None, schedule_df: pd.DataFrame = None) -> str:
    """Generate complete HTML dashboard."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NFL QUANT - Multi-View Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}

            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}

            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}

            .header h1 {{
                font-size: 32px;
                margin-bottom: 10px;
            }}

            .header p {{
                opacity: 0.9;
                font-size: 14px;
            }}

            .stats-summary {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                padding: 30px;
                background: #f9fafb;
                border-bottom: 2px solid #e5e7eb;
            }}

            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}

            .stat-label {{
                font-size: 12px;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .tabs {{
                display: flex;
                background: #f9fafb;
                border-bottom: 2px solid #e5e7eb;
                padding: 0 30px;
                overflow-x: auto;
            }}

            .tab {{
                padding: 15px 25px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 14px;
                font-weight: 600;
                color: #6b7280;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
                white-space: nowrap;
            }}

            .tab:hover {{
                color: #667eea;
                background: rgba(102, 126, 234, 0.1);
            }}

            .tab.active {{
                color: #667eea;
                border-bottom-color: #667eea;
            }}

            .view-section {{
                display: none;
                padding: 30px;
            }}

            .view-section.active {{
                display: block;
            }}

            h2 {{
                font-size: 24px;
                margin-bottom: 20px;
                color: #1f2937;
            }}

            h3 {{
                font-size: 18px;
                margin-bottom: 15px;
                color: #374151;
            }}

            .expand-collapse-controls {{
                margin-bottom: 20px;
            }}

            .expand-collapse-controls button {{
                background: #667eea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
                font-weight: 600;
                margin-right: 10px;
                transition: all 0.3s;
            }}

            .expand-collapse-controls button:hover {{
                background: #5568d3;
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            }}

            .collapsible-container {{
                margin-bottom: 20px;
                background: #f9fafb;
                border-radius: 8px;
                overflow: hidden;
            }}

            .collapsible-header {{
                padding: 15px 20px;
                cursor: pointer;
                user-select: none;
                display: flex;
                align-items: center;
                gap: 12px;
                transition: background 0.3s;
            }}

            .collapsible-header:hover {{
                background: #f3f4f6;
            }}

            .toggle-icon {{
                font-size: 14px;
                transition: transform 0.3s;
                display: inline-block;
            }}

            .toggle-icon.collapsed {{
                transform: rotate(-90deg);
            }}

            .header-stats {{
                color: #6b7280;
                font-size: 14px;
                font-weight: normal;
                margin-left: auto;
            }}

            .collapsible-content {{
                max-height: 2000px;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
            }}

            .collapsible-content.collapsed {{
                max-height: 0;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            thead {{
                background: #f9fafb;
            }}

            th {{
                padding: 12px 15px;
                text-align: left;
                font-size: 12px;
                font-weight: 600;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #e5e7eb;
                cursor: pointer;
                user-select: none;
                transition: background 0.2s;
            }}

            th:hover {{
                background: #f3f4f6;
            }}

            th.sorted-asc::after {{
                content: ' ↑';
                color: #667eea;
            }}

            th.sorted-desc::after {{
                content: ' ↓';
                color: #667eea;
            }}

            td {{
                padding: 12px 15px;
                border-bottom: 1px solid #f3f4f6;
                font-size: 14px;
                color: #374151;
            }}

            tbody tr:hover {{
                background: #f9fafb;
            }}

            .expandable-row:hover {{
                background: #eef2ff !important;
            }}

            .logic-row {{
                border-top: none !important;
            }}

            .logic-row td {{
                border-bottom: 2px solid #e5e7eb !important;
            }}

            .info-icon {{
                display: inline-block;
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{
                    opacity: 1;
                }}
                50% {{
                    opacity: 0.5;
                }}
            }}

            .footer {{
                padding: 20px;
                text-align: center;
                background: #f9fafb;
                border-top: 2px solid #e5e7eb;
                color: #6b7280;
                font-size: 12px;
            }}

            @media (max-width: 768px) {{
                .stats-summary {{
                    grid-template-columns: repeat(2, 1fr);
                }}

                table {{
                    font-size: 12px;
                }}

                th, td {{
                    padding: 8px;
                }}

                .header-stats {{
                    display: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL QUANT Multi-View Dashboard</h1>
                <p>Generated {timestamp}</p>
            </div>

            {generate_stats_summary(df)}

            <div class="tabs">
                <button class="tab active" onclick="showView('top-picks')">🔥 Top Picks</button>
                <button class="tab" onclick="showView('by-game')">🏈 By Game</button>
                <button class="tab" onclick="showView('by-prop')">📊 By Prop Type</button>
                <button class="tab" onclick="showView('by-player')">👤 By Player</button>
                <button class="tab" onclick="showView('game-lines')">🎯 Game Lines</button>
            </div>

            {generate_top_picks_section(df)}
            {generate_by_game_section(df, schedule_df)}
            {generate_by_prop_section(df)}
            {generate_by_player_section(df)}
            {generate_game_lines_section(game_lines_df if game_lines_df is not None else pd.DataFrame(), schedule_df)}

            <div class="footer">
                NFL QUANT Framework v3.0 | {len(df)} recommendations generated |
                Data sourced from NFLverse & DraftKings
            </div>
        </div>

        <script>
            function showView(viewId) {{
                // Hide all views
                document.querySelectorAll('.view-section').forEach(section => {{
                    section.classList.remove('active');
                }});

                // Remove active from all tabs
                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.classList.remove('active');
                }});

                // Show selected view
                document.getElementById(viewId).classList.add('active');

                // Mark tab as active
                event.target.classList.add('active');
            }}

            function toggleSection(sectionId) {{
                const content = document.getElementById(sectionId);
                const icon = document.getElementById(sectionId + '-icon');

                if (content.classList.contains('collapsed')) {{
                    content.classList.remove('collapsed');
                    icon.classList.remove('collapsed');
                }} else {{
                    content.classList.add('collapsed');
                    icon.classList.add('collapsed');
                }}
            }}

            function expandAllSections(prefix) {{
                document.querySelectorAll('[id^="' + prefix + '-"]').forEach(el => {{
                    if (el.classList.contains('collapsible-content')) {{
                        el.classList.remove('collapsed');
                    }}
                    if (el.classList.contains('toggle-icon')) {{
                        el.classList.remove('collapsed');
                    }}
                }});
            }}

            function collapseAllSections(prefix) {{
                document.querySelectorAll('[id^="' + prefix + '-"]').forEach(el => {{
                    if (el.classList.contains('collapsible-content')) {{
                        el.classList.add('collapsed');
                    }}
                    if (el.classList.contains('toggle-icon')) {{
                        el.classList.add('collapsed');
                    }}
                }});
            }}

            // Toggle logic row visibility
            function toggleLogic(rowId) {{
                const logicRow = document.getElementById('logic-' + rowId);
                if (logicRow) {{
                    if (logicRow.style.display === 'none') {{
                        logicRow.style.display = 'table-row';
                    }} else {{
                        logicRow.style.display = 'none';
                    }}
                }}
            }}

            // Table sorting function
            function sortTable(headerElement, columnIndex, dataType) {{
                const table = headerElement.closest('table');
                const tbody = table.querySelector('tbody');

                // Get all rows, but separate data rows from logic rows
                const allRows = Array.from(tbody.querySelectorAll('tr'));
                const dataRows = allRows.filter(row => !row.classList.contains('logic-row'));

                // Build a map of data rows to their associated logic rows
                const rowPairs = [];
                dataRows.forEach(row => {{
                    // Find associated logic row (if any)
                    const nextRow = row.nextElementSibling;
                    const logicRow = (nextRow && nextRow.classList.contains('logic-row')) ? nextRow : null;
                    rowPairs.push({{ dataRow: row, logicRow: logicRow }});
                }});

                // Determine sort direction
                const currentDirection = headerElement.dataset.sortDirection || 'desc';
                const newDirection = currentDirection === 'asc' ? 'desc' : 'asc';

                // Sort row pairs based on data row values
                rowPairs.sort((a, b) => {{
                    const cellA = a.dataRow.children[columnIndex];
                    const cellB = b.dataRow.children[columnIndex];

                    if (!cellA || !cellB) return 0;

                    // Get sort values (use data-sort attribute if available, otherwise text content)
                    let valueA = cellA.dataset.sort || cellA.textContent.trim();
                    let valueB = cellB.dataset.sort || cellB.textContent.trim();

                    // Convert to numbers if dataType is number
                    if (dataType === 'number') {{
                        // Remove % and other symbols
                        valueA = parseFloat(valueA.replace(/[^0-9.-]/g, '')) || 0;
                        valueB = parseFloat(valueB.replace(/[^0-9.-]/g, '')) || 0;
                    }} else {{
                        // String comparison (case-insensitive)
                        valueA = valueA.toLowerCase();
                        valueB = valueB.toLowerCase();
                    }}

                    // Compare
                    if (valueA < valueB) return newDirection === 'asc' ? -1 : 1;
                    if (valueA > valueB) return newDirection === 'asc' ? 1 : -1;
                    return 0;
                }});

                // Clear tbody and re-append sorted rows with their logic rows
                while (tbody.firstChild) {{
                    tbody.removeChild(tbody.firstChild);
                }}

                rowPairs.forEach(pair => {{
                    tbody.appendChild(pair.dataRow);
                    if (pair.logicRow) {{
                        tbody.appendChild(pair.logicRow);
                    }}
                }});

                // Update sort direction
                headerElement.dataset.sortDirection = newDirection;

                // Update visual indicator - reset all headers then highlight current
                const allHeaders = table.querySelectorAll('th');
                allHeaders.forEach(th => {{
                    th.style.background = '';
                    th.classList.remove('sorted-asc', 'sorted-desc');
                }});
                headerElement.style.background = '#eef2ff';
                headerElement.classList.add(newDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
            }}
        </script>
    </body>
    </html>
    """

    return html


def main():
    """Main execution."""
    logger.info("Generating multi-view dashboard...")

    # Load data
    df = load_recommendations()
    game_lines_df = load_game_lines()

    # Get week from recommendations if available
    week = df['week'].iloc[0] if 'week' in df.columns and len(df) > 0 else None
    schedule_df = load_schedule(week=week)

    # Generate HTML
    html = generate_html(df, game_lines_df, schedule_df=schedule_df)

    # Save
    output_path = Path('reports/multiview_dashboard.html')
    output_path.write_text(html)

    logger.info(f"✅ Dashboard saved to {output_path}")
    logger.info(f"   Player props: {len(df)}")
    if not game_lines_df.empty:
        logger.info(f"   Game lines: {len(game_lines_df)}")
    logger.info(f"   Open: file://{output_path.absolute()}")


if __name__ == '__main__':
    main()
