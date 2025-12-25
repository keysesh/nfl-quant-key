"""
Game Line Edge Implementation (V29)

EPA-based edge detection for spreads and totals with proper calibration.

Key characteristics:
- EPA-based power ratings for spreads
- Pace × Efficiency totals model (unit-consistent)
- Weather/roof adjustments for totals
- Explicit sign conventions with clear variable naming
- Logging and observability for debugging

Sign Conventions (CRITICAL):
- off_epa: EPA per play when team has the ball
    - Positive = good offense (generating expected points)
    - Negative = bad offense
- def_epa_allowed: EPA per play allowed to opponents when defending
    - Positive = bad defense (allowing expected points)
    - Negative = good defense (limiting expected points)

For TOTALS:
- Higher combined_off_epa → MORE scoring
- Higher combined_def_epa_allowed → MORE scoring (worse defenses)
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set up logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# League averages (calibrated from 2025 data)
LEAGUE_AVG_PPG = 23.3  # Average points per game per team
LEAGUE_AVG_PACE = 60.0  # Average plays per team per game
LEAGUE_AVG_TOTAL = 46.6  # Average combined score

# Edge thresholds
MIN_SPREAD_EDGE = 2.0  # Minimum point edge to trigger spread bet
MIN_TOTAL_EDGE_PTS = 3.0  # Minimum point edge to trigger total bet

# Weather adjustments for totals (empirically derived)
WEATHER_ADJUSTMENTS = {
    'dome': 1.5,      # Dome games score ~1.5 pts higher
    'cold': -2.0,     # < 32°F reduces scoring
    'wind': -0.3,     # Per mph over 15 mph
    'rain': -1.5,     # Precipitation reduces scoring
}

# V26: Load calibrated EPA factor (fallback to defaults if not calibrated)
_CALIBRATION_PATH = Path(__file__).parent.parent.parent / 'data' / 'models' / 'calibrated_epa_factor.joblib'
_TOTALS_CALIBRATION_PATH = Path(__file__).parent.parent.parent / 'data' / 'models' / 'calibrated_totals_factor.joblib'


def _load_calibrated_values() -> dict:
    """Load calibrated values from file, or use defaults."""
    defaults = {
        'epa_to_points_factor': 3.5,
        'home_field_advantage': 2.5,
        'per_day_rest_value': 0.3,
        'divisional_shrinkage': 0.7,
    }

    if _CALIBRATION_PATH.exists():
        try:
            calibration = joblib.load(_CALIBRATION_PATH)
            return {
                'epa_to_points_factor': calibration.get('epa_to_points_factor', defaults['epa_to_points_factor']),
                'home_field_advantage': calibration.get('home_field_advantage', defaults['home_field_advantage']),
                'per_day_rest_value': calibration.get('per_day_rest_value', defaults['per_day_rest_value']),
                'divisional_shrinkage': calibration.get('divisional_shrinkage', defaults['divisional_shrinkage']),
            }
        except Exception:
            pass

    return defaults


def _load_totals_calibration() -> dict:
    """Load totals-specific calibration, or use defaults."""
    defaults = {
        'points_per_play': 0.38,  # Calibrated: 46.6 / 121.4 = 0.3835
        'epa_total_factor': 5.0,  # How much EPA affects totals
        'pace_weight': 0.6,       # Weight on pace-based projection
        'epa_weight': 0.4,        # Weight on EPA-based adjustment
    }

    if _TOTALS_CALIBRATION_PATH.exists():
        try:
            calibration = joblib.load(_TOTALS_CALIBRATION_PATH)
            return {
                'points_per_play': calibration.get('points_per_play', defaults['points_per_play']),
                'epa_total_factor': calibration.get('epa_total_factor', defaults['epa_total_factor']),
                'pace_weight': calibration.get('pace_weight', defaults['pace_weight']),
                'epa_weight': calibration.get('epa_weight', defaults['epa_weight']),
            }
        except Exception:
            pass

    return defaults


# Load calibrated values at module import
_CALIBRATED = _load_calibrated_values()
_TOTALS_CALIBRATED = _load_totals_calibration()

EPA_TO_POINTS_FACTOR = _CALIBRATED['epa_to_points_factor']
HOME_FIELD_ADVANTAGE = _CALIBRATED['home_field_advantage']

# Rest days adjustments (proven +/- point swing)
REST_ADJUSTMENTS = {
    3: -1.5,   # Short week (Thursday game) - team underperforms
    4: 0.0,   # Normal rest
    5: 0.5,   # Extended rest
    6: 0.5,   # Extended rest
    7: 1.0,   # Full week + extra day
    10: 1.5,  # Coming off bye week
    11: 1.5,  # Coming off bye week
    12: 1.5,  # Coming off bye week
    13: 1.5,  # Coming off bye week
    14: 1.5,  # Coming off bye week
}

# Divisional game adjustment (games tend to be closer)
DIVISIONAL_SPREAD_SHRINK = _CALIBRATED['divisional_shrinkage']


class GameLineEdge:
    """
    EPA-based game line predictions with proper totals modeling.

    Uses team offensive/defensive EPA to:
    1. Calculate power ratings for spread predictions
    2. Project totals using Pace × Efficiency model

    V29 Changes:
    - Unit-consistent totals: plays_total × points_per_play
    - Explicit sign conventions with clear variable naming
    - Weather/roof adjustments for totals
    - Logging for observability
    """

    def __init__(self, verbose: bool = False):
        """Initialize game line edge with default parameters."""
        self.home_field_advantage = HOME_FIELD_ADVANTAGE
        self.min_spread_edge = MIN_SPREAD_EDGE
        self.min_total_edge_pts = MIN_TOTAL_EDGE_PTS
        self.team_stats: Dict[str, Dict] = {}
        self.version = "2.0"
        self.verbose = verbose

        # Totals calibration parameters
        self.points_per_play = _TOTALS_CALIBRATED['points_per_play']
        self.epa_total_factor = _TOTALS_CALIBRATED['epa_total_factor']
        self.pace_weight = _TOTALS_CALIBRATED['pace_weight']
        self.epa_weight = _TOTALS_CALIBRATED['epa_weight']

    def calculate_team_epa(
        self,
        pbp_df: pd.DataFrame,
        team: str,
        current_week: int,
        lookback_weeks: int = 6
    ) -> Dict[str, float]:
        """
        Calculate team EPA metrics from play-by-play data.

        SIGN CONVENTIONS:
        - off_epa: EPA per play on offense (positive = good)
        - def_epa_allowed: EPA per play allowed on defense (positive = bad defense)

        Args:
            pbp_df: Play-by-play DataFrame with 'epa', 'posteam', 'defteam', 'week'
            team: Team abbreviation
            current_week: Current week (for filtering)
            lookback_weeks: Number of weeks to look back

        Returns:
            Dict with 'off_epa', 'def_epa_allowed', 'pace', 'games'
        """
        # Filter to recent weeks (before current week to prevent leakage)
        min_week = max(1, current_week - lookback_weeks)
        recent_pbp = pbp_df[
            (pbp_df['week'] >= min_week) &
            (pbp_df['week'] < current_week) &
            (pbp_df['play_type'].isin(['pass', 'run']))
        ]

        # Offensive EPA (when team has the ball)
        off_plays = recent_pbp[recent_pbp['posteam'] == team]

        # Defensive EPA ALLOWED (EPA scored by opponents against this team)
        # Higher value = worse defense (allowing more expected points)
        def_plays = recent_pbp[recent_pbp['defteam'] == team]

        if len(off_plays) == 0 or len(def_plays) == 0:
            return {
                'off_epa': 0.0,
                'def_epa_allowed': 0.0,  # Renamed for clarity
                'def_epa': 0.0,  # Keep for backwards compatibility
                'pace': LEAGUE_AVG_PACE,
                'games': 0
            }

        # Calculate EPA per play
        off_epa = off_plays['epa'].mean()
        def_epa_allowed = def_plays['epa'].mean()  # EPA allowed to opponents

        # Calculate pace (plays per game)
        games = len(off_plays['game_id'].unique())
        pace = len(off_plays) / games if games > 0 else LEAGUE_AVG_PACE

        # Regress EPA toward 0 for small sample sizes
        # After 6 games, use mostly raw EPA; fewer games = more regression
        regression_factor = min(1.0, games / 6)
        off_epa = off_epa * regression_factor
        def_epa_allowed = def_epa_allowed * regression_factor

        return {
            'off_epa': off_epa,
            'def_epa_allowed': def_epa_allowed,  # Explicit naming
            'def_epa': def_epa_allowed,  # Backwards compatibility
            'pace': pace,
            'games': games
        }

    def calculate_spread_edge(
        self,
        home_epa: Dict[str, float],
        away_epa: Dict[str, float],
        market_spread: float,
        home_rest: int = 7,
        away_rest: int = 7,
        is_divisional: bool = False
    ) -> Tuple[Optional[str], float, float]:
        """
        Calculate spread edge from EPA power ratings.

        Model spread = -(home_power - away_power + home_field + rest_adj)
        Edge = model_spread - market_spread

        Uses matchup-based EPA differential (appropriate for spreads).

        Args:
            home_epa: Home team EPA stats
            away_epa: Away team EPA stats
            market_spread: Market spread (negative = home favored)
            home_rest: Days since home team's last game
            away_rest: Days since away team's last game
            is_divisional: Whether this is a divisional matchup

        Returns:
            Tuple of (direction, edge_points, confidence)
            direction: 'HOME' or 'AWAY' or None if no edge
        """
        # Get EPA values (support both old and new naming)
        home_off = home_epa['off_epa']
        away_off = away_epa['off_epa']
        home_def_allowed = home_epa.get('def_epa_allowed', home_epa.get('def_epa', 0))
        away_def_allowed = away_epa.get('def_epa_allowed', away_epa.get('def_epa', 0))

        # Power rating differential for SPREADS
        # home_power = how well home offense does vs away defense
        # Subtract def_epa_allowed because higher = worse defense = more points for offense
        home_power = (home_off - away_def_allowed) * EPA_TO_POINTS_FACTOR
        away_power = (away_off - home_def_allowed) * EPA_TO_POINTS_FACTOR

        # Rest days adjustment
        home_rest_adj = REST_ADJUSTMENTS.get(home_rest, 0.0)
        away_rest_adj = REST_ADJUSTMENTS.get(away_rest, 0.0)

        if home_rest not in REST_ADJUSTMENTS:
            closest_key = min(REST_ADJUSTMENTS.keys(), key=lambda x: abs(x - home_rest))
            home_rest_adj = REST_ADJUSTMENTS[closest_key]
        if away_rest not in REST_ADJUSTMENTS:
            closest_key = min(REST_ADJUSTMENTS.keys(), key=lambda x: abs(x - away_rest))
            away_rest_adj = REST_ADJUSTMENTS[closest_key]

        rest_adjustment = home_rest_adj - away_rest_adj

        # Model spread: negative = home favored
        model_spread = -(home_power - away_power + self.home_field_advantage + rest_adjustment)

        # For divisional games, shrink the spread toward 0
        if is_divisional:
            model_spread = model_spread * DIVISIONAL_SPREAD_SHRINK

        # Edge in points
        edge_points = market_spread - model_spread

        if abs(edge_points) < self.min_spread_edge:
            return None, 0.0, 0.0

        direction = 'HOME' if edge_points > 0 else 'AWAY'
        confidence = 0.5 + min(0.15, abs(edge_points) * 0.03)

        return direction, abs(edge_points), confidence

    def calculate_total_edge(
        self,
        home_epa: Dict[str, float],
        away_epa: Dict[str, float],
        market_total: float,
        is_dome: bool = False,
        temperature: Optional[float] = None,
        wind_speed: Optional[float] = None,
    ) -> Tuple[Optional[str], float, float, Dict]:
        """
        Calculate total edge using Pace × Efficiency model.

        V29 FORMULA (unit-consistent):

        1. plays_total = f(home_pace, away_pace) with game script adjustment
        2. points_per_play = baseline + epa_adjustment + weather_adjustment
        3. model_total = plays_total × points_per_play

        SIGN CONVENTIONS:
        - Higher combined_off_epa → MORE scoring (good offenses)
        - Higher combined_def_epa_allowed → MORE scoring (bad defenses allow points)

        Args:
            home_epa: Home team EPA stats
            away_epa: Away team EPA stats
            market_total: Market total (e.g., 45.5)
            is_dome: Whether game is in a dome/closed roof
            temperature: Game temperature in Fahrenheit
            wind_speed: Wind speed in mph

        Returns:
            Tuple of (direction, edge_pct, confidence, debug_info)
        """
        # =================================================================
        # STEP 1: Calculate expected plays
        # =================================================================
        home_pace = home_epa.get('pace', LEAGUE_AVG_PACE)
        away_pace = away_epa.get('pace', LEAGUE_AVG_PACE)

        # Combined plays per game (both teams)
        plays_total = home_pace + away_pace

        # =================================================================
        # STEP 2: Calculate points per play (baseline + adjustments)
        # =================================================================

        # Baseline points per play (calibrated from historical data)
        ppp_baseline = self.points_per_play  # ~0.38

        # EPA-based efficiency adjustment
        # Get EPA values (support both naming conventions)
        home_off = home_epa['off_epa']
        away_off = away_epa['off_epa']
        home_def_allowed = home_epa.get('def_epa_allowed', home_epa.get('def_epa', 0))
        away_def_allowed = away_epa.get('def_epa_allowed', away_epa.get('def_epa', 0))

        # Combined offensive quality (both teams' ability to generate points)
        # Higher = more scoring
        combined_off_epa = home_off + away_off

        # Combined defensive weakness (both teams allow points)
        # Higher = worse defenses = more scoring
        combined_def_epa_allowed = home_def_allowed + away_def_allowed

        # Total efficiency adjustment
        # Both terms should be POSITIVE when we expect MORE scoring
        epa_efficiency = combined_off_epa + combined_def_epa_allowed

        # Scale EPA to points per play adjustment
        # EPA is per-play, so this converts to PPP adjustment
        ppp_epa_adj = epa_efficiency * (self.epa_total_factor / 100)  # Scale appropriately

        # Weather/environment adjustment to PPP
        ppp_weather_adj = 0.0

        if is_dome:
            ppp_weather_adj += WEATHER_ADJUSTMENTS['dome'] / plays_total
        elif temperature is not None:
            if temperature < 32:
                ppp_weather_adj += WEATHER_ADJUSTMENTS['cold'] / plays_total
            if wind_speed is not None and wind_speed > 15:
                ppp_weather_adj += WEATHER_ADJUSTMENTS['wind'] * (wind_speed - 15) / plays_total

        # Total points per play
        points_per_play = ppp_baseline + ppp_epa_adj + ppp_weather_adj

        # Sanity bounds on PPP (NFL range: ~0.30 to 0.50)
        points_per_play = max(0.30, min(0.50, points_per_play))

        # =================================================================
        # STEP 3: Calculate model total
        # =================================================================

        # Raw model total
        raw_model_total = plays_total * points_per_play

        # Sanity bounds: NFL games rarely go below 30 or above 70
        model_total = max(30.0, min(70.0, raw_model_total))
        was_clipped = (model_total != raw_model_total)

        # =================================================================
        # STEP 4: Calculate edge
        # =================================================================

        edge_pts = model_total - market_total

        # Only bet when model disagrees by threshold
        if abs(edge_pts) < self.min_total_edge_pts:
            debug_info = self._build_totals_debug_info(
                plays_total, ppp_baseline, ppp_epa_adj, ppp_weather_adj,
                combined_off_epa, combined_def_epa_allowed,
                raw_model_total, model_total, market_total, was_clipped,
                is_dome, temperature, wind_speed
            )
            return None, 0.0, 0.0, debug_info

        # Direction: model higher = OVER, model lower = UNDER
        direction = 'OVER' if edge_pts > 0 else 'UNDER'

        # Confidence: scale edge to probability
        # 3 pts = 52%, 5 pts = 55%, 8 pts = 58%
        confidence = 0.50 + min(0.10, abs(edge_pts) * 0.01)

        # Edge as percentage for consistency
        edge_pct = abs(edge_pts) / market_total * 100

        # Build debug info for logging
        debug_info = self._build_totals_debug_info(
            plays_total, ppp_baseline, ppp_epa_adj, ppp_weather_adj,
            combined_off_epa, combined_def_epa_allowed,
            raw_model_total, model_total, market_total, was_clipped,
            is_dome, temperature, wind_speed
        )

        if self.verbose:
            logger.info(f"Total edge: {direction} {edge_pts:+.1f} pts | {debug_info}")

        return direction, edge_pct, confidence, debug_info

    def _build_totals_debug_info(
        self,
        plays_total: float,
        ppp_baseline: float,
        ppp_epa_adj: float,
        ppp_weather_adj: float,
        combined_off_epa: float,
        combined_def_epa_allowed: float,
        raw_model_total: float,
        model_total: float,
        market_total: float,
        was_clipped: bool,
        is_dome: bool,
        temperature: Optional[float],
        wind_speed: Optional[float],
    ) -> Dict:
        """Build debug info dict for observability."""
        return {
            'plays_total': round(plays_total, 1),
            'ppp_baseline': round(ppp_baseline, 4),
            'ppp_epa_adj': round(ppp_epa_adj, 4),
            'ppp_weather_adj': round(ppp_weather_adj, 4),
            'ppp_total': round(ppp_baseline + ppp_epa_adj + ppp_weather_adj, 4),
            'combined_off_epa': round(combined_off_epa, 4),
            'combined_def_epa_allowed': round(combined_def_epa_allowed, 4),
            'raw_model_total': round(raw_model_total, 1),
            'clipped_model_total': round(model_total, 1),
            'market_total': round(market_total, 1),
            'edge_pts': round(model_total - market_total, 1),
            'was_clipped': was_clipped,
            'is_dome': is_dome,
            'temperature': temperature,
            'wind_speed': wind_speed,
        }

    def generate_recommendations(
        self,
        schedule_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        pbp_df: pd.DataFrame,
        week: int,
        season: int = 2025
    ) -> List[Dict]:
        """
        Generate game line recommendations for a week.

        Args:
            schedule_df: NFLverse schedule with home_team, away_team
            odds_df: Odds DataFrame with market spread/total
            pbp_df: Play-by-play data for EPA calculation
            week: Current week number
            season: Season year

        Returns:
            List of recommendation dicts
        """
        recommendations = []

        # Get games for this week
        week_games = schedule_df[
            (schedule_df['week'] == week) &
            (schedule_df['season'] == season)
        ]

        if len(week_games) == 0:
            print(f"  No games found for week {week}")
            return recommendations

        # Calculate team EPA for all teams
        teams = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())
        for team in teams:
            self.team_stats[team] = self.calculate_team_epa(pbp_df, team, week)

        # Get current time for kickoff filtering
        import pytz
        now = datetime.now(pytz.timezone('US/Eastern'))

        # Process each game
        for _, game in week_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            game_id = game.get('game_id', f"{away_team}@{home_team}")

            # Skip games that have already kicked off
            gameday = game.get('gameday')
            gametime = game.get('gametime')
            if gameday is not None and gametime is not None:
                try:
                    game_dt_str = f"{gameday} {gametime}"
                    game_dt = datetime.strptime(game_dt_str, "%Y-%m-%d %H:%M")
                    game_dt = pytz.timezone('US/Eastern').localize(game_dt)

                    if now >= game_dt:
                        continue
                except (ValueError, TypeError):
                    pass

            home_epa = self.team_stats.get(home_team, {
                'off_epa': 0, 'def_epa_allowed': 0, 'def_epa': 0, 'pace': LEAGUE_AVG_PACE, 'games': 0
            })
            away_epa = self.team_stats.get(away_team, {
                'off_epa': 0, 'def_epa_allowed': 0, 'def_epa': 0, 'pace': LEAGUE_AVG_PACE, 'games': 0
            })

            # Skip if insufficient data
            if home_epa['games'] < 3 or away_epa['games'] < 3:
                continue

            # Extract schedule context
            home_rest = int(game.get('home_rest', 7)) if pd.notna(game.get('home_rest')) else 7
            away_rest = int(game.get('away_rest', 7)) if pd.notna(game.get('away_rest')) else 7
            is_divisional = bool(game.get('div_game', False))

            # Weather context for totals
            roof = str(game.get('roof', '')).lower()
            is_dome = roof in ['dome', 'closed', 'retractable roof closed']
            temp_val = game.get('temp')
            wind_val = game.get('wind')
            temperature = float(temp_val) if pd.notna(temp_val) else (72.0 if is_dome else None)
            wind_speed = float(wind_val) if pd.notna(wind_val) else None

            # Find odds for this game
            game_odds = self._find_game_odds(odds_df, home_team, away_team)

            # --- SPREAD EDGE ---
            if game_odds.get('spread') is not None:
                market_spread = game_odds['spread']
                direction, edge_pts, confidence = self.calculate_spread_edge(
                    home_epa, away_epa, market_spread,
                    home_rest=home_rest, away_rest=away_rest,
                    is_divisional=is_divisional
                )

                if direction is not None:
                    pick = f"{home_team} {market_spread:+.1f}" if direction == 'HOME' else f"{away_team} +{-market_spread:.1f}"

                    reasoning_parts = [f"EPA edge: {edge_pts:.1f} pts toward {direction}"]
                    if home_rest != away_rest:
                        rest_diff = home_rest - away_rest
                        rest_team = "home" if rest_diff > 0 else "away"
                        reasoning_parts.append(f"Rest: {rest_team} +{abs(rest_diff)}d")
                    if is_divisional:
                        reasoning_parts.append("Divisional (spread shrunk)")

                    recommendations.append({
                        'game': f"{away_team} @ {home_team}",
                        'game_id': game_id,
                        'bet_type': 'spread',
                        'pick': pick,
                        'direction': direction,
                        'market_line': market_spread,
                        'edge_pct': edge_pts,
                        'combined_confidence': confidence,
                        'units': round(confidence * 1.5, 2),
                        'source': 'GAME_LINE_SPREAD',
                        'reasoning': " | ".join(reasoning_parts),
                        'home_off_epa': home_epa['off_epa'],
                        'away_off_epa': away_epa['off_epa'],
                        'home_def_epa': home_epa.get('def_epa_allowed', home_epa.get('def_epa')),
                        'away_def_epa': away_epa.get('def_epa_allowed', away_epa.get('def_epa')),
                        'home_rest': home_rest,
                        'away_rest': away_rest,
                        'is_divisional': is_divisional,
                    })

            # --- TOTAL EDGE ---
            if game_odds.get('total') is not None:
                market_total = game_odds['total']
                result = self.calculate_total_edge(
                    home_epa, away_epa, market_total,
                    is_dome=is_dome, temperature=temperature, wind_speed=wind_speed
                )
                direction, edge_pct, confidence, debug_info = result

                if direction is not None:
                    pick = f"{direction} {market_total}"

                    # Build reasoning with debug info
                    reasoning = f"Model: {debug_info['clipped_model_total']:.1f} vs Market: {market_total:.1f}"
                    if is_dome:
                        reasoning += " | Dome (+1.5)"
                    elif temperature and temperature < 32:
                        reasoning += f" | Cold {temperature:.0f}F (-2.0)"

                    recommendations.append({
                        'game': f"{away_team} @ {home_team}",
                        'game_id': game_id,
                        'bet_type': 'total',
                        'pick': pick,
                        'direction': direction,
                        'market_line': market_total,
                        'edge_pct': edge_pct,
                        'combined_confidence': confidence,
                        'units': round(confidence * 1.5, 2),
                        'source': 'GAME_LINE_TOTAL',
                        'reasoning': reasoning,
                        'home_off_epa': home_epa['off_epa'],
                        'away_off_epa': away_epa['off_epa'],
                        'home_pace': home_epa['pace'],
                        'away_pace': away_epa['pace'],
                        'model_total': debug_info['clipped_model_total'],
                        'is_dome': is_dome,
                        'temperature': temperature,
                        'wind_speed': wind_speed,
                        'debug_info': debug_info,
                    })

        return recommendations

    def _find_game_odds(
        self,
        odds_df: pd.DataFrame,
        home_team: str,
        away_team: str
    ) -> Dict[str, Optional[float]]:
        """Find spread and total odds for a game."""
        result = {'spread': None, 'total': None}

        if len(odds_df) == 0:
            return result

        # Format 1: Long format with game_id and side columns
        if 'game_id' in odds_df.columns and 'side' in odds_df.columns:
            game_mask = odds_df['game_id'].str.contains(f"_{away_team}_{home_team}", na=False)

            if game_mask.any():
                game_odds = odds_df[game_mask]

                home_spread_row = game_odds[game_odds['side'] == 'home_spread']
                if len(home_spread_row) > 0:
                    result['spread'] = home_spread_row['point'].iloc[0]

                over_row = game_odds[game_odds['side'] == 'over']
                if len(over_row) > 0:
                    result['total'] = over_row['point'].iloc[0]

            return result

        # Format 2: Standard format with team columns
        game_mask = (
            (odds_df.get('home_team', pd.Series()) == home_team) &
            (odds_df.get('away_team', pd.Series()) == away_team)
        ) | (
            (odds_df.get('home', pd.Series()) == home_team) &
            (odds_df.get('away', pd.Series()) == away_team)
        )

        if game_mask.any():
            game_row = odds_df[game_mask].iloc[0]
            result['spread'] = game_row.get('spread', game_row.get('point_spread_home'))
            result['total'] = game_row.get('total', game_row.get('over_under'))

        return result

    def save(self, path: Path = None):
        """Save edge model."""
        if path is None:
            path = Path(__file__).parent.parent.parent / 'data' / 'models' / 'game_line_edge.joblib'

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'version': self.version,
            'home_field_advantage': self.home_field_advantage,
            'min_spread_edge': self.min_spread_edge,
            'min_total_edge_pts': self.min_total_edge_pts,
            'points_per_play': self.points_per_play,
            'epa_total_factor': self.epa_total_factor,
            'saved_date': datetime.now().isoformat(),
        }

        joblib.dump(model_data, path)
        print(f"Saved game line edge to {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'GameLineEdge':
        """Load edge model."""
        if path is None:
            path = Path(__file__).parent.parent.parent / 'data' / 'models' / 'game_line_edge.joblib'

        if not path.exists():
            print(f"No saved model found at {path}, using defaults")
            return cls()

        model_data = joblib.load(path)
        edge = cls()
        edge.version = model_data.get('version', '1.0')
        edge.home_field_advantage = model_data.get('home_field_advantage', HOME_FIELD_ADVANTAGE)
        edge.min_spread_edge = model_data.get('min_spread_edge', MIN_SPREAD_EDGE)
        edge.min_total_edge_pts = model_data.get('min_total_edge_pts', MIN_TOTAL_EDGE_PTS)
        edge.points_per_play = model_data.get('points_per_play', _TOTALS_CALIBRATED['points_per_play'])
        edge.epa_total_factor = model_data.get('epa_total_factor', _TOTALS_CALIBRATED['epa_total_factor'])

        return edge


# Convenience function for integration
def generate_game_line_edge_recommendations(
    week: int,
    season: int = 2025,
    pbp_path: Path = None,
    schedule_path: Path = None,
    odds_path: Path = None,
) -> pd.DataFrame:
    """
    Generate game line recommendations using the edge approach.

    Args:
        week: Week number
        season: Season year
        pbp_path: Path to PBP parquet
        schedule_path: Path to schedule parquet
        odds_path: Path to odds CSV

    Returns:
        DataFrame with game line recommendations
    """
    project_root = Path(__file__).parent.parent.parent

    # Default paths
    if pbp_path is None:
        pbp_path = project_root / 'data' / 'nflverse' / 'pbp.parquet'
        if not pbp_path.exists():
            pbp_path = project_root / 'data' / 'nflverse' / f'pbp_{season}.parquet'

    if schedule_path is None:
        schedule_path = project_root / 'data' / 'nflverse' / 'schedules.parquet'

    if odds_path is None:
        odds_path = project_root / 'data' / f'odds_week{week}.csv'
        if not odds_path.exists():
            odds_path = project_root / 'data' / 'odds' / f'odds_week{week}.csv'

    # Load data
    print(f"  Loading PBP from {pbp_path}")
    pbp_df = pd.read_parquet(pbp_path)

    print(f"  Loading schedule from {schedule_path}")
    schedule_df = pd.read_parquet(schedule_path)

    if odds_path.exists():
        print(f"  Loading odds from {odds_path}")
        odds_df = pd.read_csv(odds_path)
    else:
        print(f"  Warning: No odds file found at {odds_path}")
        odds_df = pd.DataFrame()

    # Generate recommendations
    edge = GameLineEdge()
    recommendations = edge.generate_recommendations(
        schedule_df, odds_df, pbp_df, week, season
    )

    if not recommendations:
        return pd.DataFrame()

    return pd.DataFrame(recommendations)
