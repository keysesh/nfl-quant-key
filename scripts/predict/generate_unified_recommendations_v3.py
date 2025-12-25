#!/usr/bin/env python3
"""
NFL QUANT - Unified Recommendation Generator V3 (COMPLETE - CALIBRATED)

This script:
1. Loads model_predictions_weekX.csv (all 54+ features from Monte Carlo simulation)
2. Matches to current prop lines from DraftKings
3. Applies isotonic calibration to raw probabilities
4. Uses no-vig market probabilities for edge calculation
5. PRESERVES ALL MODEL FEATURES in output (no feature dropout)
6. Uses actual model_std from simulation (no hardcoded CV values)

KEY FIXES APPLIED:
- ✅ Calibration applied via load_calibrator_for_market()
- ✅ No-vig probabilities via remove_vig_two_way()
- ✅ All 54+ features from model_predictions preserved in output
- ✅ No hardcoded CV values - uses model_std exclusively
- ✅ ValueError raised if model_std missing (enforces upstream data quality)
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm, poisson, nbinom
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT, MODELS_DIR, DATA_DIR, REPORTS_DIR
project_root = PROJECT_ROOT

# Import calibration and betting utility modules
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market
from nfl_quant.calibration.bias_correction import apply_bias_correction, get_correction_factor
from nfl_quant.utils.unified_integration import apply_p1_adjustments, get_spread_adjustment
from nfl_quant.utils.odds import load_game_status_map
from nfl_quant.core.unified_betting import (
    remove_vig_two_way,
    assign_confidence_tier,
    calculate_kelly_fraction,
    calculate_recommended_units
)
from configs.model_config import get_active_model_path, FEATURES, FEATURE_COUNT, get_market_tier, get_confidence_threshold, MODEL_VERSION_FULL, CLASSIFIER_MARKETS, MARKET_DIRECTION_CONSTRAINTS
from nfl_quant.models.classifier_registry import get_active_model as get_registry_model, get_active_model_info
from nfl_quant.betting.snr_filter import SNRFilter, filter_bets_by_snr
from nfl_quant.config_enhanced import config
from nfl_quant.features.feature_defaults import FEATURE_DEFAULTS
from nfl_quant.integration import get_unified_prediction
from nfl_quant.simulation.model_guided_simulator import simulate_with_model_guidance
from nfl_quant.filters import (
    CONSERVATIVE_FILTER, ELITE_FILTER, should_take_bet, calculate_player_volatility
)
from nfl_quant.features.ir_return_detector import (
    detect_returning_players_snap_status,
    apply_snap_ramp_adjustments,
    load_manual_overrides
)
from nfl_quant.features.team_synergy import (
    calculate_team_synergy_adjustment,
    apply_synergy_adjustments,
    generate_synergy_report,
    load_player_statuses_from_injuries,
    detect_returning_players_from_stats,
    get_synergy_betting_implications,
    PlayerStatus,
)

# Feature flag for team synergy adjustments (V25)
USE_TEAM_SYNERGY = True  # Set to False to disable synergy calculations

# Feature flag for model-guided simulation (V4 integrated approach)
USE_MODEL_GUIDED_SIMULATION = True  # Set to False to use legacy MC simulation

# Market direction constraints imported from configs.model_config
# See MARKET_DIRECTION_CONSTRAINTS in model_config.py for full market-specific constraints

# Load betting thresholds from config (single source of truth)
CONFIDENCE_FLOOR = config.betting.performance_adjustment.get('confidence_floor', 0.50)
CONFIDENCE_CEILING = config.betting.performance_adjustment.get('confidence_ceiling', 0.90)
LOW_CONFIDENCE_THRESHOLD = config.betting.confidence_levels.get('low', 0.55)
MEDIUM_CONFIDENCE_THRESHOLD = config.betting.confidence_levels.get('medium', 0.70)

# V6 Value Classifier import (replaces V5 Edge Classifier)
import joblib
V6_MODEL_PATH = project_root / 'data' / 'models' / 'v6_value_classifier.joblib'
V5_MODEL_PATH = project_root / 'data' / 'models' / 'v5_edge_classifier.joblib'  # Backwards compat
_v6_model = None
_v6_metrics = None
_v6_recommended = None

# ACTIVE MODEL - Use canonical path from configs.model_config
ACTIVE_MODEL_PATH = get_active_model_path()
_active_model = None
_active_thresholds = None
_active_validation = None
_active_version = None  # Stored for logging only, not used in logic

# Import opponent stats module
try:
    from nfl_quant.features.opponent_stats import (
        get_opponent_conflict_flag,
        calculate_opponent_divergence,
        create_opponent_interaction_features
    )
    OPPONENT_STATS_AVAILABLE = True
except ImportError:
    OPPONENT_STATS_AVAILABLE = False

# Import FeatureEngine for comprehensive feature extraction
from nfl_quant.features import get_feature_engine
from nfl_quant.utils.player_names import normalize_player_name

# Import centralized model config for smooth sweet spot
try:
    from configs.model_config import (
        FEATURE_FLAGS,
        smooth_sweet_spot,
        FEATURES,
    )
    MODEL_CONFIG_AVAILABLE = True
except ImportError:
    MODEL_CONFIG_AVAILABLE = False
    # Fallback defaults
    class FeatureFlagsDefault:
        use_smooth_sweet_spot = False
        use_lvt_x_defense = False
        use_lvt_x_rest = False
    FEATURE_FLAGS = FeatureFlagsDefault()
_feature_engine = None
_game_context_cache = None  # Cache for game context (spread, total, home/away)
_historical_odds_cache = None  # Cache for historical odds actuals
_nflverse_weekly_cache = None  # Cache for NFLverse weekly stats


# Market to NFLverse stat column mapping
MARKET_TO_STAT_COLUMN = {
    'player_reception_yds': 'receiving_yards',
    'player_receptions': 'receptions',
    'player_rush_yds': 'rushing_yards',
    'player_rush_attempts': 'carries',
    'player_pass_yds': 'passing_yards',
    'player_pass_completions': 'completions',
    'player_pass_attempts': 'attempts',
    'player_pass_tds': 'passing_tds',
    'player_rush_tds': 'rushing_tds',
    'player_reception_tds': 'receiving_tds',
    'player_targets': 'targets',
    'player_interceptions': 'interceptions',
    # TD Props - special handling (sum of columns)
    'player_anytime_td': ['rushing_tds', 'receiving_tds'],  # Sum of both
    'player_1st_td': ['rushing_tds', 'receiving_tds'],
}


def load_nflverse_weekly_stats():
    """Load NFLverse weekly stats for calculating actual trailing averages."""
    global _nflverse_weekly_cache
    if _nflverse_weekly_cache is not None:
        return _nflverse_weekly_cache

    try:
        weekly_path = project_root / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not weekly_path.exists():
            logger.warning(f"NFLverse weekly stats not found: {weekly_path}")
            return None

        weekly = pd.read_parquet(weekly_path)
        # Normalize player names for matching
        weekly['player_norm'] = weekly['player_display_name'].apply(normalize_player_name)
        _nflverse_weekly_cache = weekly
        logger.info(f"Loaded {len(weekly)} NFLverse weekly stat records")
        return weekly
    except Exception as e:
        logger.warning(f"Could not load NFLverse weekly stats: {e}")
        return None


def load_historical_odds():
    """Load historical odds with actuals for calculating historical rates."""
    global _historical_odds_cache
    if _historical_odds_cache is not None:
        return _historical_odds_cache

    try:
        hist_path = project_root / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
        if not hist_path.exists():
            logger.warning(f"Historical odds file not found: {hist_path}")
            return None

        hist = pd.read_csv(hist_path)
        hist['player_norm'] = hist['player'].apply(normalize_player_name)
        _historical_odds_cache = hist
        logger.info(f"Loaded {len(hist)} historical props with actuals")
        return hist
    except Exception as e:
        logger.warning(f"Could not load historical odds: {e}")
        return None


def get_player_historical_stats(player_name: str, market: str, current_season: int = 2025, current_week: int = 14) -> dict:
    """
    Get historical stats for a player in a given market.

    Uses ONLY prior data (no leakage) - excludes current season/week.

    FIX (Dec 7, 2025): Now uses NFLverse weekly_stats for hist_avg instead of
    historical props file. The props file had multiple entries per game and
    diluted recent performance with old data, causing ~20% error in trailing avg.

    Returns:
        dict with:
            - hist_over_rate: how often player goes OVER (0-1) - from props history
            - hist_avg: EWMA trailing average from actual game stats (4-game span)
            - hist_count: number of games used for trailing avg
    """
    player_norm = normalize_player_name(player_name)

    # Get hist_over_rate from props history (still useful for hit rate)
    hist = load_historical_odds()
    hist_over_rate = 0.5
    if hist is not None:
        player_hist = hist[
            (hist['player_norm'] == player_norm) &
            (hist['market'] == market)
        ].copy()
        # Exclude current week (no leakage)
        player_hist = player_hist[
            ~((player_hist['season'] == current_season) & (player_hist['week'] >= current_week))
        ]
        if len(player_hist) > 0 and 'over_hit' in player_hist.columns:
            hist_over_rate = player_hist['over_hit'].mean()

    # Get hist_avg from NFLverse weekly stats (ACTUAL game performance)
    weekly = load_nflverse_weekly_stats()
    if weekly is None:
        return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}

    # Map market to stat column(s)
    stat_col = MARKET_TO_STAT_COLUMN.get(market)
    if stat_col is None:
        logger.warning(f"Unknown market {market} - no stat column mapping")
        return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}

    # Handle list of columns (e.g., TD props sum rushing_tds + receiving_tds)
    if isinstance(stat_col, list):
        # Verify all columns exist
        for col in stat_col:
            if col not in weekly.columns:
                logger.warning(f"Stat column {col} not in weekly stats")
                return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}
    else:
        if stat_col not in weekly.columns:
            logger.warning(f"Stat column {stat_col} not in weekly stats")
            return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}

    # Filter to this player, current season, weeks before current
    player_games = weekly[
        (weekly['player_norm'] == player_norm) &
        (weekly['season'] == current_season) &
        (weekly['week'] < current_week) &
        (weekly['week'] > 0)
    ].copy()

    if len(player_games) == 0:
        # Try previous season if no current season data
        player_games = weekly[
            (weekly['player_norm'] == player_norm) &
            (weekly['season'] == current_season - 1)
        ].copy()

    if len(player_games) == 0:
        return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}

    # Sort by week and calculate EWMA trailing average (span=4)
    player_games = player_games.sort_values('week')

    # Handle list of columns (sum them)
    if isinstance(stat_col, list):
        stat_values = player_games[stat_col].fillna(0).sum(axis=1)
    else:
        stat_values = player_games[stat_col].dropna()

    if len(stat_values) == 0:
        return {'hist_over_rate': hist_over_rate, 'hist_avg': None, 'hist_count': 0}

    # Use EWMA with span=4 for trailing average (matches model training)
    # This gives more weight to recent games: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
    hist_avg = stat_values.ewm(span=4, min_periods=1).mean().iloc[-1]

    return {
        'hist_over_rate': hist_over_rate,
        'hist_avg': float(hist_avg),
        'hist_count': len(stat_values)
    }


# ============================================================================
# DISPLAY FORMATTING HELPERS
# ============================================================================

def format_market_name(market: str) -> str:
    """
    Convert internal market name to human-readable display format.

    Examples:
        player_pass_completions -> Pass Completions
        player_reception_yds -> Receiving Yards
        player_rush_yds -> Rushing Yards
    """
    market_display_names = {
        'player_receptions': 'Receptions',
        'player_reception_yds': 'Receiving Yards',
        'player_rush_yds': 'Rushing Yards',
        'player_rush_attempts': 'Rush Attempts',
        'player_pass_yds': 'Passing Yards',
        'player_pass_completions': 'Pass Completions',
        'player_pass_attempts': 'Pass Attempts',
        'player_pass_tds': 'Passing TDs',
        'player_rush_tds': 'Rushing TDs',
        'player_reception_tds': 'Receiving TDs',
        'player_targets': 'Targets',
        'player_interceptions': 'Interceptions',
        # TD scorer markets
        'player_anytime_td': 'Anytime TD',
        'player_1st_td': 'First TD Scorer',
        'player_last_td': 'Last TD Scorer',
    }
    return market_display_names.get(market, market.replace('player_', '').replace('_', ' ').title())


# ============================================================================
# COMPREHENSIVE FEATURE EXTRACTION (Matches V16 Training)
# ============================================================================

def get_feature_engine_instance():
    """Get or create singleton FeatureEngine instance."""
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = get_feature_engine()
    return _feature_engine


def load_game_context(season: int, week: int):
    """Load game context (spread, total, home/away) from schedules."""
    global _game_context_cache

    if _game_context_cache is not None:
        return _game_context_cache

    try:
        sched_path = project_root / 'data' / 'nflverse' / 'schedules.parquet'
        sched = pd.read_parquet(sched_path)

        # Filter to current week
        week_games = sched[(sched['season'] == season) & (sched['week'] == week)]

        # Build lookup dict
        context = {}
        for _, game in week_games.iterrows():
            home = game['home_team']
            away = game['away_team']
            spread = game.get('spread_line', 0) or 0
            total = game.get('total_line', 45) or 45
            is_div = game.get('div_game', 0) or 0

            # Store for both teams
            context[home] = {
                'is_home': 1,
                'spread': spread,  # Home perspective
                'total': total,
                'is_divisional': is_div,
                'opponent': away
            }
            context[away] = {
                'is_home': 0,
                'spread': -spread,  # Away perspective
                'total': total,
                'is_divisional': is_div,
                'opponent': home
            }

        _game_context_cache = context
        return context
    except Exception as e:
        return {}


def extract_prediction_features(row: dict, market: str, season: int, week: int) -> dict:
    """
    Extract ALL 39 features required by the V23 model using FeatureEngine.

    CRITICAL FIX (Dec 7, 2025): Feature names MUST match model_config.py EXACTLY.
    Previous version had wrong feature names causing 34/39 features to default to 0.

    Args:
        row: Dict with player data (from prediction dataframe)
        market: Market type (player_receptions, player_rush_yds, etc.)
        season: NFL season
        week: NFL week

    Returns:
        Dict with all 39 model features matching configs/model_config.py
    """
    from configs.model_config import smooth_sweet_spot

    engine = get_feature_engine_instance()
    features = {}

    # Extract key identifiers
    player_id = row.get('player_id', '')
    player_name = row.get('player', '')
    team = row.get('team', '')
    opponent = row.get('opponent', '')
    position = row.get('position', '')

    # Get game context
    game_context = load_game_context(season, week)
    team_context = game_context.get(team, {})

    # Get line and trailing stat
    line = row.get('line', 0) or 0
    trailing = row.get('trailing_stat', row.get('hist_avg', row.get('model_projection', line))) or line
    lvt = (line - trailing) / (trailing + 0.1) if trailing else 0

    # ==========================================================================
    # V12 CORE FEATURES (12 features) - EXACT NAMES from model_config.py
    # ==========================================================================

    # 1. line_vs_trailing - Primary feature
    features['line_vs_trailing'] = lvt

    # 2. line_level - Raw line value
    features['line_level'] = line

    # 3. line_in_sweet_spot - Gaussian decay sweet spot
    features['line_in_sweet_spot'] = smooth_sweet_spot(line, market=market)

    # 4. player_under_rate - Historical under hit rate for this player
    # Use data from historical_odds if available, otherwise default
    features['player_under_rate'] = row.get('player_under_rate', 0.5)

    # 5. player_bias - How much player over/under-performs vs line
    features['player_bias'] = row.get('player_bias', 0.0)

    # 6. market_under_rate - Market-wide under hit rate
    features['market_under_rate'] = row.get('market_under_rate', 0.5)

    # 7-12. Interaction terms
    player_tendency = features['player_under_rate'] - 0.5
    features['LVT_x_player_tendency'] = lvt * player_tendency
    features['LVT_x_player_bias'] = lvt * features['player_bias']
    features['LVT_x_regime'] = lvt * row.get('regime_indicator', 0)
    features['LVT_in_sweet_spot'] = lvt * features['line_in_sweet_spot']
    features['market_bias_strength'] = abs(features['market_under_rate'] - 0.5) * 2
    features['player_market_aligned'] = 1.0 if (features['player_under_rate'] > 0.5) == (features['market_under_rate'] > 0.5) else 0.0

    # ==========================================================================
    # V17 FEATURES (8 features) - Interactions + Skill
    # ==========================================================================

    # Defense interaction with LVT
    try:
        if 'rush' in market:
            def_epa = engine.get_rush_defense_epa(opponent, season, week)
        else:
            def_epa = engine.get_pass_defense_epa(opponent, season, week)
    except Exception:
        def_epa = row.get('opponent_def_epa_vs_position', 0)
    features['lvt_x_defense'] = lvt * def_epa

    # Rest advantage interaction
    rest_days = row.get('rest_days', 7)
    rest_advantage = (rest_days - 7) / 7  # Normalized: 0 = normal, positive = extra rest
    features['lvt_x_rest'] = lvt * rest_advantage

    # NGS receiving metrics
    try:
        features['avg_separation'] = engine.get_avg_separation(player_id, season, week)
    except Exception:
        features['avg_separation'] = row.get('team_avg_separation', 2.8)

    try:
        features['avg_cushion'] = engine.get_avg_cushion(player_id, season, week)
    except Exception:
        features['avg_cushion'] = 6.0

    # Trailing catch rate
    try:
        features['trailing_catch_rate'] = engine.get_trailing_catch_rate(player_id, season, week)
    except Exception:
        features['trailing_catch_rate'] = 0.65  # League average

    # Usage metrics
    try:
        features['snap_share'] = engine.get_snap_share(player_name, season, week)
    except Exception:
        features['snap_share'] = row.get('snap_share', 0.5)

    try:
        features['target_share'] = engine.get_target_share(player_id, season, week)
    except Exception:
        features['target_share'] = row.get('target_share', 0.15)

    # WR1-specific defense (for WR receptions only)
    if market == 'player_receptions' and position == 'WR' and opponent:
        try:
            features['opp_wr1_receptions_allowed'] = engine.get_wr1_defense_stat(opponent, season, week)
        except Exception:
            features['opp_wr1_receptions_allowed'] = 5.5  # League average WR1 receptions
    else:
        features['opp_wr1_receptions_allowed'] = 5.5

    # ==========================================================================
    # V18 FEATURES (7 features) - Game Context + Skill
    # ==========================================================================

    # Game pace
    try:
        features['game_pace'] = engine.get_game_pace(team, opponent, season, week)
    except Exception:
        features['game_pace'] = row.get('game_pace', 64.0)  # League average plays per game

    # Vegas context
    features['vegas_total'] = team_context.get('total', row.get('vegas_total', 44.0)) or 44.0
    features['vegas_spread'] = team_context.get('spread', row.get('vegas_spread', 0.0)) or 0.0

    # Implied team total
    vegas_total = features['vegas_total']
    vegas_spread = features['vegas_spread']
    features['implied_team_total'] = (vegas_total / 2) - (vegas_spread / 2)

    # aDOT for receiving markets
    if market in ['player_receptions', 'player_reception_yds'] and player_id:
        try:
            features['adot'] = engine.get_adot(player_id, season, week)
        except Exception:
            features['adot'] = row.get('adot', 8.5)
    else:
        features['adot'] = 8.5

    # Pressure rates - actual league avg is 15.4% (not 25%)
    ACTUAL_PRESSURE_RATE = 0.154
    try:
        features['pressure_rate'] = engine.get_pressure_rate(team, season, week)
    except Exception:
        features['pressure_rate'] = row.get('pressure_rate_generated', ACTUAL_PRESSURE_RATE)

    try:
        features['opp_pressure_rate'] = engine.get_opp_pressure_rate(opponent, season, week)
    except Exception:
        features['opp_pressure_rate'] = row.get('opp_pressure_rate_generated', ACTUAL_PRESSURE_RATE)

    # ==========================================================================
    # V19 FEATURES (4 features) - Rush/Receiving Improvements
    # ==========================================================================

    # O-line health score (1.0 = healthy, lower = injuries)
    features['oline_health_score'] = row.get('oline_health_score', 1.0)

    # Expected box count (defenders near line of scrimmage)
    # Higher spread (team favored) = likely more stacked boxes
    features['box_count_expected'] = 7.0 + (vegas_spread * 0.15)
    features['box_count_expected'] = max(6.0, min(9.0, features['box_count_expected']))

    # Slot snap percentage (proxy from aDOT - lower aDOT = more slot)
    adot = features['adot']
    features['slot_snap_pct'] = max(0.0, min(1.0, 1.0 - (adot / 15.0)))

    # Target share trailing (4-week rolling)
    features['target_share_trailing'] = features['target_share']  # Use same as target_share

    # ==========================================================================
    # V23 FEATURES (4 features) - Opponent Context
    # ==========================================================================

    # Opponent pass defense vs league average
    try:
        pass_def_epa = engine.get_pass_defense_epa(opponent, season, week)
        # Convert to z-score (positive = bad defense, negative = good defense)
        features['opp_pass_def_vs_avg'] = pass_def_epa / 0.05 if pass_def_epa else 0.0
    except Exception:
        features['opp_pass_def_vs_avg'] = 0.0

    # Opponent rush defense vs league average
    try:
        rush_def_epa = engine.get_rush_defense_epa(opponent, season, week)
        features['opp_rush_def_vs_avg'] = rush_def_epa / 0.05 if rush_def_epa else 0.0
    except Exception:
        features['opp_rush_def_vs_avg'] = 0.0

    # Opponent defense EPA (raw)
    features['opp_def_epa'] = row.get('opponent_def_epa_vs_position', def_epa if 'def_epa' in dir() else 0.0)

    # Flag for opponent context availability
    has_opp_data = features['opp_pass_def_vs_avg'] != 0 or features['opp_rush_def_vs_avg'] != 0
    features['has_opponent_context'] = 1 if has_opp_data else 0

    # ==========================================================================
    # V24 FEATURES (4 features) - Injury Context
    # ==========================================================================

    # Player injury status (0=healthy, 1=questionable, 2=doubtful)
    injury_status = row.get('injury_qb_status', row.get('severity', 0))
    if isinstance(injury_status, str):
        injury_map = {'healthy': 0, 'questionable': 1, 'doubtful': 2, 'out': 3}
        injury_status = injury_map.get(injury_status.lower(), 0)
    features['player_injury_status'] = injury_status or 0

    # QB injury status
    qb_status = row.get('injury_qb_status', 0)
    if isinstance(qb_status, str):
        qb_map = {'healthy': 0, 'questionable': 1, 'doubtful': 2, 'out': 3, 'backup': 3}
        qb_status = qb_map.get(qb_status.lower(), 0)
    features['qb_injury_status'] = qb_status or 0

    # Teammate injuries (opportunity boost flags)
    features['team_wr1_out'] = 1 if row.get('injury_wr1_status', '') in ['out', 'ir'] else 0
    features['team_rb1_out'] = 1 if row.get('injury_rb1_status', '') in ['out', 'ir'] else 0

    return features


def generate_model_reasoning(row: dict) -> str:
    """
    Generate SHAP-based explanation showing how each factor contributed to the prediction.

    Uses XGBoost feature contributions (log-odds space) with proper transformation:
        Base log-odds: 0.0 (50%)
        + line_vs_trailing: +0.60 (line high vs avg)
        + player_under_tendency: +0.32 (player tends to go under)
        - snap_share: -0.20 (high usage = more volume)
        = Raw log-odds: +0.72
        Sigmoid transform: 67%
        Final P(UNDER): 67%

    Args:
        row: Dictionary containing recommendation data

    Returns:
        String with SHAP-based formulaic breakdown
    """
    import numpy as np

    parts = []
    market = row.get('market', '')
    # Pick is now set to V23 model_best_direction for validated bets (ensemble disabled)
    pick_raw = row.get('pick') or row.get('model_best_direction') or 'UNDER'
    pick = str(pick_raw).upper()

    # Get all key values
    line = row.get('line', 0) or 0
    lvt = row.get('line_vs_trailing', 0) or 0
    # FIX: Use trailing_stat directly (actual historical average) instead of back-calculating from LVT
    # trailing_stat = actual 4-week EWMA from historical props
    # model_projection = MC simulation output (adjusted/calibrated mean)
    trailing = row.get('trailing_stat', row.get('hist_avg', row.get('model_projection', 0))) or 0
    projection = row.get('model_projection', 0) or 0
    p_under = row.get('model_p_under', 0.5) or 0.5
    edge = row.get('model_best_ev', row.get('edge_pct', 0)) or 0

    # Feature contributions (SHAP values in log-odds space)
    feature_contribs = row.get('feature_contributions', [])

    # Feature name mapping for clarity
    FEATURE_DISPLAY_NAMES = {
        'line_vs_trailing': 'line_vs_avg',
        'player_bias': 'projection_bias',
        'player_under_rate': 'player_under_tendency',
        'LVT_x_player_bias': 'LVT_x_proj_bias',
        'LVT_x_player_tendency': 'LVT_x_under_tendency',
        'target_share_trailing': 'target_share',
        'target_share': 'target_share',
        'avg_separation': 'receiver_separation',
        'avg_cushion': 'defender_cushion',
        'market_under_rate': 'market_tendency',
        'snap_share': 'snap_share',
        'opp_def_epa': 'defense_strength',
    }

    # ============================================
    # SHAP BREAKDOWN - Log-odds space
    # ============================================
    parts.append(f"CALCULATION (log-odds):")
    parts.append(f"  Base: 0.00 (50%)")

    if feature_contribs:
        # Calculate sum for consistency check
        shap_sum = sum(c[1] for c in feature_contribs)
        total_abs = sum(abs(c[1]) for c in feature_contribs)

        # Find dominant factor
        if feature_contribs:
            dominant_feat, dominant_val = max(feature_contribs, key=lambda x: abs(x[1]))
            dominant_pct = abs(dominant_val) / total_abs * 100 if total_abs > 0 else 0

        # Group factors by direction for clearer display
        over_factors = []
        under_factors = []

        for feat_name, contrib_val in feature_contribs[:5]:
            display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name).replace('_', ' ')
            is_dominant = feat_name == dominant_feat
            strength = abs(contrib_val)

            if contrib_val < 0:  # Negative SHAP = pushes toward OVER
                over_factors.append((display_name, strength, is_dominant))
            else:  # Positive SHAP = pushes toward UNDER
                under_factors.append((display_name, strength, is_dominant))

        # Show OVER factors (reasons to bet OVER)
        if over_factors:
            parts.append(f"  Favors OVER:")
            for name, strength, is_dom in sorted(over_factors, key=lambda x: -x[1]):
                marker = " *" if is_dom else ""
                parts.append(f"    • {name} ({strength:.2f}){marker}")

        # Show UNDER factors (reasons to bet UNDER)
        if under_factors:
            parts.append(f"  Favors UNDER:")
            for name, strength, is_dom in sorted(under_factors, key=lambda x: -x[1]):
                marker = " *" if is_dom else ""
                parts.append(f"    • {name} ({strength:.2f}){marker}")

        # Show raw sum and sigmoid transformation
        raw_prob_from_shap = 1 / (1 + np.exp(-shap_sum))  # Sigmoid
        parts.append(f"  --------")
        parts.append(f"  Sum: {shap_sum:+.2f} -> sigmoid -> {raw_prob_from_shap:.0%}")

        # Show calibration adjustment if different
        calibration_delta = p_under - raw_prob_from_shap
        if abs(calibration_delta) > 0.01:
            parts.append(f"  Calibration: {calibration_delta:+.0%}")

        parts.append(f"  = P(UNDER): {p_under:.0%}")

        # Add dominant factor note
        if dominant_pct > 30:
            dominant_display = FEATURE_DISPLAY_NAMES.get(dominant_feat, dominant_feat).replace('_', ' ')
            parts.append(f"  * Dominant: {dominant_display} ({dominant_pct:.0f}% influence)")
    else:
        # Fallback if no SHAP values available
        parts.append(f"  {'+' if lvt > 0 else ''}{lvt:.2f} LVT (line vs trailing)")
        parts.append(f"  = P(UNDER): {p_under:.0%}")

    # ============================================
    # CONTEXT SECTION
    # ============================================
    parts.append(f"")
    parts.append(f"INPUTS:")
    parts.append(f"  Line: {line:.1f} | Trailing: {trailing:.1f}")

    # Defense EPA if available
    def_epa = row.get('trailing_def_epa', 0) or 0
    if def_epa != 0:
        def_quality = "soft" if def_epa > 0 else "tough"
        parts.append(f"  Defense: {def_epa:+.3f} EPA ({def_quality})")

    # Snap share if available
    snap_share = row.get('snap_share', 0) or 0
    if snap_share > 0:
        parts.append(f"  Snaps: {snap_share:.0%}")

    # ENSEMBLE DISABLED - Using V23 model direction only
    model_direction = 'UNDER' if p_under > 0.5 else 'OVER'

    # ============================================
    # DECISION SECTION
    # ============================================
    parts.append(f"")
    lvt_gap = line - trailing  # Line vs Trailing (positive = line above trailing = UNDER signal)
    parts.append(f"PICK: {pick} (LVT: {lvt_gap:+.1f}) | Edge: {edge:+.1f}%")

    return " | ".join([p for p in parts if p])


def format_confidence_pct(confidence_value) -> str:
    """
    Format confidence value as percentage.

    Handles both numeric (0-1) and string (ELITE/HIGH/STANDARD/LOW) formats.
    """
    if isinstance(confidence_value, (int, float)):
        return f"{confidence_value * 100:.0f}%"
    elif isinstance(confidence_value, str):
        # Map tier names to approximate percentages
        tier_pcts = {
            'ELITE': '90%+',
            'HIGH': '75-90%',
            'STANDARD': '60-75%',
            'LOW': '<60%'
        }
        return tier_pcts.get(confidence_value.upper(), confidence_value)
    return str(confidence_value)


def load_active_model():
    """
    Load the active model via registry (single source of truth).

    Uses the classifier_registry to load the active model, which provides:
    - models: Dict of market-specific models
    - thresholds: Dict of market-specific thresholds
    - validation_results: Dict of validation metrics
    - fingerprint: Feature fingerprint for version tracking

    Returns:
        Tuple of (models_dict, thresholds_dict, validation_dict)
    """
    global _active_model, _active_thresholds, _active_validation, _active_version

    if _active_model is None:
        try:
            # Use registry to load active model (version-free architecture)
            bundle = get_registry_model()
            if bundle is None:
                # Fallback to direct load if registry not available
                if ACTIVE_MODEL_PATH.exists():
                    bundle = joblib.load(ACTIVE_MODEL_PATH)
                else:
                    logger.warning("No active model found in registry or filesystem")
                    return None, {}, {}

            _active_model = bundle.get('models', {})
            # V17+: Thresholds are stored in 'metrics' key, fallback to 'thresholds' for compatibility
            _active_thresholds = bundle.get('metrics', bundle.get('thresholds', {}))
            _active_validation = bundle.get('validation_results', {})
            _active_version = bundle.get('model_id', bundle.get('version', 'active'))
            fingerprint = bundle.get('fingerprint', 'unknown')

            logger.info(f"✅ Loaded Active Model [{fingerprint}] with {len(_active_model)} markets")
            for market, threshold_info in _active_thresholds.items():
                if isinstance(threshold_info, dict):
                    # V17: Thresholds stored as 'best_threshold' and 'best_roi'
                    threshold = threshold_info.get('best_threshold', threshold_info.get('threshold', 0.55))
                    roi = threshold_info.get('best_roi', threshold_info.get('roi', 0))
                    if not threshold_info.get('excluded', False):
                        logger.info(f"   {market}: {threshold:.0%} threshold, {roi:+.1f}% ROI")

        except Exception as e:
            logger.warning(f"Could not load active model: {e}")

    return _active_model, _active_thresholds, _active_validation


# Cache for player historical rates
_player_historical_rates = None
_market_historical_rates = None


def _load_historical_rates():
    """Load player and market historical under rates for V17 model."""
    global _player_historical_rates, _market_historical_rates

    if _player_historical_rates is None:
        player_rates_path = DATA_DIR / 'player_historical_rates.csv'
        market_rates_path = DATA_DIR / 'market_historical_rates.csv'

        if player_rates_path.exists():
            _player_historical_rates = pd.read_csv(player_rates_path)
            logger.debug(f"Loaded {len(_player_historical_rates)} player historical rates")
        else:
            _player_historical_rates = pd.DataFrame()

        if market_rates_path.exists():
            _market_historical_rates = pd.read_csv(market_rates_path)
        else:
            _market_historical_rates = pd.DataFrame()

    return _player_historical_rates, _market_historical_rates


def apply_active_model(row: dict, market: str, trailing_def_epa: float = 0.0) -> dict:
    """
    Apply the active V17 model to get UNDER probability for a given market.

    V17 model uses 14 features including interactions and new defense/rest features.
    Uses percentage-based LVT for normalized sensitivity across all markets.
    Now uses actual player historical under_rates instead of defaults.

    Args:
        row: Dict with player data including line, model_projection, trailing stats
        market: The market type (player_rush_yds, player_receptions, etc.)
        trailing_def_epa: Opponent's trailing defense EPA (positive = bad defense)

    Returns:
        Dict with model_p_under, model_best_direction, model_best_ev, model_validated
    """
    import xgboost as xgb
    from nfl_quant.features import get_feature_engine
    from nfl_quant.utils.player_names import normalize_player_name

    models, thresholds, validation = load_active_model()

    if models is None or market not in models:
        return {
            'model_p_under': 0.50,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'{MODEL_VERSION_FULL} model not available for {market}'
        }

    # Get the market-specific model
    model = models.get(market)
    threshold_info = thresholds.get(market, {})

    # Check if market is excluded
    if isinstance(threshold_info, dict) and threshold_info.get('excluded', False):
        return {
            'model_p_under': 0.50,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'Market excluded: {threshold_info.get("reason", "")}'
        }

    # Get FeatureEngine for centralized feature calculation
    engine = get_feature_engine()

    line = row.get('line', 0)
    # FIX: Use trailing_stat FIRST (actual historical average), then model_projection as fallback
    # trailing_stat = actual 4-week EWMA from NFLverse
    # model_projection = MC simulation output (adjusted/calibrated mean)
    trailing = row.get('trailing_stat', row.get('hist_avg', row.get('model_projection', line)))

    # Load player historical rates for actual under_rate/bias values
    player_rates, market_rates = _load_historical_rates()

    # Look up player's actual historical under rate for this market
    player_name = row.get('player', row.get('player_name', ''))
    player_norm = normalize_player_name(player_name)

    player_under_rate = 0.5  # default
    player_bias = 0.0        # default

    if len(player_rates) > 0:
        player_match = player_rates[
            (player_rates['player_norm'] == player_norm) &
            (player_rates['market'] == market)
        ]
        if len(player_match) > 0:
            player_under_rate = player_match['under_rate'].iloc[0]
            player_bias = player_match['player_bias'].iloc[0]

    # Look up market-level under rate
    market_under_rate = 0.5  # default
    if len(market_rates) > 0:
        market_match = market_rates[market_rates['market'] == market]
        if len(market_match) > 0:
            market_under_rate = market_match['market_under_rate'].iloc[0]

    # Build all core features using FeatureEngine
    # Uses percentage-based LVT for normalized sensitivity across markets
    # Now uses actual player historical under_rates from lookup table
    features = engine.calculate_core_features(
        line=line,
        trailing_stat=trailing,
        player_under_rate=player_under_rate,  # Actual player history
        player_bias=player_bias,              # Actual player bias
        market_under_rate=market_under_rate,  # Actual market rate
        market=market,
        opp_position_def_epa=trailing_def_epa,  # Use opponent's defense EPA
        days_rest=7             # Default: standard rest
    )

    # Extract skill features for model
    # These address the feature bottleneck identified in gap analysis
    player_id = row.get('player_id', '')
    position = row.get('position', 'WR')
    opponent = row.get('opponent', row.get('opponent_team', ''))
    team = row.get('team', '')

    # Get week and season from row (CRITICAL: these were missing before!)
    week = row.get('week', 14)  # Default to week 14 if not provided
    season = row.get('season', 2025)  # Use current season

    # Add skill features if model expects them
    if player_id and market in ['player_receptions', 'player_reception_yds']:
        # Get NGS features (separation, cushion)
        features['avg_separation'] = engine.get_avg_separation(player_id, season, week)
        features['avg_cushion'] = engine.get_avg_cushion(player_id, season, week)
        features['trailing_catch_rate'] = engine.get_trailing_catch_rate(player_id, season, week)
        features['target_share'] = engine.get_target_share(player_id, season, week)

    # Get snap share for all markets
    if player_name:
        features['snap_share'] = engine.get_snap_share(player_name, season, week)

    # WR1-specific defense for receptions market
    if market == 'player_receptions' and position == 'WR' and opponent:
        features['opp_wr1_receptions_allowed'] = engine.get_wr1_defense_stat(opponent, season, week)

    # Use features from centralized config
    # Model stores its expected features in model data, but we use config FEATURES
    # which should match the trained model
    feature_cols = FEATURES
    logger.debug(f"Using {len(feature_cols)} features from config")

    # Build feature vector, using defaults for any missing features
    feature_values = []
    for col in feature_cols:
        feature_values.append(features.get(col, 0.0))

    X = pd.DataFrame([feature_values], columns=feature_cols)

    # Initialize feature contributions
    top_contributions = []

    try:
        # V17 model is XGBoost, get booster and use proper DMatrix
        booster = model.get_booster() if hasattr(model, 'get_booster') else model
        # Use .values and explicit feature_names to avoid DMatrix warning
        dmatrix = xgb.DMatrix(X.values, feature_names=feature_cols)
        p_under = float(booster.predict(dmatrix)[0])

        # Get feature contributions (SHAP-like values) for explainability
        try:
            contribs = booster.predict(dmatrix, pred_contribs=True)

            # contribs shape: (1, n_features + 1), last is bias
            feature_contribs = list(zip(feature_cols, contribs[0][:-1]))

            # Sort by absolute value to get most impactful features
            sorted_contribs = sorted(feature_contribs, key=lambda x: abs(x[1]), reverse=True)

            # Get top 3 contributors (positive = pushes toward UNDER, negative = pushes toward OVER)
            top_contributions = sorted_contribs[:5]
        except Exception as contrib_err:
            logger.debug(f"Could not get feature contributions: {contrib_err}")

    except Exception as e:
        logger.debug(f"V17 model prediction failed: {e}")
        # Simple fallback using percentage LVT
        lvt = features.get('line_vs_trailing', 0.0)
        p_under = 0.5 + lvt * 0.3

    # Clamp to valid probability range
    p_under = max(0.01, min(0.99, p_under))

    # Calculate EV
    ev_under = calculate_bet_ev(p_under, -110)
    ev_over = calculate_bet_ev(1 - p_under, -110)

    # Get threshold for this market - V17 uses 'best_threshold'
    threshold = threshold_info.get('best_threshold', threshold_info.get('threshold', 0.55)) if isinstance(threshold_info, dict) else 0.55

    # Format feature contributions for thesis
    # Positive contrib = pushes toward UNDER, Negative = pushes toward OVER
    contrib_str = ""
    if top_contributions:
        # Format top 3 as: "LVT: +0.12, snap_share: -0.08, player_bias: +0.05"
        contrib_parts = []
        for name, val in top_contributions[:3]:
            # Simplify feature names for readability
            simple_name = name.replace('_trailing', '').replace('player_', '').replace('target_share', 'tgt_share')
            contrib_parts.append(f"{simple_name}: {val:+.2f}")
        contrib_str = " | " + ", ".join(contrib_parts)

    # ==========================================================================
    # ROOT FIX V2 (Dec 7, 2025): Use PROPER DISTRIBUTIONS, not normal approximation
    # ==========================================================================
    # PROBLEM: Normal CDF doesn't capture skewed distributions
    #   - MC simulation uses NegativeBinomial (counts) + Lognormal (efficiency)
    #   - Product of these creates right-skewed, non-normal distributions
    #
    # SOLUTION: Use distribution-appropriate CDFs
    #   - Count stats (receptions, attempts): Negative Binomial CDF
    #   - Yards stats (receiving, rushing, passing): Gamma CDF (captures right-skew)
    #   - This properly models the skewness of NFL stat distributions
    # ==========================================================================
    from scipy.stats import norm, gamma, nbinom

    model_projection = row.get('model_projection', trailing)  # MC simulation mean
    lvt_pct = features.get('line_vs_trailing', 0.0)

    # Get MC distribution parameters (mean and std) from predictions
    # Map market to prediction column names
    MARKET_TO_STAT_COL = {
        'player_receptions': ('receptions_mean', 'receptions_std'),
        'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
        'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
        'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
        'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
        'player_pass_completions': ('passing_completions_mean', 'passing_completions_std'),
        'player_pass_attempts': ('passing_attempts_mean', 'passing_attempts_std'),
        'player_pass_tds': ('passing_tds_mean', 'passing_tds_std'),
    }

    mean_col, std_col = MARKET_TO_STAT_COL.get(market, ('model_projection', None))
    mc_mean = row.get(mean_col, model_projection)
    mc_std = row.get(std_col, abs(model_projection * 0.3)) if std_col else abs(model_projection * 0.3)

    # Ensure std is reasonable (at least 0.5 for count stats, 5.0 for yards)
    is_count_stat = 'receptions' in market or 'attempts' in market or 'completions' in market or 'tds' in market
    min_std = 0.5 if is_count_stat else 5.0
    mc_std = max(mc_std, min_std)

    # Calculate P(UNDER) using distribution-appropriate CDF
    # This captures the actual shape of our MC simulation
    if is_count_stat:
        # COUNT STATS: Use Negative Binomial CDF (discrete, overdispersed)
        # NegBin parameterization: mean μ = n*(1-p)/p, variance σ² = μ/p
        # Solve: p = μ/σ², n = μ*p/(1-p)
        if mc_std > 0 and mc_mean > 0:
            variance = mc_std ** 2
            # Ensure variance > mean (required for NegBin)
            if variance <= mc_mean:
                variance = mc_mean * 1.5  # Force overdispersion
            p_negbin = mc_mean / variance
            p_negbin = max(0.01, min(0.99, p_negbin))  # Bound p
            n_negbin = mc_mean * p_negbin / (1 - p_negbin)
            n_negbin = max(1, n_negbin)  # n must be >= 1
            # P(X < line) for discrete: CDF at floor(line - 0.5)
            # For half-integer lines like 5.5, we want P(X <= 5)
            mc_p_under = float(nbinom.cdf(int(line - 0.5), n_negbin, p_negbin))
        else:
            mc_p_under = 0.5
    else:
        # YARDS STATS: Use Gamma CDF (continuous, right-skewed)
        # Gamma parameterization: shape k = (μ/σ)², scale θ = σ²/μ
        if mc_std > 0 and mc_mean > 0:
            shape = (mc_mean / mc_std) ** 2
            scale = (mc_std ** 2) / mc_mean
            mc_p_under = float(gamma.cdf(line, a=shape, scale=scale))
        else:
            mc_p_under = 0.5

    # Use MC probability instead of (or blended with) classifier
    # For now, use MC probability as the primary source since classifier features were broken
    # TODO: Once features are fixed, consider ensemble of classifier + MC
    effective_p_under = mc_p_under

    # Calculate EV using MC-based probability
    mc_ev_under = calculate_bet_ev(effective_p_under, -110)
    mc_ev_over = calculate_bet_ev(1 - effective_p_under, -110)

    # Log comparison for debugging
    logger.debug(f"MC vs Classifier: mc_p_under={mc_p_under:.1%} (mean={mc_mean:.1f}, std={mc_std:.1f}), "
                 f"xgb_p_under={p_under:.1%}, line={line}")

    # DIRECTION FROM MC DISTRIBUTION (probability-based, not just mean)
    projection_direction = 'UNDER' if effective_p_under > 0.5 else 'OVER'
    projection_diff = line - mc_mean  # Positive = projection below line = UNDER
    projection_diff_pct = projection_diff / line if line > 0 else 0

    # Check market direction constraints
    direction_constraint = MARKET_DIRECTION_CONSTRAINTS.get(market, None)

    # Calculate edge based on MC probability direction
    if projection_direction == 'UNDER':
        direction_prob = effective_p_under
        direction_ev = mc_ev_under
    else:
        direction_prob = 1 - effective_p_under
        direction_ev = mc_ev_over

    # Store both probabilities for transparency
    p_under = effective_p_under  # Use MC-based probability

    # Log when classifier and projection disagree (for monitoring)
    classifier_direction = 'UNDER' if p_under > 0.5 else 'OVER'
    if classifier_direction != projection_direction:
        logger.debug(f"Direction conflict: classifier={classifier_direction} ({p_under:.1%}), "
                    f"projection={projection_direction} (proj={model_projection:.1f} vs line={line})")

    # Apply direction constraint (e.g., pass_yds UNDER_ONLY, anytime_td OVER_ONLY)
    if projection_direction == 'OVER' and direction_constraint == 'UNDER_ONLY':
        return {
            'model_p_under': p_under,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'{MODEL_VERSION_FULL}: OVER blocked for {market} (backtest: -35% ROI)',
            'feature_contributions': top_contributions
        }
    if projection_direction == 'UNDER' and direction_constraint == 'OVER_ONLY':
        return {
            'model_p_under': p_under,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'{MODEL_VERSION_FULL}: UNDER blocked for {market} (can only bet YES)',
            'feature_contributions': top_contributions
        }

    # Validate with minimum edge threshold
    min_edge = 0.02  # 2% minimum edge
    if direction_ev > min_edge and abs(projection_diff_pct) > 0.02:  # Projection differs by >2%
        return {
            'model_p_under': p_under,
            'model_best_direction': projection_direction,
            'model_best_ev': direction_ev * 100,
            'model_validated': True,
            'model_reason': f'{MODEL_VERSION_FULL}: {projection_direction} (proj={model_projection:.1f} vs line={line}, diff={projection_diff_pct:+.0%}){contrib_str}',
            'feature_contributions': top_contributions
        }
    else:
        return {
            'model_p_under': p_under,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'{MODEL_VERSION_FULL}: Insufficient edge (proj={model_projection:.1f} vs line={line}, EV={direction_ev*100:.1f}%)',
            'feature_contributions': top_contributions
        }


# Legacy function aliases for backward compatibility
def load_v16_model():
    """Legacy alias - use load_active_model() instead."""
    models, thresholds, validation = load_active_model()
    return {'models': models, 'thresholds': thresholds, 'validation_results': validation}


def apply_model(row: dict, market: str) -> dict:
    """
    Apply the active model to get UNDER probability.

    Args:
        row: Dict with ALL player data including stats, shares, EPA
        market: Market type (player_receptions, player_rush_yds, etc.)

    Returns:
        Dict with model_p_under, model_best_direction, model_best_ev, model_validated
    """
    bundle = load_active_model()
    models, thresholds, _ = bundle if bundle else (None, None, None)

    if models is None:
        return {
            'model_p_under': None,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': 'Model not available'
        }

    if market not in models:
        return {
            'model_p_under': None,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'Model not available for {market}'
        }

    market_bundle = models[market]
    model = market_bundle['model']
    scaler = market_bundle['scaler']
    imputer = market_bundle['imputer']
    features = market_bundle['features']
    thresh_info = thresholds.get(market, {})
    threshold = thresh_info.get('threshold', 0.55)

    # Check if market is excluded
    if thresh_info.get('excluded', False):
        return {
            'model_p_under': None,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'Market excluded: {thresh_info.get("reason", "negative ROI")}'
        }

    # Extract features from row
    feature_values = []
    for feat in features:
        val = row.get(feat, 0)
        if pd.isna(val):
            val = 0
        feature_values.append(val)

    # Prepare input
    X = np.array([feature_values])

    try:
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        raw_prob = float(model.predict_proba(X_scaled)[0, 1])
    except Exception as e:
        logger.debug(f"Model prediction failed: {e}")
        return {
            'model_p_under': None,
            'model_best_direction': None,
            'model_best_ev': 0,
            'model_validated': False,
            'model_reason': f'Prediction error: {e}'
        }

    # Handle TD markets differently
    model_type = thresh_info.get('type', 'over_under')

    if market == 'player_anytime_td':
        # Anytime TD: raw_prob = P(scores TD)
        # For YES/NO markets: validate if P(TD) is high enough vs typical +odds
        p_yes = raw_prob
        p_no = 1 - raw_prob

        # Typical anytime TD odds are +150 to +400
        # Use odds from row if available, else assume +200
        td_odds = row.get('over_price', 200)  # YES price
        if td_odds is None or pd.isna(td_odds):
            td_odds = 200

        ev_yes = calculate_bet_ev(p_yes, td_odds)

        if p_yes >= threshold and ev_yes > 0:
            return {
                'model_p_under': p_no,  # Store P(NO) for consistency
                'model_best_direction': 'YES',
                'model_best_ev': ev_yes * 100,
                'model_validated': True,
                'model_reason': f'{p_yes:.1%} TD prob (threshold {threshold:.0%})'
            }
        else:
            return {
                'model_p_under': p_no,
                'model_best_direction': None,
                'model_best_ev': 0,
                'model_validated': False,
                'model_reason': f'TD prob {p_yes:.1%} below threshold {threshold:.0%}'
            }

    elif market == 'player_pass_tds':
        # Passing TDs: raw_prob = P(2+ TDs) = P(OVER 1.5)
        p_over = raw_prob
        p_under = 1 - raw_prob

        ev_over = calculate_bet_ev(p_over, -110)
        ev_under = calculate_bet_ev(p_under, -110)

        if p_over >= threshold and ev_over > 0:
            return {
                'model_p_under': p_under,
                'model_best_direction': 'OVER',
                'model_best_ev': ev_over * 100,
                'model_validated': True,
                'model_reason': f'{p_over:.1%} OVER (threshold {threshold:.0%})'
            }
        elif p_under >= threshold and ev_under > 0:
            return {
                'model_p_under': p_under,
                'model_best_direction': 'UNDER',
                'model_best_ev': ev_under * 100,
                'model_validated': True,
                'model_reason': f'{p_under:.1%} UNDER (threshold {threshold:.0%})'
            }
        else:
            return {
                'model_p_under': p_under,
                'model_best_direction': None,
                'model_best_ev': 0,
                'model_validated': False,
                'model_reason': f'Below threshold ({p_over:.1%} vs {threshold:.0%})'
            }

    else:
        # Standard Over/Under markets
        p_under = raw_prob

        # Calculate EV
        ev_under = calculate_bet_ev(p_under, -110)
        ev_over = calculate_bet_ev(1 - p_under, -110)

        # Determine best direction
        if p_under >= threshold and ev_under > 0:
            return {
                'model_p_under': p_under,
                'model_best_direction': 'UNDER',
                'model_best_ev': ev_under * 100,
                'model_validated': True,
                'model_reason': f'{p_under:.1%} UNDER (threshold {threshold:.0%})'
            }
        elif (1 - p_under) >= threshold and ev_over > 0:
            return {
                'model_p_under': p_under,
                'model_best_direction': 'OVER',
                'model_best_ev': ev_over * 100,
                'model_validated': True,
                'model_reason': f'{1-p_under:.1%} OVER (threshold {threshold:.0%})'
            }
        else:
            return {
                'model_p_under': p_under,
                'model_best_direction': None,
                'model_best_ev': 0,
                'model_validated': False,
                'model_reason': f'Below threshold ({p_under:.1%} vs {threshold:.0%})'
            }


def load_v6_value_classifier():
    """Load the V6 value classifier model (bidirectional EV-based)."""
    global _v6_model, _v6_metrics, _v6_recommended

    if _v6_model is None:
        # Try V6 first, fall back to V5 for backwards compatibility
        model_path = V6_MODEL_PATH if V6_MODEL_PATH.exists() else V5_MODEL_PATH

        if model_path.exists():
            try:
                bundle = joblib.load(model_path)
                _v6_model = bundle.get('models', {})
                _v6_metrics = bundle.get('metrics', {})
                _v6_recommended = bundle.get('recommended', {})
                version = bundle.get('version', 'unknown')
                logger.info(f"Loaded {version} with {len(_v6_model)} market models")

                # Log key findings if available
                findings = bundle.get('key_findings', {})
                if findings:
                    logger.info(f"  Key findings: {findings.get('no_reliable_edge', 'N/A')}")
            except Exception as e:
                logger.warning(f"Could not load value classifier: {e}")

    return _v6_model, _v6_metrics, _v6_recommended


# Legacy alias for backwards compatibility
def load_v5_edge_classifier():
    """Legacy alias - loads V6 model."""
    models, metrics, _ = load_v6_value_classifier()
    return models, metrics


def load_game_totals_for_td_calculation(week: int) -> dict:
    """
    Load game totals from odds data and calculate expected TDs per game.

    Used for 1st TD probability calculation:
    P(1st TD) = player_expected_tds / game_total_expected_tds

    Args:
        week: Week number

    Returns:
        Dict mapping game_id -> expected_total_tds
        Example: {'2025_13_GB_DET': 6.9, ...}
    """
    game_totals = {}

    # Try to load game odds
    odds_files = [
        Path(f'data/odds_week{week}.csv'),
        Path('data/game_odds.csv'),
    ]

    for odds_file in odds_files:
        if odds_file.exists():
            try:
                df = pd.read_csv(odds_file)
                # Look for over/under lines
                for _, row in df.iterrows():
                    game_id = row.get('game_id', '')
                    side = str(row.get('side', '')).lower()
                    point = row.get('point', 0)

                    # Get total from over/under line
                    if 'over' in side or 'under' in side:
                        if point and point > 30:  # Sanity check
                            # Estimate TDs: total points / 7 (avg points per TD)
                            # Slight reduction for field goals
                            estimated_tds = point / 7.0
                            game_totals[game_id] = estimated_tds

                if game_totals:
                    logger.info(f"Loaded game totals for {len(game_totals)} games (for 1st TD calculation)")
                    return game_totals

            except Exception as e:
                logger.warning(f"Could not load game odds from {odds_file}: {e}")

    # Fallback: use league average
    logger.warning("No game odds found - using league average for 1st TD calculation")
    return {}


def get_game_expected_tds(game_id: str, game_totals: dict) -> float:
    """
    Get expected TDs for a game.

    Args:
        game_id: Game identifier (e.g., '2025_13_GB_DET')
        game_totals: Dict from load_game_totals_for_td_calculation

    Returns:
        Expected total TDs in game (float)
    """
    LEAGUE_AVG_TDS = 5.0  # NFL average ~5 TDs per game

    if game_id in game_totals:
        return game_totals[game_id]

    # Try partial match (in case game_id format differs)
    for gid, tds in game_totals.items():
        if gid in game_id or game_id in gid:
            return tds

    return LEAGUE_AVG_TDS


def calculate_bet_ev(prob, odds):
    """
    Calculate Expected Value for a bet.

    Args:
        prob: Our estimated probability of winning
        odds: American odds

    Returns:
        EV as decimal (e.g., 0.05 = +5% EV)
    """
    if odds < 0:
        payout = 100 / abs(odds)
    else:
        payout = odds / 100

    # EV = (prob_win * payout) - (prob_lose * stake)
    return prob * payout - (1 - prob)


# kelly_bet_size removed - use calculate_kelly_fraction from nfl_quant.core.unified_betting
# which is already imported at top of file

# Line movement integration
_line_movement = None

def load_line_movement(week: int) -> dict:
    """Load line movement data for the week if available."""
    global _line_movement
    if _line_movement is not None:
        return _line_movement

    movement_path = project_root / 'data' / 'line_movement' / f'line_movement_week{week}.csv'
    if not movement_path.exists():
        logger.info("No line movement data available (run fetch_line_movement.py)")
        return {}

    try:
        df = pd.read_csv(movement_path)
        _line_movement = {}
        for _, row in df.iterrows():
            key = (str(row['player']).lower().strip(), row['market'])
            _line_movement[key] = {
                'line_movement': row['line_movement'],
                'movement_direction': row['movement_direction'],
                'sharp_under': row['movement_direction'] == 'down',
                'sharp_over': row['movement_direction'] == 'up',
            }
        logger.info(f"Loaded line movement for {len(_line_movement)} player/markets")
        return _line_movement
    except Exception as e:
        logger.warning(f"Could not load line movement: {e}")
        return {}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_simulation_config() -> dict:
    """Load simulation config from JSON file."""
    config_path = project_root / 'configs' / 'simulation_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load simulation config: {e}")
        return {}

# Load simulation config for tiered calibration
CONFIG_PATH = project_root / 'configs' / 'simulation_config.json'
_sim_config = {}
_calibration_tiers = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _sim_config = json.load(f)
        _calibration_tiers = _sim_config.get('calibration_tiers', {})
        _use_tiered_calibration = _sim_config.get('feature_flags', {}).get('use_tiered_calibration', True)
    logger.info(f"Loaded calibration tiers config: {len(_calibration_tiers)} market types")
except Exception as e:
    logger.warning(f"Could not load simulation config: {e}. Using default shrinkage.")
    _use_tiered_calibration = False


def classify_player_tier(player_data: dict, market: str) -> tuple:
    """
    Classify player into a tier based on snap_share and usage metrics.

    UPDATED 2025-11-26: Now uses market-specific tier keys (e.g., wr_receiving_yards vs wr_receptions)
    to provide correct bias adjustments for both yards and count markets.

    Returns: (tier_name, tier_config) where tier_config contains:
        - bias_adjustment: additive adjustment to model mean (market-specific)
        - shrinkage_factor: how much to shrink probability toward 0.5
        - confidence_boost: adjustment to confidence tier scoring
    """
    if not _use_tiered_calibration or not _calibration_tiers:
        # Return default tier with standard shrinkage
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    snap_share = player_data.get('snap_share', 0.0) or 0.0
    targets = player_data.get('targets_mean', 0.0) or 0.0
    carries = player_data.get('rushing_attempts_mean', 0.0) or 0.0
    position = player_data.get('position', '').upper()

    # Map market to SPECIFIC calibration tier config key
    # New structure: {position}_{category}_{market_type} e.g., wr_receiving_yards, wr_receptions
    market_lower = market.lower()

    # Determine market type (yards vs count)
    is_yards_market = market in ['player_reception_yds', 'player_rush_yds', 'player_pass_yds', 'player_rush_reception_yds']
    is_receptions_market = market == 'player_receptions'
    is_rush_attempts_market = market == 'player_rush_attempts'
    is_pass_completions_market = market == 'player_pass_completions'

    # Map to specific tier key
    if market == 'player_pass_yds':
        tier_key = 'qb_passing_yards'
        usage_value = snap_share  # Use snap share for QBs
        usage_key = 'snap_share_min'
    elif market == 'player_pass_completions':
        tier_key = 'qb_passing_completions'
        usage_value = snap_share
        usage_key = 'snap_share_min'
    elif market == 'player_rush_yds':
        if position == 'QB':
            tier_key = 'qb_rushing_yards'  # QB rushing has its own tier (not passing!)
            usage_value = snap_share
            usage_key = 'snap_share_min'
        else:
            tier_key = 'rb_rushing_yards'
            usage_value = carries
            usage_key = 'carries_min'
    elif market == 'player_rush_attempts':
        if position == 'QB':
            tier_key = 'qb_rushing_attempts'  # QB rushing attempts tier
            usage_value = snap_share
            usage_key = 'snap_share_min'
        else:
            tier_key = 'rb_rushing_attempts'
            usage_value = carries
            usage_key = 'carries_min'
    elif market == 'player_reception_yds':
        if position == 'RB':
            tier_key = 'rb_receiving_yards'
        elif position == 'TE':
            tier_key = 'te_receiving_yards'
        else:  # WR or unknown
            tier_key = 'wr_receiving_yards'
        usage_value = targets
        usage_key = 'targets_min'
    elif market == 'player_receptions':
        if position == 'RB':
            tier_key = 'rb_receptions'
        elif position == 'TE':
            tier_key = 'te_receptions'
        else:  # WR or unknown
            tier_key = 'wr_receptions'
        usage_value = targets
        usage_key = 'targets_min'
    else:
        # Default for other markets (TDs, etc.)
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    tier_config = _calibration_tiers.get(tier_key, {})

    if not tier_config:
        # Tier key not found in config, return default
        logger.debug(f"No tier config found for {tier_key}, using default")
        return ('default', {
            'bias_adjustment': 0.0,
            'shrinkage_factor': 0.30,
            'confidence_boost': 0.0
        })

    # Iterate through tiers in order (highest first due to dict ordering)
    # Tiers are named tier_1_*, tier_2_*, etc.
    for tier_name, tier_params in sorted(tier_config.items(), key=lambda x: x[0]):
        if tier_name.startswith('_'):
            continue  # Skip comment keys
        if not isinstance(tier_params, dict):
            continue  # Skip non-dict entries

        snap_min = tier_params.get('snap_share_min', 0.0)
        usage_min = tier_params.get(usage_key, tier_params.get('targets_min', tier_params.get('carries_min', 0.0)))

        if snap_share >= snap_min and usage_value >= usage_min:
            return (tier_name, {
                'bias_adjustment': tier_params.get('bias_adjustment', 0.0),
                'shrinkage_factor': tier_params.get('shrinkage_factor', 0.30),
                'confidence_boost': tier_params.get('confidence_boost', 0.0)
            })

    # Fallback to last tier if nothing matched
    return ('tier_4_default', {
        'bias_adjustment': 0.0,
        'shrinkage_factor': 0.45,
        'confidence_boost': -0.05
    })


def get_current_season() -> int:
    """Determine current NFL season."""
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


def get_current_week() -> int:
    """Estimate current NFL week."""
    now = datetime.now()
    if now.month >= 9:
        return min(18, (now - datetime(now.year, 9, 1)).days // 7 + 1)
    elif now.month <= 2:
        return 18
    else:
        return 1


def american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def get_market_priority(position: str, market: str) -> str:
    """
    Assign market priority based on historical model accuracy.

    Priority levels:
    - HIGH: Most accurate predictions (WR receptions, RB attempts, QB pass yds)
    - MEDIUM: Good accuracy (WR rec yds, RB receptions, TE targets)
    - STANDARD: Less proven markets (TDs, longest reception, alt lines)

    Args:
        position: Player position (QB, RB, WR, TE)
        market: Market type (player_receptions, player_reception_yds, etc.)

    Returns:
        Priority level: "HIGH", "MEDIUM", or "STANDARD"
    """
    # HIGH priority combinations (most accurate)
    high_priority = {
        'WR': ['player_receptions', 'player_targets'],
        'RB': ['player_rush_attempts'],
        'TE': ['player_receptions'],
        'QB': ['player_pass_yds']
    }

    # MEDIUM priority combinations (good accuracy)
    medium_priority = {
        'WR': ['player_reception_yds'],
        'RB': ['player_receptions', 'player_targets'],
        'TE': ['player_targets'],
        'QB': ['player_pass_completions', 'player_pass_attempts']
    }

    if position in high_priority and market in high_priority[position]:
        return 'HIGH'
    elif position in medium_priority and market in medium_priority[position]:
        return 'MEDIUM'
    else:
        return 'STANDARD'


def load_model_predictions(week: int) -> pd.DataFrame:
    """Load model predictions for specified week."""
    pred_file = Path(f'data/model_predictions_week{week}.csv')
    if not pred_file.exists():
        raise FileNotFoundError(f"Model predictions not found: {pred_file}")

    df = pd.read_csv(pred_file)
    logger.info(f"Loaded {len(df)} player predictions from {pred_file}")
    return df


def detect_week_from_props() -> int:
    """
    Auto-detect NFL week from props file game dates.

    This solves the issue where props are fetched before the week starts,
    or when running predictions multiple times during a week.
    """
    props_file = Path('data/nfl_player_props_draftkings.csv')
    sched_file = Path('data/nflverse/schedules.parquet')

    if not props_file.exists() or not sched_file.exists():
        return get_current_week()  # Fallback

    try:
        props = pd.read_csv(props_file)
        sched = pd.read_parquet(sched_file)
        sched = sched[sched['season'] == get_current_season()]

        # Extract game dates from props
        if 'commence_time' in props.columns:
            props['game_date'] = pd.to_datetime(props['commence_time']).dt.date
        elif 'game_time' in props.columns:
            props['game_date'] = pd.to_datetime(props['game_time']).dt.date
        else:
            return get_current_week()

        sched['gameday'] = pd.to_datetime(sched['gameday']).dt.date

        # Count which NFL week the props belong to
        week_counts = {}
        for prop_date in props['game_date'].unique():
            matching = sched[sched['gameday'] == prop_date]
            if len(matching) > 0:
                week = int(matching['week'].iloc[0])
                week_counts[week] = week_counts.get(week, 0) + 1

        if week_counts:
            detected_week = max(week_counts, key=week_counts.get)
            logger.info(f"Auto-detected Week {detected_week} from props file dates")
            return detected_week

    except Exception as e:
        logger.warning(f"Could not auto-detect week from props: {e}")

    return get_current_week()


def load_prop_lines() -> pd.DataFrame:
    """Load current prop lines from DraftKings."""
    props_file = Path('data/nfl_player_props_draftkings.csv')
    if not props_file.exists():
        raise FileNotFoundError(f"Props file not found: {props_file}")

    df = pd.read_csv(props_file)
    # Normalize column names for consistency across different fetch formats
    column_renames = {}
    if 'player' in df.columns and 'player_name' not in df.columns:
        column_renames['player'] = 'player_name'
    if 'direction' in df.columns and 'outcome_type' not in df.columns:
        column_renames['direction'] = 'outcome_type'
    if column_renames:
        df = df.rename(columns=column_renames)

    # Handle odds column - use over_odds as default 'odds' if not present
    if 'odds' not in df.columns:
        if 'over_odds' in df.columns:
            df['odds'] = df['over_odds']  # Default to over odds
        elif 'market_odds' in df.columns:
            df['odds'] = df['market_odds']

    logger.info(f"Loaded {len(df)} prop lines from {props_file}")
    return df


def filter_props_by_game_status(
    props_df: pd.DataFrame,
    week: int,
    season: int = 2025
) -> pd.DataFrame:
    """Filter out prop lines for games that have started or completed.

    Uses real-time kickoff detection to skip in-progress games,
    preventing wasted compute on stale predictions.

    Args:
        props_df: DataFrame with prop lines (must have home_team, away_team columns)
        week: NFL week number
        season: NFL season year

    Returns:
        Filtered DataFrame with only pre-game props
    """
    from nfl_quant.utils.team_names import normalize_team_name

    if props_df.empty:
        return props_df

    # Get game status map
    game_status_map = load_game_status_map(week, season)

    if not game_status_map:
        logger.warning("Could not load game status map - skipping game status filter")
        return props_df

    original_count = len(props_df)

    def normalize_to_nflverse(team_name: str) -> str:
        """Normalize team name to NFLverse abbreviation format."""
        abbr = normalize_team_name(team_name)
        # NFLverse uses 'LA' for Rams, not 'LAR'
        if abbr == 'LAR':
            return 'LA'
        return abbr

    # Build game_id for each prop row and filter
    def get_game_status(row):
        away = row.get('away_team', '')
        home = row.get('home_team', '')
        if not away or not home:
            return 'unknown'
        # Normalize team names to NFLverse abbreviations
        away_abbr = normalize_to_nflverse(away)
        home_abbr = normalize_to_nflverse(home)
        # NFLverse game_id format: {season}_{week}_{away}_{home}
        game_id = f"{season}_{week}_{away_abbr}_{home_abbr}"
        return game_status_map.get(game_id, 'unknown')

    props_df = props_df.copy()
    props_df['_game_status'] = props_df.apply(get_game_status, axis=1)

    # Keep only pre-game props (or unknown if we can't match)
    pre_game_mask = props_df['_game_status'].isin(['pre_game', 'unknown'])
    filtered_df = props_df[pre_game_mask].drop(columns=['_game_status'])

    skipped_count = original_count - len(filtered_df)
    if skipped_count > 0:
        # Log what we're skipping
        skipped_games = props_df[~pre_game_mask][['home_team', 'away_team', '_game_status']].drop_duplicates()
        logger.info(f"Skipped {skipped_count} props from {len(skipped_games)} in-progress/complete games:")
        for _, row in skipped_games.iterrows():
            logger.info(f"   {row['away_team']}@{row['home_team']}: {row['_game_status']}")

    logger.info(f"Game status filter: {original_count} -> {len(filtered_df)} props ({len(filtered_df)} actionable)")
    return filtered_df


def match_player_name(model_name: str, prop_name: str) -> bool:
    """Check if player names match (handles variations)."""
    # Normalize names
    model_clean = model_name.lower().strip().replace('.', '').replace("'", "")
    prop_clean = prop_name.lower().strip().replace('.', '').replace("'", "")

    # Direct match
    if model_clean == prop_clean:
        return True

    # Check if one contains the other
    if model_clean in prop_clean or prop_clean in model_clean:
        return True

    # Handle suffix variations (Jr., Sr., III, etc.)
    suffixes = [' jr', ' sr', ' iii', ' ii', ' iv']
    for suffix in suffixes:
        if model_clean.replace(suffix, '') == prop_clean.replace(suffix, ''):
            return True

    return False


def get_model_stat_for_market(player_row: pd.Series, market: str) -> tuple:
    """
    Get the appropriate model stat (mean and std) for a given market type.

    Returns (mean, std) tuple.
    """
    market_to_stat = {
        'player_receptions': ('receptions_mean', 'receptions_std'),
        'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
        'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
        'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
        'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
        'player_pass_tds': ('passing_tds_mean', 'passing_tds_std'),
        'player_pass_attempts': ('passing_attempts_mean', 'passing_attempts_std'),
        'player_pass_completions': ('passing_completions_mean', 'passing_completions_std'),
        'player_anytime_td': (None, None),  # Special case - calculated below
        'player_1st_td': (None, None),  # Special case - calculated below
        'player_last_td': (None, None),  # Special case - calculated below
        'player_rush_reception_yds': (None, None),  # Will calculate as sum
        'player_longest_reception': (None, None),  # Special case - calculated below
        'player_reception_longest': (None, None),  # DK naming
        'player_rush_longest': (None, None),  # Special case - calculated below
        'player_interceptions': (None, None),  # Special case
        'player_pass_interceptions': (None, None),  # DK naming
        'player_sacks': (None, None),  # Defensive stat
        'player_solo_tackles': (None, None),  # Defensive stat
        'player_tackles_assists': (None, None),  # Defensive stat
        'player_kicking_points': (None, None),  # Kicker stat
        'player_field_goals': (None, None),  # Kicker stat
    }

    if market not in market_to_stat:
        return (None, None)

    mean_col, std_col = market_to_stat[market]

    # Special case for combined rush + rec yards
    if market == 'player_rush_reception_yds':
        rush_mean = player_row.get('rushing_yards_mean', 0)
        rush_std = player_row.get('rushing_yards_std', 0)
        rec_mean = player_row.get('receiving_yards_mean', 0)
        rec_std = player_row.get('receiving_yards_std', 0)

        if pd.isna(rush_mean):
            rush_mean = 0
        if pd.isna(rec_mean):
            rec_mean = 0
        if pd.isna(rush_std):
            rush_std = 0
        if pd.isna(rec_std):
            rec_std = 0

        combined_mean = rush_mean + rec_mean

        # Combine variances using Pythagorean addition (assumes independence)
        if rush_std > 0 or rec_std > 0:
            combined_std = np.sqrt(rush_std**2 + rec_std**2)
        else:
            # If both std values are missing, raise error instead of hardcoding
            logger.error(f"Missing both rushing_std and receiving_std for combined market")
            raise ValueError(
                "Combined rush+reception market requires both rushing_std and receiving_std. "
                "Ensure model_predictions_weekX.csv includes these columns."
            )

        return (combined_mean, combined_std)

    # Special case for Anytime TD (combine rushing + receiving TDs)
    if market in ['player_anytime_td', 'player_1st_td', 'player_last_td']:
        rush_tds = player_row.get('rushing_tds_mean', 0)
        rec_tds = player_row.get('receiving_tds_mean', 0)
        pass_tds = player_row.get('passing_tds_mean', 0)  # For QBs rushing TDs

        if pd.isna(rush_tds):
            rush_tds = 0
        if pd.isna(rec_tds):
            rec_tds = 0
        if pd.isna(pass_tds):
            pass_tds = 0

        # Calculate anytime TD probability using Poisson
        # P(at least 1 TD) = 1 - P(0 rush TDs) * P(0 rec TDs)
        # For expected TDs, use combined rate
        total_expected_tds = rush_tds + rec_tds

        # For QBs, they can also score rushing TDs (already in rush_tds)
        position = player_row.get('position', '')
        if position == 'QB':
            # QB rushing TDs are already in rushing_tds_mean
            pass

        # Return expected TDs as mean, std approximated using Poisson (std = sqrt(mean))
        if total_expected_tds > 0:
            return (total_expected_tds, np.sqrt(total_expected_tds))
        return (None, None)

    # Special case for Pass Completions (use actual column if available)
    if market == 'player_pass_completions':
        mean_val = player_row.get('passing_completions_mean', None)
        std_val = player_row.get('passing_completions_std', None)
        if mean_val is None or pd.isna(mean_val):
            # Fallback: estimate from attempts * completion rate
            attempts = player_row.get('passing_attempts_mean', 0)
            if attempts and not pd.isna(attempts):
                mean_val = attempts * 0.65  # League avg completion rate
                std_val = np.sqrt(mean_val)  # Poisson approximation
        if mean_val is not None and not pd.isna(mean_val):
            return (mean_val, std_val if std_val and not pd.isna(std_val) else np.sqrt(mean_val))
        return (None, None)

    # Special case for Pass Attempts
    if market == 'player_pass_attempts':
        mean_val = player_row.get('passing_attempts_mean', None)
        std_val = player_row.get('passing_attempts_std', None)
        if mean_val is not None and not pd.isna(mean_val):
            return (mean_val, std_val if std_val and not pd.isna(std_val) else np.sqrt(mean_val))
        return (None, None)

    # Special case for Longest Reception
    # Estimate based on expected yards per reception and number of receptions
    # Using Gumbel distribution approximation: E[max] ≈ μ + σ * √(2 * ln(n))
    if market in ['player_reception_longest', 'player_longest_reception']:
        rec_yards = player_row.get('receiving_yards_mean', 0)
        receptions = player_row.get('receptions_mean', 0)

        if pd.isna(rec_yards) or pd.isna(receptions) or receptions <= 0:
            return (None, None)

        # Average yards per reception
        yards_per_rec = rec_yards / receptions if receptions > 0 else 10

        # Estimate longest using Gumbel approximation
        # For n attempts with mean μ and std σ, E[max] ≈ μ + σ * √(2 * ln(n))
        # Using empirical adjustment: longest ≈ 1.5 * avg_yards_per_rec + 5
        # This captures that longest plays tend to be about 1.5-2x average
        n = max(receptions, 1)
        sigma = yards_per_rec * 0.8  # Typical CV for single reception yards
        expected_longest = yards_per_rec + sigma * np.sqrt(2 * np.log(n + 1))

        # Add some base value (even 1 reception has positive expected yards)
        expected_longest = max(expected_longest, yards_per_rec * 1.2)

        # Std is high for longest plays (empirically ~13 yards)
        longest_std = 13.0

        return (expected_longest, longest_std)

    # Special case for Longest Rush
    if market in ['player_rush_longest']:
        rush_yards = player_row.get('rushing_yards_mean', 0)
        attempts = player_row.get('rushing_attempts_mean', 0)

        if pd.isna(rush_yards) or pd.isna(attempts) or attempts <= 0:
            return (None, None)

        yards_per_carry = rush_yards / attempts if attempts > 0 else 4

        n = max(attempts, 1)
        sigma = yards_per_carry * 1.0  # Higher variance for rushing
        expected_longest = yards_per_carry + sigma * np.sqrt(2 * np.log(n + 1))
        expected_longest = max(expected_longest, yards_per_carry * 1.3)

        longest_std = 11.0  # Empirical std for longest rush

        return (expected_longest, longest_std)

    # Special case for QB Pass Interceptions
    if market in ['player_pass_interceptions', 'player_interceptions']:
        # Use passing attempts and historical INT rate
        pass_attempts = player_row.get('passing_attempts_mean', 0)
        if pd.isna(pass_attempts) or pass_attempts <= 0:
            return (None, None)

        # League average INT rate is about 2.5%
        int_rate = 0.025
        expected_ints = pass_attempts * int_rate

        return (expected_ints, np.sqrt(expected_ints))

    # Defensive stats - these require roster-level modeling which we don't have yet
    # Skip for now - return None to exclude these markets
    if market in ['player_sacks', 'player_solo_tackles', 'player_tackles_assists']:
        # TODO: Add defensive player prediction model
        # For now, skip these markets as we don't have reliable predictions
        return (None, None)

    # Kicking stats
    if market in ['player_kicking_points', 'player_field_goals', 'player_pats']:
        # TODO: Add kicker prediction model based on team scoring opportunities
        # For now, skip these markets
        return (None, None)

    mean_val = player_row.get(mean_col, None) if mean_col else None
    std_val = player_row.get(std_col, None) if std_col else None

    if pd.isna(mean_val):
        mean_val = None
    if std_val is not None and pd.isna(std_val):
        std_val = None

    return (mean_val, std_val)


def calculate_over_under_prob(mean: float, std: float, line: float, market: str,
                               game_total_tds: float = None) -> tuple:
    """
    Calculate probability of going Over and Under the line.

    Uses normal distribution for continuous stats (yards) and
    Poisson for count stats (receptions, TDs).

    For 1st TD market: Uses player's share of expected TDs in game.

    Args:
        mean: Expected value (e.g., expected TDs for TD markets)
        std: Standard deviation from Monte Carlo
        line: Betting line
        market: Market type
        game_total_tds: Total expected TDs in game (required for 1st TD calculation)

    CRITICAL: std should come from model_std (Monte Carlo simulation output).
    If model_std is missing, raise an error instead of using hardcoded fallback.
    """
    if mean is None or mean <= 0:
        return (0.5, 0.5)

    # Validate std is provided from model predictions
    # CRITICAL: Reject near-zero std (indicates Monte Carlo bug or constant prediction)
    MIN_STD_THRESHOLD = 0.01  # Minimum realistic std for any stat

    if std is None or std <= 0:
        logger.error(f"Missing or invalid model_std for market {market}, mean={mean}")
        raise ValueError(
            f"model_std is required but got {std}. "
            "Ensure model_predictions_weekX.csv includes std columns. "
            "Do not hardcode CV values - use actual Monte Carlo simulation output."
        )

    # BUG FIX: Detect near-zero std (Monte Carlo returned all identical values)
    if std < MIN_STD_THRESHOLD:
        logger.warning(
            f"Near-zero model_std detected: {std:.2e} for {market}, mean={mean}. "
            f"Applying minimum variance threshold of {MIN_STD_THRESHOLD}. "
            f"Check Monte Carlo simulation for this player - may have constant predictions."
        )
        # Use minimum realistic std based on stat type
        if market in ['player_receptions', 'player_rush_attempts', 'player_pass_attempts']:
            std = max(std, np.sqrt(mean))  # Poisson: variance = mean
        else:
            std = max(std, mean * 0.2)  # Yards: assume 20% CV minimum

    # P0 FIX 3: Heteroscedastic variance scaling (2025-12-07)
    # Analysis showed: actual_std = 2.57 + 0.62 * mean (R² = 0.98)
    # The Monte Carlo std is underestimated by ~40%, especially for high predictions
    # Apply scaling to match observed variance pattern from backtest
    HETEROSCEDASTIC_INTERCEPT = 2.57  # Base std for low predictions
    HETEROSCEDASTIC_SLOPE = 0.62  # How much std increases per unit of mean

    # Calculate expected std based on prediction magnitude
    expected_std_from_mean = HETEROSCEDASTIC_INTERCEPT + HETEROSCEDASTIC_SLOPE * mean

    # Use the maximum of Monte Carlo std and expected std
    # This ensures we don't underestimate uncertainty for high predictions
    original_std = std
    std = max(std, expected_std_from_mean)

    # Log if we had to adjust significantly
    if std > original_std * 1.5:
        logger.debug(
            f"Heteroscedastic adjustment for {market}: "
            f"MC std {original_std:.2f} -> {std:.2f} (mean={mean:.1f})"
        )

    # Count-based stats use Poisson
    count_markets = ['player_receptions', 'player_rush_attempts', 'player_pass_tds',
                     'player_pass_attempts', 'player_pass_completions']

    # TD markets - different logic for each type
    if market == 'player_1st_td':
        # 1st TD probability = Player's share of expected TDs
        # P(1st TD) ≈ player_expected_tds / game_total_expected_tds
        #
        # Logic: If a game has 5 expected TDs and player expects 0.5 TDs,
        # their share of the first TD is roughly 0.5/5 = 10%
        #
        # This is an approximation - actual probability depends on:
        # - When in the game the player's TDs are likely to occur
        # - Whether they're more likely to score early (goal-line back) vs late
        # For now, we use equal probability assumption
        if game_total_tds and game_total_tds > 0:
            # Player's share of first TD
            prob_over = min(mean / game_total_tds, 0.95)
        else:
            # Fallback: Use anytime TD logic scaled down
            # Average NFL game has ~5 TDs, so rough estimate
            TYPICAL_GAME_TDS = 5.0
            prob_over = min(mean / TYPICAL_GAME_TDS, 0.95)
        prob_under = 1 - prob_over

    elif market == 'player_last_td':
        # Last TD uses same logic as 1st TD (share of game TDs)
        if game_total_tds and game_total_tds > 0:
            prob_over = min(mean / game_total_tds, 0.95)
        else:
            TYPICAL_GAME_TDS = 5.0
            prob_over = min(mean / TYPICAL_GAME_TDS, 0.95)
        prob_under = 1 - prob_over

    elif market == 'player_anytime_td':
        # Anytime TD: P(at least 1 TD) using Poisson
        # P(Over 0.5) = P(at least 1 TD) = 1 - P(0 TDs) = 1 - e^(-lambda)
        prob_over = 1 - poisson.cdf(0, mean)  # P(X >= 1) = 1 - P(X = 0)
        prob_under = poisson.cdf(0, mean)  # P(X = 0)
    elif market == 'player_receptions':
        # P1-C: Use Negative Binomial for receptions (better for overdispersed counts)
        # Negative Binomial handles variance > mean better than Poisson
        var = std ** 2

        # Ensure variance > mean for nbinom (if not, fall back to Poisson)
        if var <= mean:
            # Poisson is appropriate when variance ≈ mean
            prob_over = 1 - poisson.cdf(line - 0.5, mean)
            prob_under = poisson.cdf(line - 0.5, mean)
        else:
            # Negative Binomial parameterization for scipy:
            # scipy uses: mean = n * (1-p) / p, variance = n * (1-p) / p²
            # Solving: n = mean² / (var - mean), p = mean / var
            try:
                n = mean ** 2 / (var - mean)
                p = mean / var  # This is the success probability

                # Ensure valid parameters
                if n > 0 and 0 < p < 1:
                    # scipy.stats.nbinom uses p as success probability
                    # P(X <= k) is the CDF
                    # For half-line like 4.5, P(Over 4.5) = P(X >= 5) = 1 - P(X <= 4)
                    prob_under = nbinom.cdf(int(line), n, p)
                    prob_over = 1 - prob_under
                else:
                    # Fallback to Poisson if params invalid
                    prob_over = 1 - poisson.cdf(line - 0.5, mean)
                    prob_under = poisson.cdf(line - 0.5, mean)
            except Exception:
                # Fallback to Poisson
                prob_over = 1 - poisson.cdf(line - 0.5, mean)
                prob_under = poisson.cdf(line - 0.5, mean)
    elif market in count_markets:
        # For other counts, P(Over X) = P(count > X) = 1 - P(count <= X-0.5)
        prob_over = 1 - poisson.cdf(line - 0.5, mean)
        prob_under = poisson.cdf(line - 0.5, mean)
    else:
        # For continuous stats, use normal distribution
        z_score = (line - mean) / std
        prob_over = 1 - norm.cdf(z_score)
        prob_under = norm.cdf(z_score)

    return (prob_over, prob_under)


def generate_recommendations(week: int, season: int = None) -> pd.DataFrame:
    """Generate all recommendations by matching model predictions to prop lines."""

    if season is None:
        season = get_current_season()

    logger.info(f"Generating recommendations for Week {week}, Season {season}")

    # Load simulation config for probability caps and strategy filters
    config = load_simulation_config()

    # Load game totals for 1st TD calculation
    game_totals = load_game_totals_for_td_calculation(week)

    # Load data
    model_preds = load_model_predictions(week)
    prop_lines = load_prop_lines()

    # ============================================================================
    # FILTER OUT IN-PROGRESS AND COMPLETED GAMES
    # ============================================================================
    # Skip props for games that have already started (30+ min after kickoff)
    # This prevents wasted compute and stale predictions
    prop_lines = filter_props_by_game_status(prop_lines, week, season)

    if prop_lines.empty:
        logger.warning("No actionable prop lines remain after game status filter!")
        return pd.DataFrame()

    # ============================================================================
    # SNAP RAMP ADJUSTMENTS FOR RETURNING PLAYERS
    # ============================================================================
    # Players returning from IR often have limited snap counts ("pitch count")
    # This adjusts predictions based on expected snap share
    logger.info("=" * 60)
    logger.info("APPLYING SNAP RAMP ADJUSTMENTS FOR RETURNING PLAYERS")
    logger.info("=" * 60)

    # Load manual overrides (takes precedence)
    manual_overrides = load_manual_overrides()
    override_applied = []

    if manual_overrides:
        logger.info(f"Found {len(manual_overrides)} manual snap overrides")
        for key, override in manual_overrides.items():
            if override['week'] == week and override['season'] == season:
                player_name = override['player_name']
                snap_factor = override['snap_factor']
                reason = override.get('reason', 'Manual override')

                # Find player in predictions
                mask = model_preds['player_name'].str.lower().str.contains(player_name.lower(), na=False)
                if mask.any():
                    idx = model_preds[mask].index[0]
                    original_rec = model_preds.loc[idx, 'receptions_mean'] if 'receptions_mean' in model_preds.columns else 0
                    original_yds = model_preds.loc[idx, 'receiving_yards_mean'] if 'receiving_yards_mean' in model_preds.columns else 0

                    # Apply snap factor to volume-based columns
                    for col in ['receptions_mean', 'receiving_yards_mean', 'targets_mean',
                                'rushing_yards_mean', 'rushing_attempts_mean']:
                        if col in model_preds.columns:
                            model_preds.loc[idx, col] *= snap_factor

                    new_rec = model_preds.loc[idx, 'receptions_mean'] if 'receptions_mean' in model_preds.columns else 0
                    new_yds = model_preds.loc[idx, 'receiving_yards_mean'] if 'receiving_yards_mean' in model_preds.columns else 0

                    logger.info(f"   MANUAL OVERRIDE: {player_name}")
                    logger.info(f"      Snap factor: {snap_factor:.0%}")
                    logger.info(f"      Reason: {reason[:60]}...")
                    if original_rec > 0:
                        logger.info(f"      Receptions: {original_rec:.1f} -> {new_rec:.1f}")
                    if original_yds > 0:
                        logger.info(f"      Rec yards: {original_yds:.1f} -> {new_yds:.1f}")

                    override_applied.append(player_name)

    # Also try automatic detection (for players not in manual overrides)
    try:
        weekly_stats = pd.read_parquet(DATA_DIR / 'nflverse' / 'weekly_stats.parquet')
        returning_players = detect_returning_players_snap_status(weekly_stats, week, season)

        # Filter out players already handled by manual overrides
        returning_players = [
            rp for rp in returning_players
            if rp['player_name'] not in override_applied
        ]

        if returning_players:
            logger.info(f"Auto-detected {len(returning_players)} additional returning players")
            model_preds = apply_snap_ramp_adjustments(model_preds, returning_players)
    except Exception as e:
        logger.warning(f"Could not apply automatic snap ramp detection: {e}")

    # DEBUG: Verify adjustments persisted
    logger.info("-" * 60)
    logger.info("SNAP RAMP VERIFICATION (post-adjustment values):")
    debug_players = ['mike evans', 'chris godwin', 'emeka egbuka']
    for dp in debug_players:
        mask = model_preds['player_name'].str.lower().str.contains(dp, na=False)
        if mask.any():
            row = model_preds[mask].iloc[0]
            rec = row.get('receptions_mean', 'N/A')
            yds = row.get('receiving_yards_mean', 'N/A')
            logger.info(f"   {dp.title()}: rec={rec:.2f}, yds={yds:.2f}")
    logger.info("=" * 60)

    # =========================================================================
    # TEAM HEALTH SYNERGY ADJUSTMENTS (V25)
    # Models compound effects when multiple players return simultaneously
    # =========================================================================
    team_synergy_data = {}  # Store synergy results for dashboard/output

    if USE_TEAM_SYNERGY:
        logger.info("=" * 60)
        logger.info("TEAM HEALTH SYNERGY ANALYSIS")
        logger.info("=" * 60)

        try:
            # Load roster and injury data
            rosters_path = DATA_DIR / 'nflverse' / 'rosters.parquet'
            injuries_path = DATA_DIR / 'nflverse' / 'injuries.parquet'
            depth_path = DATA_DIR / 'nflverse' / 'depth_charts.parquet'

            rosters_df = None
            injuries_df = None
            depth_df = None

            if rosters_path.exists():
                rosters_df = pd.read_parquet(rosters_path)
            else:
                # Try CSV fallback
                rosters_csv = DATA_DIR / 'nflverse' / f'rosters_{season}.csv'
                if rosters_csv.exists():
                    rosters_df = pd.read_csv(rosters_csv)

            if injuries_path.exists():
                injuries_df = pd.read_parquet(injuries_path)
            else:
                injuries_csv = DATA_DIR / 'nflverse' / f'injuries_{season}.csv'
                if injuries_csv.exists():
                    injuries_df = pd.read_csv(injuries_csv)

            if depth_path.exists():
                depth_df = pd.read_parquet(depth_path)

            if rosters_df is not None:
                # Get unique teams from predictions
                team_col = 'team' if 'team' in model_preds.columns else 'recent_team'
                if team_col in model_preds.columns:
                    unique_teams = model_preds[team_col].dropna().unique()

                    # PERFORMANCE FIX: Load player statuses ONCE for all teams (not per-team)
                    all_player_statuses = load_player_statuses_from_injuries(
                        injuries_df if injuries_df is not None else pd.DataFrame(),
                        rosters_df,
                        depth_df,
                        None,
                        week,
                        season
                    )

                    # PERFORMANCE FIX: Detect returning players ONCE for all teams
                    if weekly_stats is not None:
                        try:
                            from nfl_quant.features.ir_return_detector import detect_returning_players_snap_status
                            all_returning_info = detect_returning_players_snap_status(
                                weekly_stats, week, season, min_weeks_missed=2
                            )
                            # Update all player statuses with returning info
                            for info in all_returning_info:
                                for status in all_player_statuses:
                                    if (status.player_name.lower() == info['player_name'].lower() and
                                        status.team == info['team']):
                                        status.is_returning = True
                                        status.snap_expectation = info.get('expected_snap_pct', 0.7)
                                        status.games_since_return = info.get('games_since_return', 0)
                                        status.weeks_missed = info.get('weeks_missed', 0)
                                        break
                        except Exception as e:
                            logger.warning(f"Could not detect returning players: {e}")

                    # Pre-index player statuses by team for O(1) lookup
                    statuses_by_team = {}
                    for status in all_player_statuses:
                        if status.team not in statuses_by_team:
                            statuses_by_team[status.team] = []
                        statuses_by_team[status.team].append(status)

                    for team in unique_teams:
                        try:
                            # Get player statuses for this team (O(1) lookup)
                            team_statuses = statuses_by_team.get(team, [])

                            if not team_statuses:
                                continue

                            # Get returning players for this team
                            returning_players = [p for p in team_statuses if p.is_returning]

                            # Calculate synergy
                            synergy = calculate_team_synergy_adjustment(
                                team_statuses, team, returning_players
                            )

                            # Store for output
                            team_synergy_data[team] = {
                                'offense_multiplier': synergy.offense_multiplier,
                                'defense_multiplier': synergy.defense_multiplier,
                                'team_multiplier': synergy.team_multiplier,
                                'active_synergies': synergy.active_synergies,
                                'player_cascades': synergy.player_cascades,
                                'confidence': synergy.confidence,
                            }

                            # Log synergy info
                            if synergy.offense_multiplier != 1.0 or synergy.active_synergies:
                                logger.info(f"\n{team} Team Synergy:")
                                logger.info(f"   Offense multiplier: {synergy.offense_multiplier:.3f}x")
                                for syn in synergy.active_synergies[:3]:
                                    logger.info(f"   ✓ {syn['name']}: +{(syn['multiplier']-1)*100:.1f}%")
                                if synergy.player_cascades:
                                    logger.info(f"   Cascade effects: {len(synergy.player_cascades)} players affected")

                            # Apply synergy adjustments to predictions
                            team_mask = model_preds[team_col] == team

                            # Apply offense multiplier to volume stats
                            volume_cols = [
                                'receptions_mean', 'receiving_yards_mean', 'targets_mean',
                                'rushing_yards_mean', 'rushing_attempts_mean',
                                'passing_yards_mean', 'passing_attempts_mean', 'completions_mean'
                            ]

                            for col in volume_cols:
                                if col in model_preds.columns:
                                    model_preds.loc[team_mask, col] *= synergy.offense_multiplier

                            # Apply player-specific cascade effects
                            for player_name, effects in synergy.player_cascades.items():
                                player_mask = team_mask & (
                                    model_preds['player_name'].str.lower() == player_name.lower()
                                )

                                if not player_mask.any():
                                    continue

                                # Apply efficiency boost to catch rate (receptions relative to targets)
                                if 'efficiency_boost' in effects and effects['efficiency_boost'] > 1.0:
                                    if 'receptions_mean' in model_preds.columns:
                                        model_preds.loc[player_mask, 'receptions_mean'] *= effects['efficiency_boost']

                                # Apply coverage reduction to yards
                                if 'coverage_reduction' in effects:
                                    if 'receiving_yards_mean' in model_preds.columns:
                                        boost = 1.0 + effects['coverage_reduction'] * 0.5
                                        model_preds.loc[player_mask, 'receiving_yards_mean'] *= boost

                            # Add synergy columns to predictions
                            model_preds.loc[team_mask, 'synergy_multiplier'] = synergy.offense_multiplier
                            model_preds.loc[team_mask, 'has_synergy_bonus'] = 1 if synergy.active_synergies else 0

                        except Exception as e:
                            logger.warning(f"Could not calculate synergy for {team}: {e}")
                            continue

                    logger.info(f"\nApplied synergy adjustments for {len(team_synergy_data)} teams")
                else:
                    logger.warning("No team column found in predictions, skipping synergy")
            else:
                logger.warning("Roster data not available, skipping synergy calculations")

        except Exception as e:
            logger.warning(f"Could not apply team synergy adjustments: {e}")

        logger.info("=" * 60)

    # Filter to relevant markets - all supported prop types
    relevant_markets = [
        # Core yardage markets
        'player_receptions',
        'player_reception_yds',
        'player_rush_yds',
        'player_rush_attempts',
        'player_rush_reception_yds',
        # Passing markets
        'player_pass_yds',
        'player_pass_tds',
        'player_pass_attempts',
        'player_pass_completions',
        # TD markets
        'player_anytime_td',
        'player_1st_td',
        'player_last_td',
        # Longest play markets (to be implemented)
        'player_reception_longest',
        'player_rush_longest',
        'player_pass_longest_completion',
        # QB props
        'player_pass_interceptions',
    ]

    prop_lines = prop_lines[prop_lines['market'].isin(relevant_markets)]
    logger.info(f"Filtered to {len(prop_lines)} relevant prop lines")

    recommendations = []
    matched_count = 0
    unmatched_players = set()

    # Group prop lines by player and market
    for idx, prop_row in prop_lines.iterrows():
        prop_player = prop_row['player_name']
        market = prop_row['market']
        line = prop_row.get('line', None)
        odds = prop_row.get('odds', None)
        outcome = prop_row.get('outcome_type', 'Over')
        # Handle NaN outcome - default to 'Over' for O/U props
        if pd.isna(outcome) or outcome == '':
            outcome = 'Over'
        outcome = str(outcome)

        # Handle TD markets specially - they don't have a line (Yes/No bets)
        # TD props are binary: YES (player scores TD) vs NO (player doesn't score)
        # DK only provides YES odds, so we calculate both sides and pick the best edge
        td_markets = ['player_anytime_td', 'player_1st_td', 'player_last_td']
        td_best_side = None  # Will be set to 'Over' (YES) or 'Under' (NO)
        if market in td_markets:
            line = 0.5  # Set line to 0.5 for TD props (at least 1 TD)
            # Default to Over (Yes) - will be adjusted below after calculating model prob
            if 'Yes' in str(outcome):
                outcome = 'Over'  # "Yes" = Over 0.5 TDs
            elif 'No' in str(outcome):
                outcome = 'Under'  # "No" = Under 0.5 TDs

        if pd.isna(odds):
            continue

        # For non-TD markets, require a line
        if market not in td_markets and pd.isna(line):
            continue

        # Find matching model prediction
        matched_player = None
        for _, model_row in model_preds.iterrows():
            model_name = model_row['player_name']
            if match_player_name(model_name, prop_player):
                matched_player = model_row
                break

        if matched_player is None:
            unmatched_players.add(prop_player)
            continue

        matched_count += 1

        # Get model statistics for this market
        model_mean_raw, model_std = get_model_stat_for_market(matched_player, market)

        if model_mean_raw is None or model_mean_raw <= 0:
            continue

        # ============================================================================
        # V4 MODEL-GUIDED SIMULATION INTEGRATION
        # ============================================================================
        # When enabled, uses XGBoost features to adjust the MC simulation baseline
        # instead of running MC independently from the model
        model_guided_result = None
        adjustment_pct = 0.0
        adjustment_breakdown = {}

        if USE_MODEL_GUIDED_SIMULATION:
            # Get historical trailing stat as baseline
            hist_stats = get_player_historical_stats(
                prop_player, market,
                current_season=season,
                current_week=week
            )
            trailing_stat = hist_stats['hist_avg'] if hist_stats['hist_avg'] is not None else model_mean_raw

            # Extract XGBoost-relevant features from matched_player
            model_features = {
                # O-Line & Rush features
                'oline_health_score': matched_player.get('oline_health_score', 0),
                'opp_rush_def_vs_avg': matched_player.get('opp_rush_def_vs_avg', 0),
                # Receiving features
                'avg_separation': matched_player.get('avg_separation', 3.0),
                'avg_cushion': matched_player.get('avg_cushion', 6.0),
                'opp_pass_def_vs_avg': matched_player.get('opp_pass_def_vs_avg', 0),
                # Volume features
                'game_pace': matched_player.get('game_pace', 60.0),
                'vegas_total': matched_player.get('vegas_total', 45.0),
                'snap_share': matched_player.get('snap_share', 0.5),
                'target_share': matched_player.get('target_share', 0.15),
                # Defense EPA
                'opp_def_epa': matched_player.get('opp_def_epa', 0),
            }

            # Get XGBoost P(UNDER) if available from classifier
            xgb_p_under = matched_player.get('p_under', None)

            # Run model-guided simulation
            try:
                model_guided_result = simulate_with_model_guidance(
                    trailing_stat=trailing_stat,
                    trailing_std=model_std,
                    line=line,
                    market=market,
                    features=model_features,
                    xgb_p_under=xgb_p_under
                )

                # Use the model-guided projection instead of raw MC output
                model_mean_raw = model_guided_result['model_projection']
                model_std = model_guided_result['model_std']
                adjustment_pct = model_guided_result['adjustment_pct']
                adjustment_breakdown = model_guided_result.get('adjustment_breakdown', {})

                logger.debug(
                    f"MODEL-GUIDED: {prop_player} {market} | "
                    f"Trailing: {trailing_stat:.1f} -> Proj: {model_mean_raw:.1f} ({adjustment_pct:+.1f}%)"
                )
            except Exception as e:
                logger.warning(f"Model-guided simulation failed for {prop_player}: {e}")
                # Fall back to original behavior
                model_guided_result = None

        # TIERED CALIBRATION: Classify player based on snap_share, usage, AND market type
        # Updated 2025-11-26: Now uses market-specific bias adjustments
        tier_name, tier_config = classify_player_tier(matched_player, market)
        bias_adj = tier_config['bias_adjustment']  # Market-specific adjustment
        tier_shrinkage = tier_config['shrinkage_factor']
        confidence_boost = tier_config['confidence_boost']

        # Apply bias correction to model mean (fixes systematic over-prediction)
        # Correction factors derived from historical backtest analysis
        model_mean = apply_bias_correction(model_mean_raw, market)

        # Apply tier-specific bias adjustment (now market-specific from config)
        # Config has separate entries for yards vs count markets with appropriate adjustments
        model_mean = model_mean + bias_adj

        # P1 FIXES: Apply Vegas-based adjustments (total and spread)
        # These add predictive signal by adjusting for game pace and game script
        try:
            player_team = matched_player.get('team', '')
            player_opp = matched_player.get('opponent', '')
            player_week = matched_player.get('week', week)
            is_home = matched_player.get('is_home', True)  # Default to home

            # Map market to stat_type for P1 adjustment
            stat_type_map = {
                'player_pass_yds': 'passing_yards',
                'player_reception_yds': 'receiving_yards',
                'player_receptions': 'receptions',
                'player_rush_yds': 'rushing_yards',
                'player_rush_attempts': 'rushing_attempts',
                'player_targets': 'targets',
            }
            stat_type = stat_type_map.get(market, market)

            if player_team and player_opp:
                adjusted_mean, p1_breakdown = apply_p1_adjustments(
                    prediction=model_mean,
                    stat_type=stat_type,
                    team=player_team,
                    opponent=player_opp,
                    week=player_week,
                    season=season,
                    is_home=is_home
                )

                # Only apply if meaningful adjustment
                if p1_breakdown.get('total_adj', 1.0) != 1.0:
                    logger.debug(
                        f"P1 adjustment for {prop_player} {market}: "
                        f"{model_mean:.1f} -> {adjusted_mean:.1f} "
                        f"(vol={p1_breakdown.get('volume_adj', 1.0):.2f}, "
                        f"spread={p1_breakdown.get('spread_adj', 1.0):.2f})"
                    )
                    model_mean = adjusted_mean
        except Exception as e:
            logger.debug(f"P1 adjustment skipped for {prop_player}: {e}")

        # Also scale std proportionally to maintain coefficient of variation
        if model_std and model_std > 0 and model_mean_raw > 0:
            correction_factor = get_correction_factor(market)
            model_std = model_std * correction_factor

        # Get game expected TDs for 1st TD calculation
        game_id = matched_player.get('game_id', prop_row.get('game_id', ''))
        game_total_tds = get_game_expected_tds(game_id, game_totals) if market in td_markets else None

        # Calculate raw probabilities (using bias-corrected mean)
        prob_over_raw, prob_under_raw = calculate_over_under_prob(
            model_mean, model_std, line, market, game_total_tds=game_total_tds
        )

        # Apply tier-specific shrinkage calibration
        # High-volume players: less shrinkage (more confidence in model)
        # Low-volume players: more shrinkage (less confidence, more uncertainty)
        prob_over_calibrated = prob_over_raw * (1 - tier_shrinkage) + 0.5 * tier_shrinkage
        prob_under_calibrated = prob_under_raw * (1 - tier_shrinkage) + 0.5 * tier_shrinkage

        # Apply probability caps from config (defined at module level)
        max_prob = CONFIDENCE_CEILING  # From config.betting.performance_adjustment
        min_prob = CONFIDENCE_FLOOR    # From config.betting.performance_adjustment

        # Cap OVER probability at max (any probability above 56% gets capped)
        prob_over_capped = min(prob_over_calibrated, max_prob)
        # Cap UNDER probability at max
        prob_under_capped = min(prob_under_calibrated, max_prob)

        # Use capped probabilities going forward
        prob_over_calibrated = prob_over_capped
        prob_under_calibrated = prob_under_capped

        # Get market implied probability
        market_prob_with_vig = american_to_probability(odds)

        # For two-way markets, need to find opposite side to remove vig
        # Look for opposing line (Over vs Under on same player/market/line)
        opposite_odds = None
        opposite_outcome = 'Under' if 'Over' in outcome else 'Over'
        for _, opp_row in prop_lines.iterrows():
            if (opp_row['player_name'] == prop_player and
                opp_row['market'] == market and
                opp_row.get('line') == line and
                opposite_outcome in str(opp_row.get('outcome_type', ''))):
                opposite_odds = opp_row.get('odds')
                break

        # Remove vig if we have both sides
        if opposite_odds is not None:
            over_prob_vig = american_to_probability(odds if 'Over' in outcome else opposite_odds)
            under_prob_vig = american_to_probability(opposite_odds if 'Over' in outcome else odds)
            over_prob_novig, under_prob_novig = remove_vig_two_way(over_prob_vig, under_prob_vig)
            market_prob_novig = over_prob_novig if 'Over' in outcome else under_prob_novig
        else:
            # No opposite side found, use vig probability with warning
            market_prob_novig = market_prob_with_vig
            logger.debug(f"No opposite side found for {prop_player} {market}, using vig probability")

        # For TD markets: pick the side with positive edge
        # DK only provides YES odds, so we need to calculate both sides
        if market in td_markets:
            # YES probability from market (what DK odds imply for scoring TD)
            market_yes_prob = market_prob_novig if 'Over' in outcome else (1 - market_prob_novig)
            market_no_prob = 1 - market_yes_prob

            # Model probabilities
            model_yes_prob = prob_over_calibrated  # P(at least 1 TD)
            model_no_prob = prob_under_calibrated  # P(0 TDs)

            # Calculate edges for both sides
            yes_edge = (model_yes_prob - market_yes_prob) * 100
            no_edge = (model_no_prob - market_no_prob) * 100

            # Pick the side with the better (more positive) edge
            if no_edge > yes_edge and no_edge > 0:
                # Bet NO (player won't score)
                outcome = 'Under'
                td_best_side = 'NO'
                # Adjust odds for NO side: convert YES odds to NO odds
                # If YES is -135, NO is approximately +110 (with vig removed)
                # Use the no-vig probability to derive fair odds
                if market_no_prob > 0 and market_no_prob < 1:
                    # Convert probability to American odds
                    if market_no_prob >= 0.5:
                        odds = -int((market_no_prob / (1 - market_no_prob)) * 100)
                    else:
                        odds = int(((1 - market_no_prob) / market_no_prob) * 100)
                logger.debug(f"TD prop {prop_player}: YES edge={yes_edge:.1f}%, NO edge={no_edge:.1f}% -> betting NO")
            elif yes_edge > 0:
                td_best_side = 'YES'
                outcome = 'Over'
                logger.debug(f"TD prop {prop_player}: YES edge={yes_edge:.1f}%, NO edge={no_edge:.1f}% -> betting YES")
            else:
                # No positive edge on either side
                logger.debug(f"TD prop {prop_player}: no positive edge (YES={yes_edge:.1f}%, NO={no_edge:.1f}%)")

        # Determine model probability based on outcome type
        if 'Over' in outcome:
            model_prob = prob_over_calibrated
            raw_prob = prob_over_raw
        else:
            model_prob = prob_under_calibrated
            raw_prob = prob_under_raw

        # Calculate edge using no-vig market probability
        edge_pct = (model_prob - market_prob_novig) * 100
        roi_pct = edge_pct / market_prob_novig if market_prob_novig > 0 else 0

        # Assign confidence tier based on edge + probability
        # Apply tier-specific confidence boost (high-volume players get boost, low-volume get penalty)
        # The boost adjusts the effective edge used for tier classification
        adjusted_edge_for_confidence = edge_pct + (confidence_boost * 100)  # Convert decimal to pct points
        confidence_tier = assign_confidence_tier(adjusted_edge_for_confidence, model_prob, bet_type="player_prop")

        # Calculate Kelly sizing
        kelly_fraction = calculate_kelly_fraction(model_prob, odds, fractional=0.25)
        kelly_units = calculate_recommended_units(kelly_fraction, base_units=100.0)

        # Assign market priority based on position + market combination
        position = matched_player.get('position', '')
        market_priority = get_market_priority(position, market)

        # Determine game
        home_team = prop_row.get('home_team', '')
        away_team = prop_row.get('away_team', '')
        game = f"{away_team} @ {home_team}"
        commence_time = prop_row.get('commence_time', '')

        # Create recommendation - PRESERVE ALL MODEL FEATURES
        # Start with core betting recommendation fields
        # For TD props, display Yes/No instead of Over/Under
        if market in td_markets:
            pick_display = 'Yes' if 'Over' in outcome else 'No'
        else:
            pick_display = outcome
        rec = {
            'player': prop_player,
            'nflverse_name': matched_player.get('player_name', prop_player),
            'pick': pick_display,
            'market': market,
            'market_display': format_market_name(market),  # Human-readable: "Anytime TD"
            'line': line,
            'odds': odds,
            'model_prob': model_prob,  # Calibrated probability
            'model_prob_pct': f"{model_prob * 100:.1f}%",  # Display format
            'raw_prob': raw_prob,  # Raw probability before calibration
            'calibrated_prob': model_prob,  # Alias for clarity
            'market_prob': market_prob_novig,  # No-vig market probability
            'market_prob_with_vig': market_prob_with_vig,  # Original with vig
            'edge_pct': edge_pct,
            'roi_pct': roi_pct,
            'model_projection': model_mean,  # Actual model mean from predictions
            'model_std': model_std,
            'strategy': 'Model Projection (Calibrated)',
            'calibration_applied': True,
            'confidence': confidence_tier.value,  # ELITE/HIGH/STANDARD/LOW
            'confidence_pct': format_confidence_pct(confidence_tier.value),  # Display: "75-90%"
            'priority': market_priority,  # HIGH/MEDIUM/STANDARD
            'kelly_fraction': kelly_fraction,
            'kelly_units': kelly_units,
            'game': game,
            'commence_time': commence_time,
            'position': position,
            'team': matched_player.get('team', ''),
            'opponent': matched_player.get('opponent', ''),
            'week': matched_player.get('week', 0),
        }

        # ADD HISTORICAL STATS for multi-factor tier scoring
        hist_stats = get_player_historical_stats(
            prop_player, market,
            current_season=season,
            current_week=week
        )
        rec['hist_over_rate'] = hist_stats['hist_over_rate']
        rec['hist_avg'] = hist_stats['hist_avg']
        rec['hist_count'] = hist_stats['hist_count']

        # ADD trailing_stat: Use hist_avg (historical actual average for this market)
        # This is the ACTUAL 4-week trailing average, NOT the model projection
        # Falls back to model_projection if no historical data available
        rec['trailing_stat'] = hist_stats['hist_avg'] if hist_stats['hist_avg'] is not None else model_mean

        # ADD ALL MODEL PREDICTION FEATURES (54+ features preserved)
        # This ensures NO feature dropout between model predictions and final output

        # Basic player stats (mean/std for all markets)
        for stat in ['rushing_yards', 'rushing_attempts', 'rushing_tds',
                     'receiving_yards', 'receptions', 'targets', 'receiving_tds',
                     'passing_yards', 'passing_completions', 'passing_attempts', 'passing_tds']:
            rec[f'{stat}_mean'] = matched_player.get(f'{stat}_mean', None)
            rec[f'{stat}_std'] = matched_player.get(f'{stat}_std', None)

        # TD probabilities
        rec['raw_td_prob'] = matched_player.get('raw_td_prob', None)
        rec['calibrated_td_prob'] = matched_player.get('calibrated_td_prob', None)

        # Defensive matchup
        rec['opponent_def_epa'] = matched_player.get('opponent_def_epa_vs_position', None)

        # Opponent-specific features (V13 support)
        # Add divergence columns based on the current market
        for mkt, stat in [
            ('player_receptions', 'receptions'),
            ('player_reception_yds', 'receiving_yards'),
            ('player_rush_yds', 'rushing_yards'),
            ('player_pass_yds', 'passing_yards'),
        ]:
            div_col = f'{mkt}_vs_opp_divergence'
            conf_col = f'{mkt}_vs_opp_confidence'
            rec[div_col] = matched_player.get(div_col, 0)
            rec[conf_col] = matched_player.get(conf_col, 0)

        # Also add vs_opp_avg columns for debugging/display
        for stat in ['receptions', 'receiving_yards', 'rushing_yards', 'passing_yards', 'completions']:
            rec[f'vs_opp_avg_{stat}'] = matched_player.get(f'vs_opp_avg_{stat}', None)
            rec[f'vs_opp_games_{stat}'] = matched_player.get(f'vs_opp_games_{stat}', 0)

        # Injury context
        rec['total_adjustment'] = matched_player.get('total_adjustment', 0)
        rec['passing_adjustment'] = matched_player.get('passing_adjustment', 0)
        rec['severity'] = matched_player.get('severity', '')
        rec['injury_qb_status'] = matched_player.get('injury_qb_status', '')
        rec['injury_wr1_status'] = matched_player.get('injury_wr1_status', '')
        rec['injury_rb1_status'] = matched_player.get('injury_rb1_status', '')

        # Weather
        rec['weather_total_adjustment'] = matched_player.get('weather_total_adjustment', 0)
        rec['weather_passing_adjustment'] = matched_player.get('weather_passing_adjustment', 0)

        # Game context
        rec['is_divisional_game'] = matched_player.get('is_divisional_game', False)
        rec['rest_epa_adjustment'] = matched_player.get('rest_epa_adjustment', 0)
        rec['is_coming_off_bye'] = matched_player.get('is_coming_off_bye', False)
        rec['travel_epa_adjustment'] = matched_player.get('travel_epa_adjustment', 0)
        rec['home_field_advantage_points'] = matched_player.get('home_field_advantage_points', 0)
        rec['is_primetime_game'] = matched_player.get('is_primetime_game', False)
        rec['primetime_type'] = matched_player.get('primetime_type', '')

        # Redzone/snap shares
        rec['redzone_target_share'] = matched_player.get('redzone_target_share', None)
        rec['redzone_carry_share'] = matched_player.get('redzone_carry_share', None)
        rec['goalline_carry_share'] = matched_player.get('goalline_carry_share', None)
        rec['snap_share'] = matched_player.get('snap_share', None)

        # Tiered calibration info
        rec['calibration_tier'] = tier_name
        rec['tier_bias_adjustment'] = bias_adj
        rec['tier_shrinkage'] = tier_shrinkage
        rec['tier_confidence_boost'] = confidence_boost

        # V4 Model-Guided Simulation info
        rec['model_guided'] = model_guided_result is not None
        rec['model_adjustment_pct'] = adjustment_pct
        if model_guided_result:
            rec['baseline_projection'] = model_guided_result.get('baseline_projection', 0)
            # Build human-readable adjustment breakdown
            breakdown_parts = []
            for feat_name, feat_info in adjustment_breakdown.items():
                contrib = feat_info.get('contribution', 0)
                if abs(contrib) > 0.005:  # Only show significant adjustments (>0.5%)
                    breakdown_parts.append(f"{feat_name}: {contrib*100:+.1f}%")
            rec['adjustment_breakdown'] = '; '.join(breakdown_parts) if breakdown_parts else 'No significant adjustments'
        else:
            rec['baseline_projection'] = None
            rec['adjustment_breakdown'] = ''

        # Altitude
        rec['elevation_feet'] = matched_player.get('elevation_feet', 0)
        rec['is_high_altitude'] = matched_player.get('is_high_altitude', False)
        rec['altitude_epa_adjustment'] = matched_player.get('altitude_epa_adjustment', 0)
        rec['field_surface'] = matched_player.get('field_surface', '')

        # Team context
        rec['team_pass_attempts'] = matched_player.get('team_pass_attempts', None)
        rec['team_rush_attempts'] = matched_player.get('team_rush_attempts', None)
        rec['team_targets'] = matched_player.get('team_targets', None)
        rec['game_script_dynamic'] = matched_player.get('game_script_dynamic', '')

        # Market/role
        rec['market_blended_prob'] = matched_player.get('market_blended_prob', None)
        rec['role_override_applied'] = matched_player.get('role_override_applied', False)

        # Player identifiers (for joining/verification)
        rec['player_dk'] = matched_player.get('player_dk', '')
        rec['player_pbp'] = matched_player.get('player_pbp', '')

        # EXCLUSION FILTER: Skip backup players with insufficient data
        # These players have unreliable projections based on limited/no playing time
        snap_share = matched_player.get('snap_share', 0) or 0
        passing_markets = ['player_pass_yds', 'player_pass_completions', 'player_pass_attempts', 'player_pass_tds']

        # Skip backup QBs for passing props (they have no real data to project)
        if tier_name == 'tier_2_backup' and position == 'QB' and market in passing_markets:
            logger.debug(f"Skipping {prop_player} - backup QB with insufficient passing data")
            continue

        # Skip any player with extremely low snap share (<15%) - not enough data
        if snap_share < 0.15 and snap_share > 0:
            logger.debug(f"Skipping {prop_player} - snap_share {snap_share:.1%} too low for reliable projection")
            continue

        recommendations.append(rec)

    logger.info(f"Matched {matched_count} prop lines to model predictions")
    if unmatched_players:
        logger.warning(f"Could not match {len(unmatched_players)} players: {list(unmatched_players)[:10]}...")

    df = pd.DataFrame(recommendations)

    # CRITICAL FILTERS: Only include valid betting opportunities
    initial_count = len(df)

    # Filter 1: Positive edge only
    df = df[df['edge_pct'] > 0]
    logger.info(f"After positive edge filter: {len(df)} bets (removed {initial_count - len(df)})")

    # Filter 2: Model probability must be >= 50% (can't bet on <50% chance)
    # EXCEPTION: TD markets can recommend "No" when prob < 50% (we're betting the player WON'T score)
    td_markets = ['player_anytime_td', 'player_1st_td', 'player_last_td']
    non_td_mask = ~df['market'].isin(td_markets)
    td_mask = df['market'].isin(td_markets)

    # For non-TD markets: model_prob >= 50%
    # For TD markets: allow all (edge already calculated correctly)
    df = df[(non_td_mask & (df['model_prob'] >= 0.50)) | td_mask]
    logger.info(f"After prob >= 50% filter: {len(df)} bets")

    # Filter 3: Model probability should be <= 95% (extreme confidence often = overfit)
    df = df[df['model_prob'] <= 0.95]
    logger.info(f"After prob <= 95% filter: {len(df)} bets")

    # Filter 4: Edge should be realistic (<30% for standard markets, <50% for TD markets)
    # TD markets can have higher edges due to binary nature
    df = df[((~df['market'].isin(td_markets)) & (df['edge_pct'] < 30.0)) |
            ((df['market'].isin(td_markets)) & (df['edge_pct'] < 50.0))]
    logger.info(f"After edge filter: {len(df)} bets")

    # Filter 5: SNR-BASED MARKET FILTER (ADDED 2025-12-05)
    # Key insight: 0.5 units is 25% of std for receptions but only 1.5% for yards
    # This filters out low-SNR markets where random variance swamps any edge
    try:
        before_snr = len(df)
        snr_filter = SNRFilter()

        # Add SNR tier info to dataframe
        df['snr_tier'] = df['market'].apply(get_market_tier)
        df['snr_conf_threshold'] = df['market'].apply(get_confidence_threshold)

        # Apply SNR filter: must pass confidence and line deviation thresholds
        snr_passed = []
        snr_rejections = []

        for idx, row in df.iterrows():
            market = row['market']
            confidence = row.get('model_prob', row.get('confidence', 0.5))
            line = row.get('line', 0)
            trailing = row.get('trailing_stat', row.get('model_projection', line))

            decision = snr_filter.evaluate_bet(
                market=market,
                model_confidence=confidence,
                line=line,
                trailing_stat=trailing,
            )

            if decision.should_bet:
                snr_passed.append(idx)
            else:
                snr_rejections.append((idx, decision.rejection_reason))

        # Keep only SNR-passed bets
        df = df.loc[snr_passed]

        # Log summary
        logger.info(f"After SNR filter: {len(df)} bets (removed {before_snr - len(df)})")
        logger.info(f"  SNR Tiers: HIGH={len(df[df['snr_tier']=='HIGH'])}, MEDIUM={len(df[df['snr_tier']=='MEDIUM'])}, LOW={len(df[df['snr_tier']=='LOW'])}")

        # Log top rejection reasons
        if snr_rejections:
            from collections import Counter
            reasons = Counter(r[1].split(':')[0] if ':' in r[1] else r[1][:30] for r in snr_rejections)
            top_reasons = reasons.most_common(3)
            logger.info(f"  Top rejection reasons: {top_reasons}")
    except Exception as e:
        logger.warning(f"SNR filter failed (continuing without): {e}")

    # Filter 6: MARKET AGREEMENT FILTER (ADDED 2025-11-26)
    # When our model strongly disagrees with the market line, the MARKET is usually right.
    # Large perceived "edges" are actually model errors, not opportunities.
    # Only bet when model prediction is within 25% of the line.
    # EXCEPTION: TD markets - line is 0.5 (threshold), not a prediction target
    try:
        config = load_simulation_config()
        market_filter = config.get('market_agreement_filter', {})
        if market_filter.get('enabled', True):
            max_deviation = market_filter.get('max_deviation_pct', 0.40)  # LOOSENED from 0.25

            # Calculate deviation: |model_projection - line| / line
            df['model_market_deviation'] = abs(df['model_projection'] - df['line']) / df['line'].clip(lower=0.1)

            before_filter = len(df)
            # Exclude TD markets from this filter - their line (0.5) is a threshold, not comparable to expected TDs
            td_markets_filter = ['player_anytime_td', 'player_1st_td', 'player_last_td']
            df = df[(df['market'].isin(td_markets_filter)) | (df['model_market_deviation'] <= max_deviation)]

            logger.info(f"After market agreement filter (max {max_deviation:.0%} deviation): {len(df)} bets (removed {before_filter - len(df)})")
        else:
            logger.info("Market agreement filter disabled")
    except Exception as e:
        logger.warning(f"Could not apply market agreement filter: {e}")

    # Filter 6: PROFITABLE STRATEGY FILTER (ADDED 2025-11-26)
    # Backtest-validated strategy: UNDER on receptions/passing when model agrees with line
    # Results: 57.8% win rate, +10.3% ROI on 438 bets (Weeks 2-11)
    try:
        config = load_simulation_config()
        strategy = config.get('profitable_strategy', {})
        rules = strategy.get('strategy_rules', {})

        if rules:
            before_filter = len(df)

            # Get strategy parameters
            direction = rules.get('direction', 'UNDER_ONLY')
            allowed_markets = rules.get('allowed_markets', ['player_receptions', 'player_pass_yds'])
            max_agreement = rules.get('model_line_agreement_max', 0.12)

            # Calculate model-line deviation
            if 'model_market_deviation' not in df.columns:
                df['model_market_deviation'] = abs(df['model_projection'] - df['line']) / df['line'].clip(lower=0.1)

            # Apply strategy filter
            if direction == 'UNDER_ONLY':
                # Only keep UNDER bets on allowed markets with model agreement
                strategy_filter = (
                    (df['pick'].str.contains('Under', case=False, na=False)) &
                    (df['market'].isin(allowed_markets)) &
                    (df['model_market_deviation'] <= max_agreement)
                )

                # Mark bets that pass the profitable strategy filter
                df['profitable_strategy'] = strategy_filter
                df['strategy_validated'] = strategy_filter

                # Keep ALL bets but flag which ones are strategy-validated
                # This allows user to see both validated and non-validated bets
                strategy_bets = df[strategy_filter]
                other_bets = df[~strategy_filter]

                logger.info(f"PROFITABLE STRATEGY: {len(strategy_bets)} bets pass validation")
                logger.info(f"  - Direction: {direction}")
                logger.info(f"  - Markets: {allowed_markets}")
                logger.info(f"  - Model-line agreement: within {max_agreement:.0%}")
                logger.info(f"  - Other bets (not strategy-validated): {len(other_bets)}")
            else:
                df['profitable_strategy'] = False
                df['strategy_validated'] = False
                logger.info("Profitable strategy filter: direction not UNDER_ONLY, skipping")
        else:
            df['profitable_strategy'] = False
            df['strategy_validated'] = False
            logger.info("No profitable strategy rules found in config")
    except Exception as e:
        df['profitable_strategy'] = False
        df['strategy_validated'] = False
        logger.warning(f"Could not apply profitable strategy filter: {e}")

    # Filter 7: V7 HYBRID APPROACH - DISABLED 2025-12-01 (replaced by V14)
    # V14 defense-aware model now handles all markets with LVT + defense EPA
    # Keeping code for reference but skipping execution
    V7_ENABLED = False  # Set to True to re-enable V7
    if V7_ENABLED:
      try:
        # Calculate core feature: line vs trailing average
        df['line_vs_trailing'] = df['line'] - df['model_projection']
        df['line_vs_trailing_pct'] = df['line_vs_trailing'] / (df['model_projection'] + 0.1)

        # V7 SIMPLE RULE: Bet UNDER when line > trailing + threshold
        # Thresholds based on backtest analysis (2023-2025):
        v7_thresholds = {
            'player_receptions': 1.5,      # 72.4% UNDER, +38.2% ROI in 2025
            'player_rush_yds': 1.5,        # 54.3% UNDER, +3.7% ROI (consistent)
            'player_pass_yds': 1.5,        # 55.3% UNDER, +5.5% ROI (borderline)
            'player_reception_yds': None,  # NO EDGE - skip this market
        }

        # V7 scoring using simple rule + regime estimation
        v7_scores = []
        for idx, row in df.iterrows():
            market = row['market']
            line_vs_trailing = row['line_vs_trailing']

            # Get market-specific threshold
            threshold = v7_thresholds.get(market)

            if threshold is None:
                # Market has no edge - don't recommend
                v7_scores.append({
                    'rule_signal': 'NO_EDGE',
                    'p_under': 0.50,
                    'p_over': 0.50,
                    'best_direction': None,
                    'best_ev': 0,
                    'kelly_size': 0,
                    'passes': False,
                    'reason': f"{market} has no statistical edge"
                })
                continue

            # Simple rule: line > trailing + threshold = bet UNDER
            if line_vs_trailing > threshold:
                # Strong UNDER signal
                # NOTE: Use conservative default until walk-forward validation is re-run
                # Market-specific rates will be computed from validated backtest results
                p_under = LOW_CONFIDENCE_THRESHOLD  # From config, conservative until validated

                # Scale probability based on distance above threshold
                # More distance = higher confidence
                excess = line_vs_trailing - threshold
                prob_boost = min(0.10, excess * 0.02)  # +2% per point, max +10%
                p_under = min(0.85, p_under + prob_boost)

                p_over = 1 - p_under
                ev_under = calculate_bet_ev(p_under, -110)
                ev_over = calculate_bet_ev(p_over, -110)
                kelly = calculate_kelly_fraction(p_under, -110)

                v7_scores.append({
                    'rule_signal': 'UNDER',
                    'p_under': p_under,
                    'p_over': p_over,
                    'best_direction': 'UNDER',
                    'best_ev': ev_under * 100,
                    'kelly_size': kelly,
                    'passes': True,
                    'reason': f"line > trailing + {threshold} ({line_vs_trailing:+.1f})"
                })
            elif line_vs_trailing < -threshold:
                # Strong OVER signal (line below trailing)
                # Note: OVER signals are less reliable historically
                p_over = CONFIDENCE_FLOOR + 0.02  # Conservative until validated
                p_under = 1 - p_over
                ev_over = calculate_bet_ev(p_over, -110)
                ev_under = calculate_bet_ev(p_under, -110)

                # Only recommend if EV positive
                if ev_over > 0:
                    kelly = calculate_kelly_fraction(p_over, -110)
                    v7_scores.append({
                        'rule_signal': 'OVER',
                        'p_under': p_under,
                        'p_over': p_over,
                        'best_direction': 'OVER',
                        'best_ev': ev_over * 100,
                        'kelly_size': kelly,
                        'passes': True,
                        'reason': f"line < trailing - {threshold} ({line_vs_trailing:+.1f})"
                    })
                else:
                    v7_scores.append({
                        'rule_signal': 'OVER_WEAK',
                        'p_under': p_under,
                        'p_over': p_over,
                        'best_direction': None,
                        'best_ev': 0,
                        'kelly_size': 0,
                        'passes': False,
                        'reason': "OVER signal too weak"
                    })
            else:
                # No signal - line is close to trailing
                v7_scores.append({
                    'rule_signal': 'NO_SIGNAL',
                    'p_under': 0.50,
                    'p_over': 0.50,
                    'best_direction': None,
                    'best_ev': 0,
                    'kelly_size': 0,
                    'passes': False,
                    'reason': f"line within +/- {threshold} of trailing"
                })

        # Add V7 columns to dataframe
        df['v7_rule_signal'] = [s['rule_signal'] for s in v7_scores]
        df['v7_p_under'] = [s['p_under'] for s in v7_scores]
        df['v7_p_over'] = [s['p_over'] for s in v7_scores]
        df['v7_best_direction'] = [s['best_direction'] for s in v7_scores]
        df['v7_best_ev'] = [s['best_ev'] for s in v7_scores]
        df['v7_kelly_size'] = [s['kelly_size'] for s in v7_scores]
        df['v7_validated'] = [s['passes'] for s in v7_scores]
        df['v7_reason'] = [s['reason'] for s in v7_scores]

        # Legacy columns for backwards compatibility (V5/V6 aliases)
        df['v6_p_under'] = df['v7_p_under']
        df['v6_p_over'] = df['v7_p_over']
        df['v6_best_direction'] = df['v7_best_direction']
        df['v6_best_ev'] = df['v7_best_ev']
        df['v6_kelly_size'] = df['v7_kelly_size']
        df['v6_value_validated'] = df['v7_validated']
        df['v5_p_under'] = df['v7_p_under']
        df['v5_threshold'] = LOW_CONFIDENCE_THRESHOLD
        df['v5_edge_validated'] = df['v7_validated']

        # Count validated bets
        v7_validated = df[df['v7_validated'] == True]
        logger.info(f"V7 HYBRID APPROACH: {len(v7_validated)} bets validated by simple rule")

        # Show breakdown
        if len(v7_validated) > 0:
            under_bets = v7_validated[v7_validated['v7_best_direction'] == 'UNDER']
            over_bets = v7_validated[v7_validated['v7_best_direction'] == 'OVER']
            logger.info(f"  - UNDER bets: {len(under_bets)}")
            logger.info(f"  - OVER bets: {len(over_bets)}")

            for market in v7_validated['market'].unique():
                market_df = v7_validated[v7_validated['market'] == market]
                avg_ev = market_df['v7_best_ev'].mean()
                logger.info(f"  - {market}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")

      except Exception as e:
        logger.warning(f"V7 hybrid approach failed: {e}")
        df['v7_rule_signal'] = 'ERROR'
        df['v7_p_under'] = CONFIDENCE_FLOOR
        df['v7_p_over'] = CONFIDENCE_FLOOR
        df['v7_best_direction'] = None
        df['v7_best_ev'] = 0
        df['v7_kelly_size'] = 0
        df['v7_validated'] = False
        df['v7_reason'] = str(e)
        df['v6_p_under'] = None
        df['v6_p_over'] = None
        df['v6_best_direction'] = None
        df['v6_best_ev'] = None
        df['v6_kelly_size'] = 0
        df['v6_value_validated'] = False
        df['v5_p_under'] = None
        df['v5_threshold'] = None
        df['v5_edge_validated'] = False
    else:
        # V7 DISABLED - initialize columns with defaults
        logger.info("V7 DISABLED - using V14 defense-aware model only")
        # FIX (Dec 7, 2025): Use trailing_stat instead of model_projection for LVT display
        trailing_col = df['trailing_stat'].fillna(df['model_projection'])
        df['line_vs_trailing'] = df['line'] - trailing_col
        df['line_vs_trailing_pct'] = df['line_vs_trailing'] / (trailing_col + 0.1)
        df['v7_rule_signal'] = 'DISABLED'
        df['v7_p_under'] = CONFIDENCE_FLOOR
        df['v7_p_over'] = CONFIDENCE_FLOOR
        df['v7_best_direction'] = None
        df['v7_best_ev'] = 0
        df['v7_kelly_size'] = 0
        df['v7_validated'] = False
        df['v7_reason'] = 'V7 disabled - using V14'
        df['v6_p_under'] = None
        df['v6_p_over'] = None
        df['v6_best_direction'] = None
        df['v6_best_ev'] = None
        df['v6_kelly_size'] = 0
        df['v6_value_validated'] = False
        df['v5_p_under'] = None
        df['v5_threshold'] = None
        df['v5_edge_validated'] = False

    # Filter 8: V17 ACTIVE MODEL
    # V16 uses LVT as hub with interaction constraints and monotonic constraints
    try:
        models, thresholds, _ = load_active_model()

        if models:
            model_results = []
            for idx, row in df.iterrows():
                market = row['market']
                row_dict = row.to_dict()
                model_result = apply_active_model(row_dict, market)
                model_results.append(model_result)

            # Add model columns
            df['model_p_under'] = [r['model_p_under'] for r in model_results]
            df['model_best_direction'] = [r['model_best_direction'] for r in model_results]
            df['model_best_ev'] = [r['model_best_ev'] for r in model_results]
            df['model_validated'] = [r['model_validated'] for r in model_results]
            df['model_reason'] = [r['model_reason'] for r in model_results]

            # Add SHAP-style feature contributions for explainability
            df['feature_contributions'] = [r.get('feature_contributions', []) for r in model_results]

            # ========================================
            # MODEL-SIMULATOR BRIDGE INTEGRATION
            # Resolves conflicts between MC projections and XGBoost predictions
            # ========================================
            # Market to (mean_col, std_col) mapping
            mc_col_map = {
                'player_receptions': ('receptions_mean', 'receptions_std'),
                'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
                'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
                'player_rush_attempts': ('rushing_attempts_mean', 'rushing_attempts_std'),
                'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
            }

            ensemble_results = []
            conflicts_found = 0
            for idx, row in df.iterrows():
                market = row['market']
                line = row.get('line', 0) or 0
                model_p_under = row.get('model_p_under', 0.5) or 0.5

                # Get MC projection for this market
                mc_cols = mc_col_map.get(market)
                if mc_cols and mc_cols[0] in df.columns:
                    mc_mean = row.get(mc_cols[0], line) or line
                    mc_std = row.get(mc_cols[1], 0) or 0
                    if pd.isna(mc_mean):
                        mc_mean = line
                    if pd.isna(mc_std):
                        mc_std = 0
                else:
                    # Use model_projection as fallback
                    mc_mean = row.get('model_projection', line) or line
                    mc_std = row.get('model_std', 0) or 0

                # Call the bridge to get unified prediction
                unified = get_unified_prediction(
                    market=market,
                    line=line,
                    mc_mean=mc_mean,
                    mc_std=mc_std if mc_std > 0 else abs(mc_mean - line) * 0.3,  # Estimate std if missing
                    model_p_under=model_p_under
                )
                ensemble_results.append(unified)

                if unified.get('had_conflict', False):
                    conflicts_found += 1

            # Add ensemble columns
            df['ensemble_p_under'] = [r['ensemble_p_under'] for r in ensemble_results]
            df['ensemble_projection'] = [r['ensemble_projection'] for r in ensemble_results]
            df['ensemble_direction'] = [r['ensemble_direction'] for r in ensemble_results]
            df['mc_p_under'] = [r['mc_p_under'] for r in ensemble_results]
            df['mc_direction'] = [r['mc_direction'] for r in ensemble_results]
            df['model_weight'] = [r['model_weight'] for r in ensemble_results]
            df['had_conflict'] = [r['had_conflict'] for r in ensemble_results]
            df['prediction_source'] = [r['source'] for r in ensemble_results]

            # Log bridge results
            logger.info(f"MODEL-SIMULATOR BRIDGE: {conflicts_found}/{len(df)} bets had MC/Model conflict")
            if conflicts_found > 0:
                logger.info(f"  Conflicts resolved using ensemble weighting (model weight: 70% when confident)")

            # Log V17 results
            model_validated_df = df[df['model_validated'] == True]
            logger.info(f"V17 ACTIVE MODEL: {len(model_validated_df)} bets validated")
            for market in model_validated_df['market'].unique():
                market_df = model_validated_df[model_validated_df['market'] == market]
                avg_ev = market_df['model_best_ev'].mean()
                logger.info(f"  - {market}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")

            # Filter 8a: For model-supported markets, ONLY keep model-validated bets
            # This ensures we don't recommend MC-driven bets when model exists
            # Uses CLASSIFIER_MARKETS (trained markets only, excludes TD props)
            model_supported_markets = CLASSIFIER_MARKETS
            before_model_filter = len(df)

            # Keep bets that are either:
            # 1. Model-validated (model_validated=True), OR
            # 2. NOT in a model-supported market (use MC for other markets)
            df = df[
                (df['model_validated'] == True) |
                (~df['market'].isin(model_supported_markets))
            ]

            model_filtered = before_model_filter - len(df)
            if model_filtered > 0:
                logger.info(f"{MODEL_VERSION_FULL} MODEL FILTER: Removed {model_filtered} bets ({MODEL_VERSION_FULL} market but not validated)")
                logger.info(f"  Remaining: {len(df)} bets ({MODEL_VERSION_FULL}-validated + non-model markets)")

            # GLOBAL DIRECTION CONSTRAINT FILTER (Dec 25, 2025)
            # Applies to ALL markets, not just CLASSIFIER_MARKETS
            # This ensures TD props with OVER_ONLY constraint don't show NO picks
            before_direction_filter = len(df)
            direction_violations = []

            def check_direction_constraint(row):
                """Return False if pick violates direction constraint."""
                market = row.get('market', '')
                constraint = MARKET_DIRECTION_CONSTRAINTS.get(market)
                if not constraint or constraint == 'BOTH':
                    return True  # No constraint or allows both

                pick = str(row.get('pick', '')).upper()
                # Map YES/NO to OVER/UNDER for TD markets
                if pick == 'YES':
                    pick = 'OVER'
                elif pick == 'NO':
                    pick = 'UNDER'

                if constraint == 'UNDER_ONLY' and pick == 'OVER':
                    direction_violations.append(f"{row.get('player')}: {market} OVER blocked (UNDER_ONLY)")
                    return False
                if constraint == 'OVER_ONLY' and pick == 'UNDER':
                    direction_violations.append(f"{row.get('player')}: {market} UNDER blocked (OVER_ONLY)")
                    return False
                return True

            df = df[df.apply(check_direction_constraint, axis=1)]
            direction_filtered = before_direction_filter - len(df)
            if direction_filtered > 0:
                logger.info(f"DIRECTION CONSTRAINT FILTER: Removed {direction_filtered} bets (violated OVER_ONLY/UNDER_ONLY)")
                for v in direction_violations[:5]:  # Log first 5
                    logger.debug(f"  {v}")

            # UPDATE PICK TO USE MODEL DIRECTION (V2 - Classifier as Source of Truth)
            # For model-validated bets:
            # 1. Use V23 classifier direction directly (from P(UNDER))
            # 2. Use ensemble_projection (derived from P(UNDER) via inverse CDF)
            # This GUARANTEES projections always align with pick direction
            model_validated_mask = df['model_validated'] == True
            if model_validated_mask.any():
                df.loc[model_validated_mask, 'pick'] = df.loc[model_validated_mask, 'model_best_direction']
                logger.info(f"MODEL PICK: Updated {model_validated_mask.sum()} picks to use V23 classifier direction")

                # V4 FIX: Use ensemble_projection (derived from model P(UNDER)) as model_projection
                # USER REQUIREMENT: "we need to use the model; not trailing average as truth"
                # The XGBoost classifier P(UNDER) is the source of truth for both:
                #   1. Direction (OVER/UNDER) - from P(UNDER) > 0.5
                #   2. Projection - derived via inverse normal CDF from P(UNDER)
                # This GUARANTEES projection always aligns with pick direction:
                #   - If P(UNDER) = 75%, projection will be BELOW the line
                #   - If P(UNDER) = 25%, projection will be ABOVE the line
                df.loc[model_validated_mask, 'model_projection'] = df.loc[model_validated_mask, 'ensemble_projection']
                logger.info(f"PROJECTION V4: Updated {model_validated_mask.sum()} projections to use model-derived values (from P(UNDER) via inverse CDF)")
        else:
            df['model_p_under'] = CONFIDENCE_FLOOR
            df['model_best_direction'] = None
            df['model_best_ev'] = 0
            df['model_validated'] = False
            df['model_reason'] = 'V16 model not loaded'

    except Exception as e:
        logger.warning(f"V16 integration failed: {e}")
        df['model_p_under'] = 0.50
        df['model_best_direction'] = None
        df['model_best_ev'] = 0
        df['model_validated'] = False
        df['model_reason'] = str(e)

    # Filter 8b: LEGACY V14 - DISABLED (replaced by active model)
    # The active model now handles all markets with defense EPA integrated
    V14_ENABLED = False  # Set to True to re-enable legacy V14
    if V14_ENABLED:
      try:
        v14_model, v14_coefs, v14_thresholds = None, {}, {}

        if v14_model:
            # Get BOTH rush and pass defense EPA data for this week
            from pathlib import Path
            pbp_path = project_root / 'data' / 'nflverse' / f'pbp_2025.parquet'

            rush_def_epa_lookup = {}
            pass_def_epa_lookup = {}
            if pbp_path.exists():
                pbp = pd.read_parquet(pbp_path)

                # Calculate trailing RUSH defense EPA per team
                rush_def = pbp[pbp['play_type'] == 'run'].groupby(['defteam', 'week']).agg(
                    rush_def_epa=('epa', 'mean')
                ).reset_index()
                rush_def = rush_def.sort_values(['defteam', 'week'])
                rush_def['trailing_def_epa'] = rush_def.groupby('defteam')['rush_def_epa'].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
                for team in rush_def['defteam'].unique():
                    team_data = rush_def[rush_def['defteam'] == team]
                    if len(team_data) > 0:
                        rush_def_epa_lookup[team] = team_data['trailing_def_epa'].iloc[-1]

                # Calculate trailing PASS defense EPA per team
                pass_def = pbp[pbp['play_type'] == 'pass'].groupby(['defteam', 'week']).agg(
                    pass_def_epa=('epa', 'mean')
                ).reset_index()
                pass_def = pass_def.sort_values(['defteam', 'week'])
                pass_def['trailing_def_epa'] = pass_def.groupby('defteam')['pass_def_epa'].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
                for team in pass_def['defteam'].unique():
                    team_data = pass_def[pass_def['defteam'] == team]
                    if len(team_data) > 0:
                        pass_def_epa_lookup[team] = team_data['trailing_def_epa'].iloc[-1]

            # Apply V14 to ALL markets (using appropriate defense EPA type)
            v14_results = []
            v14_markets = ['player_rush_yds', 'player_pass_yds', 'player_receptions', 'player_reception_yds']
            for idx, row in df.iterrows():
                market = row['market']

                if market not in v14_markets:
                    v14_results.append({
                        'v14_p_under': None,
                        'v14_best_direction': None,
                        'v14_best_ev': None,
                        'v14_validated': False,
                        'v14_reason': f'V14 not applicable for {market}',
                        'trailing_def_epa': 0.0
                    })
                    continue

                row_dict = row.to_dict()
                opponent = row.get('opponent', row.get('opponent_team', ''))

                # Use PASS defense EPA for passing/receiving markets, RUSH for rushing
                if market == 'player_rush_yds':
                    trailing_def_epa = rush_def_epa_lookup.get(opponent, 0.0)
                else:
                    # pass_yds, receptions, reception_yds all use pass defense
                    trailing_def_epa = pass_def_epa_lookup.get(opponent, 0.0)

                v14_result = apply_v14_model(row_dict, trailing_def_epa)
                v14_result['trailing_def_epa'] = trailing_def_epa  # Store for reasoning
                v14_results.append(v14_result)

            # Add V14 columns
            df['trailing_def_epa'] = [r.get('trailing_def_epa', 0.0) for r in v14_results]
            df['v14_p_under'] = [r.get('v14_p_under') for r in v14_results]
            df['v14_best_direction'] = [r.get('v14_best_direction') for r in v14_results]
            df['v14_best_ev'] = [r.get('v14_best_ev') for r in v14_results]
            df['v14_validated'] = [r.get('v14_validated', False) for r in v14_results]
            df['v14_reason'] = [r.get('v14_reason', '') for r in v14_results]

            # For ALL V14 markets, override model results with V14
            v14_validated_mask = (df['v14_validated'] == True)

            # Update model columns with V14 results for all validated markets
            df.loc[v14_validated_mask, 'model_p_under'] = df.loc[v14_validated_mask, 'v14_p_under']
            df.loc[v14_validated_mask, 'model_best_direction'] = df.loc[v14_validated_mask, 'v14_best_direction']
            df.loc[v14_validated_mask, 'model_best_ev'] = df.loc[v14_validated_mask, 'v14_best_ev']
            df.loc[v14_validated_mask, 'model_validated'] = True
            df.loc[v14_validated_mask, 'model_reason'] = df.loc[v14_validated_mask, 'v14_reason']

            # Log V14 results by market
            v14_all_validated = df[df['v14_validated'] == True]
            if len(v14_all_validated) > 0:
                logger.info(f"V14 DEFENSE-AWARE MODEL: {len(v14_all_validated)} total bets validated")
                for mkt in v14_markets:
                    mkt_validated = v14_all_validated[v14_all_validated['market'] == mkt]
                    if len(mkt_validated) > 0:
                        avg_ev = mkt_validated['v14_best_ev'].mean()
                        mkt_display = mkt.replace('player_', '').replace('_', ' ').title()
                        logger.info(f"   {mkt_display}: {len(mkt_validated)} bets (avg EV: {avg_ev:+.1f}%)")

      except Exception as e:
        logger.warning(f"V14 integration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    else:
        # V14 DISABLED - log and continue to active model
        logger.info("V14 DISABLED - using active model only")

    # Filter 8c: ACTIVE MODEL - Single source of truth
    # Uses ALL FeatureEngine methods: target_share, EPA, snap_share, etc.
    # Walk-forward validated with positive ROI across markets
    try:
        models, thresholds, _ = load_active_model()

        if models:
            supported_markets = list(models.keys())

            # Apply model to all supported markets
            model_results = []
            for idx, row in df.iterrows():
                market = row['market']

                if market in supported_markets:
                    # Build comprehensive feature dict using FeatureEngine
                    row_dict = row.to_dict()

                    # Get season and week from the data
                    current_season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
                    current_week = week  # From generate_recommendations parameter

                    # Extract ALL features using FeatureEngine (matches training)
                    extracted_features = extract_prediction_features(
                        row_dict, market, current_season, current_week
                    )

                    # Merge extracted features into row_dict
                    row_dict.update(extracted_features)

                    # V17: Use apply_active_model which handles V17 Booster format
                    model_result = apply_active_model(row_dict, market)
                    model_results.append(model_result)
                else:
                    # Market not supported - use Monte Carlo probability
                    mc_p_under = row.get('calibrated_prob', 0.5)
                    if 'over' in str(row.get('pick', '')).lower():
                        mc_p_under = 1 - mc_p_under
                    model_results.append({
                        'model_p_under': mc_p_under,
                        'model_best_direction': None,
                        'model_best_ev': 0,
                        'model_validated': False,
                        'model_reason': f'Monte Carlo fallback ({mc_p_under:.1%})'
                    })

            # Add model columns
            df['model_p_under'] = [r.get('model_p_under') for r in model_results]
            df['model_best_direction'] = [r.get('model_best_direction') for r in model_results]
            df['model_best_ev'] = [r.get('model_best_ev') for r in model_results]
            df['model_validated'] = [r.get('model_validated', False) for r in model_results]
            df['model_reason'] = [r.get('model_reason', '') for r in model_results]

            # Log model results
            validated = df[df['model_validated'] == True]
            if len(validated) > 0:
                logger.info(f"ACTIVE MODEL: {len(validated)} bets validated")
                for mkt in supported_markets:
                    mkt_validated = validated[validated['market'] == mkt]
                    if len(mkt_validated) > 0:
                        avg_ev = mkt_validated['model_best_ev'].mean()
                        roi = thresholds.get(mkt, {}).get('roi', 0)
                        logger.info(f"   {mkt}: {len(mkt_validated)} bets (EV: {avg_ev:+.1f}%, trained ROI: {roi:+.1f}%)")
        else:
            # Model not available - initialize columns with defaults
            df['model_p_under'] = None
            df['model_best_direction'] = None
            df['model_best_ev'] = 0
            df['model_validated'] = False
            df['model_reason'] = 'Model not loaded'

    except Exception as e:
        logger.warning(f"Model integration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        df['model_p_under'] = None
        df['model_validated'] = False
        df['model_reason'] = str(e)

    # Filter 9: OPPONENT CONFLICT WARNING (ADDED 2025-11-27)
    # Flag picks where opponent history contradicts model signal
    # Example: Lamar Jackson UNDER but he averages +25% vs CIN
    try:
        conflict_flags = []
        conflict_reasons = []
        divergence_threshold = 0.15  # 15% divergence to flag

        for idx, row in df.iterrows():
            market = row.get('market', '')
            pick = str(row.get('pick', '')).lower()

            # Check for opponent divergence columns from predictions
            divergence_col = f'{market}_vs_opp_divergence'
            confidence_col = f'{market}_vs_opp_confidence'

            divergence = row.get(divergence_col, row.get('vs_opponent_divergence', 0))
            confidence = row.get(confidence_col, row.get('vs_opponent_confidence', 0))

            # Need minimum confidence (2+ prior games)
            if confidence < 0.33:  # Less than 2 games
                conflict_flags.append(False)
                conflict_reasons.append('')
                continue

            is_under = 'under' in pick
            is_over = 'over' in pick

            # Conflict: Model says UNDER but player overperforms vs opponent
            if is_under and divergence > divergence_threshold:
                conflict_flags.append(True)
                conflict_reasons.append(f"⚠️ Player +{divergence*100:.0f}% vs opponent")
            # Conflict: Model says OVER but player underperforms vs opponent
            elif is_over and divergence < -divergence_threshold:
                conflict_flags.append(True)
                conflict_reasons.append(f"⚠️ Player {divergence*100:.0f}% vs opponent")
            else:
                conflict_flags.append(False)
                conflict_reasons.append('')

        df['opponent_conflict'] = conflict_flags
        df['opponent_conflict_reason'] = conflict_reasons

        # Log conflicts
        conflicts = df[df['opponent_conflict'] == True]
        if len(conflicts) > 0:
            logger.warning(f"⚠️ OPPONENT CONFLICTS: {len(conflicts)} picks have conflicting opponent history")
            for _, conflict_row in conflicts.head(5).iterrows():
                logger.warning(f"   {conflict_row['player']} {conflict_row['market']}: {conflict_row['opponent_conflict_reason']}")

    except Exception as e:
        logger.warning(f"Opponent conflict check failed: {e}")
        df['opponent_conflict'] = False
        df['opponent_conflict_reason'] = ''

    # Filter 10: LINE MOVEMENT INTEGRATION (ADDED 2025-11-26)
    # When line moves DOWN = sharp money on UNDER (follow this signal!)
    # When line moves UP = sharp money on OVER
    try:
        line_movement = load_line_movement(week)
        if line_movement:
            sharp_under_list = []
            sharp_over_list = []
            line_move_list = []

            for idx, row in df.iterrows():
                player_key = (str(row['player']).lower().strip(), row['market'])
                movement = line_movement.get(player_key, {})

                sharp_under_list.append(movement.get('sharp_under', False))
                sharp_over_list.append(movement.get('sharp_over', False))
                line_move_list.append(movement.get('line_movement', 0))

            df['sharp_under'] = sharp_under_list
            df['sharp_over'] = sharp_over_list
            df['line_movement'] = line_move_list

            # Count bets with sharp money confirmation
            sharp_confirmed = df[
                (df['v5_edge_validated'] == True) &
                (df['pick'].str.contains('Under', case=False, na=False)) &
                (df['sharp_under'] == True)
            ]

            if len(sharp_confirmed) > 0:
                logger.info(f"SHARP MONEY CONFIRMATION: {len(sharp_confirmed)} UNDER bets have line movement support")
                df['sharp_confirmed'] = (
                    (df['v5_edge_validated'] == True) &
                    (df['pick'].str.contains('Under', case=False, na=False)) &
                    (df['sharp_under'] == True)
                )
            else:
                df['sharp_confirmed'] = False
        else:
            df['sharp_under'] = False
            df['sharp_over'] = False
            df['line_movement'] = 0
            df['sharp_confirmed'] = False
    except Exception as e:
        df['sharp_under'] = False
        df['sharp_over'] = False
        df['line_movement'] = 0
        df['sharp_confirmed'] = False
        logger.warning(f"Could not apply line movement: {e}")

    # Filter 10: WALK-FORWARD VALIDATED FILTER (ADDED 2025-12-07)
    # Based on walk-forward recalibration analysis:
    # - High confidence (>60%) = +4.5% ROI
    # - Receptions market = best win rate (51.4%)
    # - OVER bets = -12.6% ROI (disabled)
    # - Lines 0-3 receptions = 81.2% win rate, +55.1% ROI
    try:
        before_wf_filter = len(df)

        # Choose filter profile (ELITE for lines 0-3, CONSERVATIVE for all)
        # Using CONSERVATIVE by default - receptions market, >60% confidence, UNDER only
        active_filter = CONSERVATIVE_FILTER

        wf_passed = []
        wf_rejections = []

        for idx, row in df.iterrows():
            market = row.get('market', '')
            line = row.get('line', 0)
            # Use model_prob for confidence (calibrated probability)
            confidence = row.get('model_prob', 0.5)
            pick = str(row.get('pick', '')).upper()
            side = 'OVER' if 'OVER' in pick else 'UNDER'

            # Calculate player volatility from trailing stats if available
            player_cv = None
            trailing_stat = row.get('trailing_stat', 0)
            model_std = row.get('model_std', 0)
            if trailing_stat and trailing_stat > 0 and model_std:
                player_cv = model_std / trailing_stat

            # Get game total for game context filter
            game_total = row.get('vegas_total', None)

            # Apply filter
            should_bet, reason = should_take_bet(
                market=market,
                line=line,
                confidence=confidence,
                side=side,
                player_cv=player_cv,
                game_total=game_total,
                config=active_filter
            )

            if should_bet:
                wf_passed.append(idx)
            else:
                wf_rejections.append((idx, reason))

        # Add filter columns to ALL rows
        df['wf_filter_passed'] = df.index.isin(wf_passed)
        df['wf_filter_reason'] = ''
        for idx, reason in wf_rejections:
            df.loc[idx, 'wf_filter_reason'] = reason

        # Count passes by reason
        from collections import Counter
        rejection_reasons = Counter([r[1].split('(')[0].strip() for r in wf_rejections])

        logger.info(f"WALK-FORWARD VALIDATED FILTER: {len(wf_passed)}/{before_wf_filter} bets passed")
        logger.info(f"  Filter profile: CONSERVATIVE (receptions, >60% conf, UNDER only)")
        if rejection_reasons:
            top_reasons = rejection_reasons.most_common(5)
            logger.info(f"  Top rejection reasons: {dict(top_reasons)}")

        # Mark elite bets (lines 0-3 in receptions) separately
        elite_mask = (
            (df['market'] == 'player_receptions') &
            (df['line'] <= 3.0) &
            (df['wf_filter_passed'] == True)
        )
        df['elite_filter_passed'] = elite_mask
        elite_count = elite_mask.sum()
        if elite_count > 0:
            logger.info(f"  ELITE BETS (lines 0-3): {elite_count} bets (81.2% historical win rate)")

        # OPTIONAL: Hard filter (comment out to keep all bets with flag)
        # Uncomment to ONLY output filtered bets:
        # df = df[df['wf_filter_passed'] == True]
        # logger.info(f"  After hard filter: {len(df)} bets remaining")

    except Exception as e:
        df['wf_filter_passed'] = True  # Don't filter if error
        df['wf_filter_reason'] = ''
        df['elite_filter_passed'] = False
        logger.warning(f"Walk-forward validated filter failed (continuing without): {e}")

    logger.info(f"Final: {len(df)} valid recommendations")

    # ADD MODEL REASONING FOR EACH RECOMMENDATION
    # This explains WHY the model made this prediction
    logger.info("Generating model reasoning explanations...")
    reasoning_list = []
    for idx, row in df.iterrows():
        reasoning = generate_model_reasoning(row.to_dict())
        reasoning_list.append(reasoning)
    df['model_reasoning'] = reasoning_list
    logger.info(f"   Added reasoning to {len(df)} recommendations")

    return df


def calculate_model_confidence(row: pd.Series) -> int:
    """
    Calculate MODEL CONFIDENCE - how much the model trusts its own prediction.
    This is DIFFERENT from hit probability.

    Factors:
    1. Relative variance (std / projection) - lower = better
    2. SNR tier - HIGH markets are more predictable
    3. Player tier - workhorses more stable
    4. Snap share - consistent usage
    5. Historical sample - more data = better
    6. Model-market agreement - closer = market validates our view

    Returns:
        int: Confidence score 0-100
    """
    score = 50  # Start at 50/100

    # 1. Relative variance (model_std / model_projection)
    proj = row.get('model_projection', 0) or 0
    std = row.get('model_std', 0) or 0
    if proj > 0 and std > 0:
        rel_var = std / proj
        if rel_var < 0.20:
            score += 15  # Very tight projection
        elif rel_var < 0.30:
            score += 10
        elif rel_var < 0.40:
            score += 5
        elif rel_var > 0.60:
            score -= 10  # High variance = less confident

    # 2. SNR tier - market predictability
    snr = str(row.get('snr_tier', '')).upper()
    if snr == 'HIGH':
        score += 12
    elif snr == 'MEDIUM':
        score += 5
    elif snr == 'LOW':
        score -= 5

    # 3. Player tier - usage stability
    tier = str(row.get('calibration_tier', '')).lower()
    if 'tier_1' in tier or 'workhorse' in tier:
        score += 10
    elif 'tier_2' in tier:
        score += 5
    elif 'tier_3' in tier or 'committee' in tier:
        score -= 5

    # 4. Snap share - usage consistency
    snap = row.get('snap_share', 0) or 0
    if snap >= 0.70:
        score += 8
    elif snap >= 0.50:
        score += 4
    elif snap < 0.30:
        score -= 8

    # 5. Historical sample size
    hist = row.get('hist_count', 0) or 0
    if hist >= 10:
        score += 8
    elif hist >= 5:
        score += 4
    elif hist < 3:
        score -= 5

    # 6. Model-market agreement (closer = market validates our view)
    deviation = row.get('model_market_deviation', 0) or 0
    if deviation < 0.10:
        score += 5  # Strong agreement
    elif deviation > 0.20:
        score -= 3  # Model disagrees with market

    return min(max(score, 0), 100)  # Clamp to 0-100


def get_confidence_label(score: int) -> str:
    """Convert numeric confidence to label."""
    if score >= 80:
        return "VERY HIGH"
    elif score >= 65:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    elif score >= 35:
        return "LOW"
    else:
        return "VERY LOW"


def archive_existing_recommendations():
    """Archive existing recommendations before generating new ones."""
    import shutil

    recs_file = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
    archive_dir = Path('data/archive/recommendations')
    archive_dir.mkdir(parents=True, exist_ok=True)

    if not recs_file.exists():
        logger.info("No existing recommendations to archive")
        return None

    # Get modification time of existing file
    mod_time = datetime.fromtimestamp(recs_file.stat().st_mtime)
    timestamp = mod_time.strftime("%Y%m%d_%H%M%S")

    # Create archive filename
    archive_name = f"recommendations_{timestamp}.csv"
    archive_path = archive_dir / archive_name

    # Don't archive if we already have this exact timestamp
    if archive_path.exists():
        logger.info(f"Archive already exists: {archive_name}")
        return archive_path

    # Copy to archive
    shutil.copy2(recs_file, archive_path)
    logger.info(f"📦 Archived existing recommendations to: {archive_name}")

    # Clean up old archives (keep last 30)
    archives = sorted(archive_dir.glob("recommendations_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(archives) > 30:
        for old_file in archives[30:]:
            old_file.unlink()
            logger.debug(f"Removed old archive: {old_file.name}")

    return archive_path


def save_recommendations(df: pd.DataFrame):
    """Save recommendations to CSV."""
    # Archive existing recommendations first
    archive_existing_recommendations()

    # Calculate model confidence scores
    df['model_confidence'] = df.apply(calculate_model_confidence, axis=1)
    df['model_confidence_label'] = df['model_confidence'].apply(get_confidence_label)
    logger.info(f"Added model confidence scores (VERY HIGH: {len(df[df['model_confidence_label']=='VERY HIGH'])}, HIGH: {len(df[df['model_confidence_label']=='HIGH'])}, MEDIUM: {len(df[df['model_confidence_label']=='MEDIUM'])})")

    output_path = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} recommendations to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total recommendations: {len(df)}")

    # PRIORITY: MODEL-VALIDATED BETS
    if 'model_validated' in df.columns:
        validated = df[df['model_validated'] == True].copy()
        if len(validated) > 0:
            # Sort by EV descending
            validated = validated.sort_values('model_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ MODEL-VALIDATED BETS ★★★")
            print("=" * 80)
            print("Strategy: LVT + Defense EPA + Usage features")
            print("Walk-forward validated with positive ROI")
            print()

            # Check for opponent conflicts
            if 'opponent_conflict' in df.columns:
                conflicts = validated[validated['opponent_conflict'] == True]
                if len(conflicts) > 0:
                    print(f"⚠️  WARNING: {len(conflicts)} picks have OPPONENT CONFLICT - review carefully!")
                    for _, row in conflicts.iterrows():
                        market_name = format_market_name(row.get('market', ''))
                        print(f"   {row['player']} ({market_name}): {row.get('opponent_conflict_reason', 'conflict')}")
                    print()

            # Show breakdown by direction
            under_bets = validated[validated['model_best_direction'] == 'UNDER']
            over_bets = validated[validated['model_best_direction'] == 'OVER']
            print(f"UNDER bets: {len(under_bets)}")
            print(f"OVER bets: {len(over_bets)}")
            print()

            # Show by market (using display names)
            print("By market:")
            for market in validated['market'].unique():
                market_df = validated[validated['market'] == market]
                avg_ev = market_df['model_best_ev'].mean()
                market_name = format_market_name(market)
                print(f"  - {market_name}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%)")

            print(f"\nValidated bets this week: {len(validated)}")

            # Use display columns with formatted names
            display_df = validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['model_best_direction']
            display_df['Prob'] = display_df['model_p_under'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            display_df['EV'] = display_df['model_best_ev'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'Prob', 'EV', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Highlight V7 HYBRID APPROACH bets (rule-based with regime estimation)
    if 'v7_validated' in df.columns:
        v7_validated = df[df['v7_validated'] == True].copy()
        if len(v7_validated) > 0:
            # Sort by EV descending
            v7_validated = v7_validated.sort_values('v7_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ V7 RULE-BASED BETS (SIMPLE RULE + REGIME) ★★★")
            print("=" * 80)
            print("Strategy: Bet UNDER when line > trailing_avg + 1.5")
            print("Backtest: Receptions 72.4% hit, +38.2% ROI (2025)")
            print("         Rushing Yards 54.3% hit, +3.7% ROI (consistent)")
            print()
            print("Why this works: Vegas sets lines above recent performance,")
            print("                but players tend to regress toward their mean.")
            print()

            # Show breakdown by direction
            under_bets = v7_validated[v7_validated['v7_best_direction'] == 'UNDER']
            over_bets = v7_validated[v7_validated['v7_best_direction'] == 'OVER']
            print(f"UNDER bets: {len(under_bets)}")
            print(f"OVER bets: {len(over_bets)}")
            print()

            # Show by market (using display names)
            print("By market:")
            for market in v7_validated['market'].unique():
                market_df = v7_validated[v7_validated['market'] == market]
                avg_ev = market_df['v7_best_ev'].mean()
                avg_kelly = market_df['v7_kelly_size'].mean() * 100
                market_name = format_market_name(market)
                print(f"  - {market_name}: {len(market_df)} bets (avg EV: {avg_ev:+.1f}%, Kelly: {avg_kelly:.1f}%)")

            print(f"\nV7-validated bets this week: {len(v7_validated)}")

            # Use display columns with formatted names
            display_df = v7_validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['v7_best_direction']
            display_df['LVT'] = display_df['line_vs_trailing'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
            display_df['EV'] = display_df['v7_best_ev'].apply(lambda x: f"{x:+.1f}%")
            display_df['Kelly'] = display_df['v7_kelly_size'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'LVT', 'EV', 'Kelly', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Legacy V6/V5 output for backwards compatibility
    elif 'v6_value_validated' in df.columns:
        v6_validated = df[df['v6_value_validated'] == True].copy()
        if len(v6_validated) > 0:
            # Sort by EV descending
            v6_validated = v6_validated.sort_values('v6_best_ev', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ V6 VALUE CLASSIFIER BETS (POSITIVE EV) ★★★")
            print("=" * 80)

            # Format display columns
            display_df = v6_validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Direction'] = display_df['v6_best_direction']
            display_df['EV'] = display_df['v6_best_ev'].apply(lambda x: f"{x:+.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'Direction', 'line', 'model_projection', 'EV', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].head(20).to_string(index=False))
            print("=" * 80)

    # Highlight PROFITABLE STRATEGY bets (backtest-validated)
    if 'strategy_validated' in df.columns:
        validated = df[df['strategy_validated'] == True].copy()
        if len(validated) > 0:
            # Sort by edge descending
            validated = validated.sort_values('edge_pct', ascending=False)

            print("\n" + "=" * 80)
            print("★★★ PROFITABLE STRATEGY BETS (RULE-BASED VALIDATION) ★★★")
            print("=" * 80)
            print("Strategy: UNDER on receptions/passing yards when model agrees with line")
            print("Backtest: 57.8% win rate, +10.3% ROI (Weeks 2-11, 2025)")
            print(f"\nValidated bets this week: {len(validated)}")

            # Format display columns
            display_df = validated.copy()
            display_df['Prop'] = display_df['market'].apply(format_market_name)
            display_df['Edge'] = display_df['edge_pct'].apply(lambda x: f"{x:+.1f}%")
            display_df['Conf'] = display_df['confidence_pct'] if 'confidence_pct' in display_df.columns else display_df['confidence']

            display_cols = ['player', 'Prop', 'pick', 'line', 'model_projection', 'Edge', 'Conf', 'model_reasoning']
            display_cols = [c for c in display_cols if c in display_df.columns]
            print(display_df[display_cols].to_string(index=False))
            print("=" * 80)

    # Summary by market (using display names)
    print(f"\nBy market:")
    market_counts = df['market'].value_counts()
    for market, count in market_counts.items():
        market_name = format_market_name(market)
        print(f"  {market_name}: {count}")

    # Top 10 by edge (using display names)
    print(f"\nTop 10 by edge:")
    top_bets = df.nlargest(10, 'edge_pct').copy()
    top_bets['Prop'] = top_bets['market'].apply(format_market_name)
    top_bets['Edge'] = top_bets['edge_pct'].apply(lambda x: f"{x:+.1f}%")
    display_cols = ['player', 'Prop', 'line', 'model_projection', 'Edge']
    print(top_bets[display_cols].to_string(index=False))
    print("\n" + "=" * 80)


def generate_game_lines_recommendations(week: int, season: int = None):
    """Generate game lines predictions and display them."""
    try:
        from scripts.predict.predict_game_lines import (
            load_game_odds, build_game_features, predict_game_lines, get_current_season
        )

        if season is None:
            season = get_current_season()

        # Load odds
        games = load_game_odds(season)
        if games.empty:
            return None

        # Build features
        games_df = build_game_features(games, season, week)

        # Generate predictions
        predictions = predict_game_lines(games_df)

        return predictions

    except Exception as e:
        logger.warning(f"Could not generate game lines: {e}")
        return None


def display_game_lines(predictions):
    """Display game lines predictions."""
    if predictions is None or predictions.empty:
        return

    totals_bets = predictions[predictions['totals_validated'] == True] if 'totals_validated' in predictions.columns else pd.DataFrame()
    spread_bets = predictions[predictions['spread_validated'] == True] if 'spread_validated' in predictions.columns else pd.DataFrame()

    if len(totals_bets) == 0 and len(spread_bets) == 0:
        return

    print("\n" + "=" * 80)
    print("★★★ GAME LINES BETS ★★★")
    print("=" * 80)
    print("Strategy: Defense EPA + Line features (walk-forward validated)")
    print()

    if len(totals_bets) > 0:
        print(f"--- TOTALS ({len(totals_bets)} bets) ---")
        for _, bet in totals_bets.iterrows():
            prob = bet.get('totals_prob', 0)
            ev = bet.get('totals_ev', 0)
            price = bet.get('totals_price', -110)
            print(f"  {bet['matchup']}: {bet['totals_pick']} {bet['totals_line']}")
            print(f"    Prob: {prob:.1%} | EV: {ev*100:+.1f}% | Price: {price}")
        print()

    if len(spread_bets) > 0:
        print(f"--- SPREADS ({len(spread_bets)} bets) ---")
        for _, bet in spread_bets.iterrows():
            prob = bet.get('spread_prob', 0)
            ev = bet.get('spread_ev', 0)
            price = bet.get('spread_price', -110)
            print(f"  {bet['matchup']}: {bet['spread_pick']}")
            print(f"    Prob: {prob:.1%} | EV: {ev*100:+.1f}% | Price: {price}")

    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate unified recommendations')
    parser.add_argument('--week', type=int, default=None, help='NFL week')
    args = parser.parse_args()

    week = args.week or detect_week_from_props()

    recommendations = generate_recommendations(week)

    if recommendations.empty:
        logger.error("No recommendations generated!")
        sys.exit(1)

    save_recommendations(recommendations)

    # Also generate game lines predictions
    game_lines = generate_game_lines_recommendations(week)
    if game_lines is not None:
        display_game_lines(game_lines)
        # Save game lines to separate file
        output_path = Path('reports') / f'game_lines_predictions_week{week}.csv'
        game_lines.to_csv(output_path, index=False)
        logger.info(f"Saved game lines to {output_path}")

    logger.info("Done! Run generate_all_recommendations.py next to add quality tiers.")


if __name__ == '__main__':
    main()
