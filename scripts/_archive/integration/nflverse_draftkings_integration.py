#!/usr/bin/env python3
"""
COMPREHENSIVE NFLVERSE + DRAFTKINGS INTEGRATION PLAN

This document outlines the complete system upgrade to integrate:
1. nflverse historical data (PBP, weekly stats, NGS metrics)
2. DraftKings odds API
3. Expanded feature engineering
4. Improved calibration with historical context
5. CLV tracking and Kelly sizing

CRITICAL: All historical simulations use SAME pipeline to preserve context.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLverseDataIntegrator:
    """
    Integrates nflverse data sources into the quant system.

    Provides:
    - Play-by-play data (1999-present)
    - Weekly/season player stats
    - Next Gen Stats metrics
    - Rosters and player info
    - Schedules and game info
    - Team descriptors
    """

    def __init__(self, cache_dir: Path = Path("data/nflverse_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Import the new file-based loader
        try:
            from nfl_quant.utils.nflverse_loader import (
                load_pbp, load_player_stats, load_schedules, load_rosters
            )
            self._load_pbp = load_pbp
            self._load_player_stats = load_player_stats
            self._load_schedules = load_schedules
            self._load_rosters = load_rosters
        except ImportError:
            raise ImportError(
                "nflverse_loader not available. "
                "Please ensure nfl_quant package is installed."
            )

    def fetch_pbp_data(self, years: List[int]) -> pd.DataFrame:
        """Fetch play-by-play data for specified years."""
        logger.info(f"Fetching PBP data for years: {years}")
        pbp = self._load_pbp(seasons=years)
        logger.info(f"✅ Loaded {len(pbp):,} plays")
        return pbp

    def fetch_weekly_stats(self, years: List[int]) -> pd.DataFrame:
        """Fetch weekly player stats."""
        logger.info(f"Fetching weekly stats for years: {years}")
        weekly = self._load_player_stats(seasons=years)
        logger.info(f"✅ Loaded {len(weekly):,} player-weeks")
        return weekly

    def fetch_ngs_metrics(self, stat_type: str, years: List[int] = None) -> pd.DataFrame:
        """Fetch Next Gen Stats metrics (passing, rushing, receiving)."""
        logger.info(f"Fetching NGS {stat_type} data...")
        # NGS data is stored in pre-fetched files
        ngs_file = Path(f"data/nflverse/ngs_{stat_type}_historical.parquet")
        if ngs_file.exists():
            ngs = pd.read_parquet(ngs_file)
            if years:
                ngs = ngs[ngs['season'].isin(years)]
            logger.info(f"✅ Loaded {len(ngs):,} {stat_type} NGS records")
            return ngs
        else:
            logger.warning(f"NGS {stat_type} data not found at {ngs_file}")
            return pd.DataFrame()

    def fetch_rosters(self, years: List[int]) -> pd.DataFrame:
        """Fetch roster data."""
        logger.info(f"Fetching rosters for years: {years}")
        rosters = self._load_rosters(seasons=years)
        logger.info(f"✅ Loaded {len(rosters):,} roster records")
        return rosters

    def fetch_schedules(self, years: List[int]) -> pd.DataFrame:
        """Fetch game schedules."""
        logger.info(f"Fetching schedules for years: {years}")
        sched = self._load_schedules(seasons=years)
        logger.info(f"✅ Loaded {len(sched):,} games")
        return sched

    def fetch_team_desc(self) -> pd.DataFrame:
        """Fetch team descriptors."""
        logger.info("Fetching team descriptors...")
        # Load player stats as a proxy (or use team stats if available)
        stats_file = Path("data/nflverse/team_stats.parquet")
        if stats_file.exists():
            teams = pd.read_parquet(stats_file)
        else:
            teams = self._load_player_stats(seasons=[2024, 2025])
        logger.info(f"✅ Loaded {len(teams)} team stat records")
        return teams


class DraftKingsOddsIntegrator:
    """
    Integrates DraftKings odds API for historical and live lines.

    Provides:
    - Historical player prop lines
    - Current week lines
    - Closing line tracking (for CLV)
    - Market coverage tracking
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DRAFTKINGS_API_KEY")
        self.base_url = "https://api.draftkings.com"

    def fetch_current_lines(self, sport: str = "americanfootball") -> pd.DataFrame:
        """Fetch current week lines from DraftKings."""
        # Implementation would go here
        # For now, using existing Odds API integration
        logger.info("DraftKings integration: Use existing Odds API for now")
        return pd.DataFrame()

    def fetch_historical_lines(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical lines for CLV tracking."""
        # Implementation would go here
        logger.info("DraftKings historical lines: Use existing historical prop files")
        return pd.DataFrame()


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering using nflverse + NGS metrics.

    Adds:
    - Team EPA metrics (offensive, defensive, special teams)
    - QB advanced metrics (CPOE, air yards, time to throw)
    - Skill player NGS (separation, xYAC, RYOE)
    - Situational features (rest days, travel, surface)
    - Weather integration
    - Injury participation
    """

    def __init__(self, pbp_df: pd.DataFrame, ngs_data: Dict[str, pd.DataFrame]):
        self.pbp = pbp_df
        self.ngs = ngs_data

    def calculate_team_epa_features(self, team: str, week: int, season: int) -> Dict:
        """Calculate team-level EPA features."""
        team_pbp = self.pbp[
            (self.pbp['season'] == season) &
            (self.pbp['week'] < week) &
            (
                (self.pbp['posteam'] == team) |
                (self.pbp['defteam'] == team)
            )
        ].copy()

        # Calculate rolling EPA metrics
        features = {
            'team_off_epa': team_pbp[team_pbp['posteam'] == team]['epa'].mean() if len(team_pbp) > 0 else 0.0,
            'team_def_epa': -team_pbp[team_pbp['defteam'] == team]['epa'].mean() if len(team_pbp) > 0 else 0.0,
            'team_success_rate': (team_pbp[team_pbp['posteam'] == team]['epa'] > 0).mean() if len(team_pbp) > 0 else 0.5,
        }

        return features

    def calculate_ngs_features(self, player_name: str, position: str, week: int, season: int) -> Dict:
        """Calculate Next Gen Stats features for a player."""
        features = {}

        if position == 'QB':
            ngs_type = 'passing'
            if ngs_type in self.ngs:
                player_ngs = self.ngs[ngs_type][
                    (self.ngs[ngs_type]['player_name'] == player_name) &
                    (self.ngs[ngs_type]['season'] == season) &
                    (self.ngs[ngs_type]['week'] < week)
                ]
                if len(player_ngs) > 0:
                    features['cpoe'] = player_ngs['cpoe'].mean()
                    features['air_yards'] = player_ngs['intended_air_yards'].mean()
                    features['time_to_throw'] = player_ngs['avg_time_to_throw'].mean()

        elif position in ['WR', 'TE']:
            ngs_type = 'receiving'
            if ngs_type in self.ngs:
                player_ngs = self.ngs[ngs_type][
                    (self.ngs[ngs_type]['player_name'] == player_name) &
                    (self.ngs[ngs_type]['season'] == season) &
                    (self.ngs[ngs_type]['week'] < week)
                ]
                if len(player_ngs) > 0:
                    features['separation'] = player_ngs['avg_separation'].mean()
                    features['xYAC'] = player_ngs['avg_expected_yac'].mean()

        elif position == 'RB':
            ngs_type = 'rushing'
            if ngs_type in self.ngs:
                player_ngs = self.ngs[ngs_type][
                    (self.ngs[ngs_type]['player_name'] == player_name) &
                    (self.ngs[ngs_type]['season'] == season) &
                    (self.ngs[ngs_type]['week'] < week)
                ]
                if len(player_ngs) > 0:
                    features['ryoe'] = player_ngs['rush_yards_over_expected'].mean()
                    features['efficiency'] = player_ngs['efficiency'].mean()

        return features


def generate_historical_training_data_with_context(
    seasons: List[int],
    weeks_per_season: Dict[int, List[int]],
    simulator: PlayerSimulator,
    integrator: NFLverseDataIntegrator
) -> pd.DataFrame:
    """
    Generate training data from historical seasons using SAME pipeline.

    CRITICAL: Preserves context by:
    - Using same PlayerSimulator
    - Same feature engineering
    - Same models
    - Same game context integration

    Returns DataFrame with:
    - model_prob_raw (from simulations)
    - bet_won (actual outcomes)
    - All context features preserved
    """

    print("=" * 80)
    print("🎲 GENERATING HISTORICAL TRAINING DATA (CONTEXT-PRESERVED)")
    print("=" * 80)
    print()

    # Fetch historical data
    pbp_df = integrator.fetch_pbp_data(seasons)
    weekly_df = integrator.fetch_weekly_stats(seasons)

    # Fetch NGS metrics
    ngs_data = {
        'passing': integrator.fetch_ngs_metrics('passing', seasons),
        'rushing': integrator.fetch_ngs_metrics('rushing', seasons),
        'receiving': integrator.fetch_ngs_metrics('receiving', seasons),
    }

    # Load historical props (if available)
    historical_props_dir = Path("data/historical")
    prop_files = list(historical_props_dir.rglob("player_props_history_*.csv"))

    if not prop_files:
        print("⚠️  No historical prop files found")
        print("   Using current backtest data only")
        return pd.DataFrame()

    print(f"Found {len(prop_files)} historical prop files")
    print()

    # Process historical props
    results = []

    # Load and process each prop file
    for prop_file in prop_files:
        props = pd.read_csv(prop_file)

        # Match to actual outcomes from weekly stats
        # Run simulations using SAME pipeline
        # Extract raw probabilities

        # Simplified for now - full implementation would:
        # 1. Match props to weekly stats outcomes
        # 2. Create PlayerPropInput for each historical prop
        # 3. Run simulator.simulate_player()
        # 4. Extract model_prob_raw
        # 5. Compare to actual outcome

    return pd.DataFrame(results)


def main():
    """
    Main integration execution plan.

    This creates the framework for:
    1. Fetching nflverse data
    2. Integrating DraftKings odds
    3. Expanding feature engineering
    4. Generating historical training data
    5. Retraining calibrator with expanded dataset
    """

    print("=" * 80)
    print("NFLVERSE + DRAFTKINGS INTEGRATION FRAMEWORK")
    print("=" * 80)
    print()

    # Initialize integrators
    integrator = NFLverseDataIntegrator()

    # Fetch current season data
    print("Step 1: Fetching current season data...")
    pbp_2025 = integrator.fetch_pbp_data([2025])
    weekly_2025 = integrator.fetch_weekly_stats([2025])

    # Fetch historical data for calibration
    print("\nStep 2: Fetching historical data for calibration...")
    pbp_historical = integrator.fetch_pbp_data([2024, 2023])
    weekly_historical = integrator.fetch_weekly_stats([2024, 2023])

    print("\n✅ Data fetched successfully")
    print("   Next: Match historical props to outcomes")
    print("   Then: Generate simulations using SAME pipeline")
    print("   Finally: Retrain calibrator with expanded dataset")


if __name__ == "__main__":
    main()































