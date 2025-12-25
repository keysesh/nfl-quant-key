#!/usr/bin/env python3
"""
Integration tests for NFL QUANT pipeline.

These tests verify:
1. End-to-end pipeline works correctly
2. New V28 features (Elo, YBC, rest_days, HFA) are extracted
3. No critical failures in the data → features → model → output flow
4. Feature extraction produces expected outputs
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# V28 ELO RATING SYSTEM TESTS
# =============================================================================

class TestEloRatingSystem:
    """Test the Elo rating system."""

    def test_elo_initialization(self):
        """Test Elo system initializes with 32 teams."""
        from nfl_quant.models.elo_ratings import EloRatingSystem

        elo = EloRatingSystem()
        assert len(elo.ratings) == 32, "Should have 32 NFL teams"
        assert all(r == 1500.0 for r in elo.ratings.values()), "All teams should start at 1500"

    def test_elo_expected_score(self):
        """Test expected score calculation."""
        from nfl_quant.models.elo_ratings import EloRatingSystem

        elo = EloRatingSystem()

        # Equal ratings = 50% expected
        exp = elo.expected_score(1500, 1500)
        assert abs(exp - 0.5) < 0.01, f"Equal ratings should give 50% expected, got {exp}"

        # 100 Elo advantage ≈ 64%
        exp = elo.expected_score(1600, 1500)
        assert 0.60 < exp < 0.68, f"100 Elo advantage should be ~64%, got {exp:.1%}"

    def test_elo_spread_prediction(self):
        """Test spread prediction from Elo."""
        from nfl_quant.models.elo_ratings import EloRatingSystem

        elo = EloRatingSystem()
        elo.ratings['KC'] = 1600
        elo.ratings['BUF'] = 1550

        # KC at home vs BUF
        spread = elo.get_spread_prediction('KC', 'BUF')

        # (1600 + 48 - 1550) / 25 = 3.92 points
        assert 3.5 < spread < 4.5, f"Spread should be ~4 points, got {spread}"

    def test_elo_game_update(self):
        """Test Elo ratings update after a game."""
        from nfl_quant.models.elo_ratings import EloRatingSystem

        elo = EloRatingSystem()
        old_kc = elo.ratings['KC']
        old_buf = elo.ratings['BUF']

        # KC beats BUF at home 35-28
        elo.update_game('KC', 'BUF', 35, 28)

        assert elo.ratings['KC'] > old_kc, "Winner rating should increase"
        assert elo.ratings['BUF'] < old_buf, "Loser rating should decrease"

    def test_elo_save_load(self, tmp_path):
        """Test saving and loading Elo ratings."""
        from nfl_quant.models.elo_ratings import EloRatingSystem

        elo = EloRatingSystem()
        elo.ratings['KC'] = 1600
        elo.ratings['BUF'] = 1550

        filepath = tmp_path / "elo_test.json"
        elo.save_ratings(str(filepath))

        # Load into new instance
        elo2 = EloRatingSystem()
        success = elo2.load_ratings(str(filepath))

        assert success, "Should load successfully"
        assert elo2.ratings['KC'] == 1600, "KC rating should be preserved"
        assert elo2.ratings['BUF'] == 1550, "BUF rating should be preserved"


# =============================================================================
# V28 SITUATIONAL FEATURES TESTS
# =============================================================================

class TestSituationalFeatures:
    """Test situational features (rest_days, HFA)."""

    def test_hfa_adjustment_range(self):
        """Test HFA adjustment values are in expected range."""
        from nfl_quant.features.situational_features import get_hfa_adjustment

        # Strong home teams
        den_hfa = get_hfa_adjustment('DEN', is_home=True)
        assert 1.2 <= den_hfa <= 1.4, f"DEN should have strong HFA, got {den_hfa}"

        # Weak home teams
        lac_hfa = get_hfa_adjustment('LAC', is_home=True)
        assert 0.8 <= lac_hfa <= 0.9, f"LAC should have weak HFA, got {lac_hfa}"

    def test_hfa_away_inversion(self):
        """Test away team gets inverse HFA."""
        from nfl_quant.features.situational_features import get_hfa_adjustment

        home_hfa = get_hfa_adjustment('DEN', is_home=True)
        away_hfa = get_hfa_adjustment('DEN', is_home=False)

        # Away = 2 - Home
        expected_away = 2.0 - home_hfa
        assert abs(away_hfa - expected_away) < 0.01, f"Away HFA should be {expected_away}, got {away_hfa}"

    def test_situational_features_dict(self):
        """Test combined situational features returns expected keys."""
        from nfl_quant.features.situational_features import get_situational_features

        features = get_situational_features('KC', 'BUF', 2025, 10, is_home=True)

        assert 'rest_days' in features
        assert 'rest_advantage' in features
        assert 'hfa_adjustment' in features
        assert 'is_primetime' in features
        assert 'is_divisional' in features


# =============================================================================
# V28 RISK OF RUIN TESTS
# =============================================================================

class TestRiskOfRuin:
    """Test risk of ruin calculations."""

    def test_ror_no_edge(self):
        """With no edge, RoR should be 1.0."""
        from nfl_quant.betting.risk_of_ruin import calculate_risk_of_ruin

        # 50% win rate at even odds = no edge
        ror = calculate_risk_of_ruin(
            bankroll=1000,
            bet_size=50,
            win_prob=0.50,
            odds=100  # Even money
        )
        assert ror == 1.0, f"No edge should give 100% RoR, got {ror}"

    def test_ror_with_edge(self):
        """With positive edge, RoR should be < 1."""
        from nfl_quant.betting.risk_of_ruin import calculate_risk_of_ruin

        # 55% win rate at -110 = positive edge
        ror = calculate_risk_of_ruin(
            bankroll=1000,
            bet_size=50,
            win_prob=0.55,
            odds=-110
        )
        assert 0 < ror < 1.0, f"Positive edge should give RoR < 1, got {ror}"

    def test_kelly_fraction(self):
        """Test Kelly fraction calculation."""
        from nfl_quant.betting.risk_of_ruin import kelly_fraction

        # 55% at -110 (decimal 1.91)
        kelly = kelly_fraction(0.55, 1.91)
        assert 0 < kelly < 0.15, f"Kelly should be modest (~5-10%), got {kelly}"

        # 60% at even money (decimal 2.0)
        kelly = kelly_fraction(0.60, 2.0)
        assert 0.15 < kelly < 0.25, f"Kelly should be ~20%, got {kelly}"

    def test_recommend_position_size(self):
        """Test position sizing recommendation."""
        from nfl_quant.betting.risk_of_ruin import recommend_position_size

        rec = recommend_position_size(
            bankroll=1000,
            edge=0.05,  # 5% edge
            odds=-110,
            risk_tolerance='moderate'
        )

        assert 'recommended_bet' in rec
        assert rec['recommended_bet'] > 0
        assert rec['recommended_bet'] <= 50, "Should be capped at 5% of bankroll"


# =============================================================================
# V28 BASELINE REGRESSION TESTS
# =============================================================================

class TestBaselineRegression:
    """Test baseline regression models."""

    def test_logistic_fit_predict(self):
        """Test baseline logistic regression fit and predict."""
        from nfl_quant.models.baseline_regression import BaselineLogistic

        # Create dummy data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + np.random.randn(100) * 0.5 > 0).astype(int)

        model = BaselineLogistic()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2), "Should return probabilities for 2 classes"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"

    def test_logistic_feature_importance(self):
        """Test feature importance extraction."""
        from nfl_quant.models.baseline_regression import BaselineLogistic

        np.random.seed(42)
        X = pd.DataFrame({
            'feat_a': np.random.randn(100),
            'feat_b': np.random.randn(100),
            'feat_c': np.random.randn(100),
        })
        y = (X['feat_a'] > 0).astype(int)

        model = BaselineLogistic()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert 'feat_a' in importance
        assert importance['feat_a'] > importance['feat_b'], "feat_a should be most important"

    def test_linear_regression(self):
        """Test baseline linear regression."""
        from nfl_quant.models.baseline_regression import BaselineLinear

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.5

        model = BaselineLinear()
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == 100
        assert preds.std() > 0, "Predictions should vary"


# =============================================================================
# V28 YBC PROXY TESTS
# =============================================================================

class TestYBCProxy:
    """Test YBC proxy feature."""

    def test_ybc_proxy_position_defaults(self):
        """Test YBC proxy has correct position defaults."""
        from nfl_quant.features.core import FeatureEngine

        engine = FeatureEngine()

        # Position defaults
        expected = {
            'WR': 8.0,
            'TE': 6.5,
            'RB': 1.5,
            'QB': 2.0,
        }

        for pos, expected_ybc in expected.items():
            # get_ybc_proxy with no data should return default
            ybc = engine.get_ybc_proxy(
                player_id='unknown_player',
                position=pos,
                season=2025,
                week=10
            )
            assert abs(ybc - expected_ybc) < 0.1, f"{pos} YBC should be {expected_ybc}, got {ybc}"


# =============================================================================
# FEATURE EXTRACTION INTEGRATION TESTS
# =============================================================================

class TestFeatureExtraction:
    """Test feature extraction pipeline."""

    def test_batch_extractor_v28_columns(self):
        """Test that V28 columns are added by batch extractor."""
        from nfl_quant.features.batch_extractor import _add_v28_elo_situational_features

        # Create minimal test dataframe
        df = pd.DataFrame({
            'team': ['KC', 'BUF', 'SF'],
            'opponent': ['BUF', 'KC', 'DAL'],
            'season': [2025, 2025, 2025],
            'week': [10, 10, 10],
            'position': ['WR', 'WR', 'RB'],
        })

        result = _add_v28_elo_situational_features(df)

        # Check V28 columns exist
        v28_columns = [
            'elo_rating_home',
            'elo_rating_away',
            'elo_diff',
            'ybc_proxy',
            'rest_days',
            'hfa_adjustment',
        ]

        for col in v28_columns:
            assert col in result.columns, f"V28 column {col} missing from result"

        # Check no NaN in critical columns
        assert result['elo_rating_home'].isna().sum() == 0, "elo_rating_home should not have NaN"
        assert result['rest_days'].isna().sum() == 0, "rest_days should not have NaN"


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestModelConfig:
    """Test model configuration."""

    def test_v28_features_in_config(self):
        """Test V28 features are in config."""
        from configs.model_config import FEATURES

        v28_features = [
            'elo_rating_home',
            'elo_rating_away',
            'elo_diff',
            'ybc_proxy',
            'rest_days',
            'hfa_adjustment',
        ]

        for feat in v28_features:
            assert feat in FEATURES, f"V28 feature {feat} missing from config"

    def test_model_version_is_28(self):
        """Test model version is updated to 28."""
        from configs.model_config import MODEL_VERSION

        assert MODEL_VERSION == "28", f"Expected version 28, got {MODEL_VERSION}"

    def test_feature_count(self):
        """Test feature count is correct."""
        from configs.model_config import FEATURES, FEATURE_COUNT

        assert len(FEATURES) == FEATURE_COUNT, f"Feature count mismatch: {len(FEATURES)} vs {FEATURE_COUNT}"
        assert FEATURE_COUNT == 55, f"Expected 55 features, got {FEATURE_COUNT}"

    def test_injury_features_in_config(self):
        """Test V28.1 injury features are in config."""
        from configs.model_config import FEATURES

        injury_features = [
            'injury_status_encoded',
            'practice_status_encoded',
            'has_injury_designation',
        ]

        for feat in injury_features:
            assert feat in FEATURES, f"Injury feature {feat} missing from config"


# =============================================================================
# V28.1 PLAYER INJURY FEATURE TESTS
# =============================================================================

class TestPlayerInjuryFeatures:
    """Test player injury feature extraction."""

    def test_injury_status_encoding(self):
        """Test injury status encoding values."""
        # Out should be highest (3), None/Probable lowest (0)
        status_map = {'Out': 3, 'Doubtful': 2, 'Questionable': 1, 'Probable': 0}

        assert status_map['Out'] > status_map['Doubtful']
        assert status_map['Doubtful'] > status_map['Questionable']
        assert status_map['Questionable'] > status_map['Probable']

    def test_practice_status_encoding(self):
        """Test practice status encoding values."""
        # DNP should be highest (2), Full lowest (0)
        practice_map = {'DNP': 2, 'Limited': 1, 'Full': 0}

        assert practice_map['DNP'] > practice_map['Limited']
        assert practice_map['Limited'] > practice_map['Full']

    def test_injury_extraction_with_dummy_data(self):
        """Test injury feature extraction adds correct columns."""
        from nfl_quant.features.batch_extractor import _add_player_injury_features

        # Create minimal test dataframe
        df = pd.DataFrame({
            'gsis_id': ['00-0035228', '00-0036212'],
            'season': [2024, 2024],
            'week': [1, 1],
            'team': ['KC', 'BUF'],
        })

        result = _add_player_injury_features(df)

        # Check injury columns exist
        assert 'injury_status_encoded' in result.columns
        assert 'practice_status_encoded' in result.columns
        assert 'has_injury_designation' in result.columns

        # Values should be integers 0-3
        assert result['injury_status_encoded'].dtype in [np.int64, np.int32, int]
        assert result['injury_status_encoded'].max() <= 3

    def test_unified_injury_data_coverage(self):
        """Test unified injury history covers 2024-2025."""
        from nfl_quant.config_paths import PROJECT_ROOT

        unified_path = PROJECT_ROOT / 'data' / 'processed' / 'unified_injury_history.parquet'

        if not unified_path.exists():
            pytest.skip("Unified injury history not generated yet")

        unified = pd.read_parquet(unified_path)

        # Should have both seasons
        seasons = set(unified['season'].unique())
        assert 2024 in seasons, "Should have 2024 data from NFLverse"
        assert 2025 in seasons, "Should have 2025 data from Sleeper"

        # Should have required columns
        assert 'injury_status_encoded' in unified.columns
        assert 'practice_status_encoded' in unified.columns
        assert 'player_name' in unified.columns

        # Should have reasonable record count
        assert len(unified) > 5000, f"Expected >5000 records, got {len(unified)}"


# =============================================================================
# PIPELINE E2E SMOKE TEST
# =============================================================================

class TestPipelineSmokeTest:
    """Smoke tests for the full pipeline."""

    @pytest.mark.slow
    def test_elo_initialization_from_schedule(self):
        """Test Elo can be initialized from schedule data (if available)."""
        from nfl_quant.models.elo_ratings import EloRatingSystem
        from nfl_quant.config_paths import PROJECT_ROOT

        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'

        if not schedule_path.exists():
            pytest.skip("Schedule data not available")

        schedule = pd.read_parquet(schedule_path)
        elo = EloRatingSystem()
        elo.initialize_from_schedule(schedule, seasons=[2024])

        # KC should have above-average rating after 2024
        assert elo.ratings['KC'] > 1500, "KC should be above average after 2024 season"

        # Rankings should show variation
        rankings = elo.get_rankings()
        assert rankings['elo'].std() > 30, "Ratings should show meaningful variation"
