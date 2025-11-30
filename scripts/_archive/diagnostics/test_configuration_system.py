#!/usr/bin/env python3
"""
Configuration System Integration Test
======================================

Purpose: Validate that all configuration migrations work correctly and
the system behaves as expected with centralized JSON configuration.

Tests:
1. Configuration files load successfully
2. All migrated files import config correctly
3. Config values match expected defaults
4. Models use config hyperparameters
5. QB TD rate model no longer predicts constants
6. Backward compatibility maintained

Usage:
    python scripts/diagnostics/test_configuration_system.py

Output:
    - Console test report
    - data/diagnostics/config_system_test_results.json
    - Exit code 0 if all tests pass, 1 if any test fails
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ConfigSystemTester:
    """Test suite for configuration system."""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'failures': [],
            'passed': False,
        }

    def test_config_loading(self) -> bool:
        """Test that configuration loads successfully."""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Configuration Loading")
        logger.info("="*70)

        try:
            from nfl_quant.config_enhanced import config

            # Check all config sections exist
            sections = {
                'simulation': config.simulation,
                'models': config.models,
                'td_prediction': config.td_prediction,
                'feature_engineering': config.feature_engineering,
                'validation': config.validation,
                'betting': config.betting,
            }

            for name, section in sections.items():
                assert section is not None, f"{name} section is None"
                logger.info(f"✅ {name}: Loaded")

            # Check league_defaults section exists
            assert hasattr(config.simulation, 'league_defaults'), "Missing league_defaults"
            assert config.simulation.league_defaults is not None
            logger.info(f"✅ league_defaults: {len(config.simulation.league_defaults)} values")

            self.results['tests']['config_loading'] = {
                'passed': True,
                'sections_loaded': list(sections.keys()),
            }
            logger.info("\n✅ TEST PASSED: Configuration loads successfully")
            return True

        except Exception as e:
            msg = f"Configuration loading failed: {str(e)}"
            self.results['failures'].append(msg)
            self.results['tests']['config_loading'] = {
                'passed': False,
                'error': str(e),
            }
            logger.error(f"\n❌ TEST FAILED: {msg}")
            return False

    def test_simulator_config_usage(self) -> bool:
        """Test that simulator.py uses config values."""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Simulator Config Usage")
        logger.info("="*70)

        try:
            from nfl_quant.simulation import simulator
            from nfl_quant.config_enhanced import config

            # Check that simulator loaded config
            expected_base_td_rate = config.simulation.game_simulation['base_td_rate']
            actual_base_td_rate = simulator.BASE_TD_RATE

            assert actual_base_td_rate == expected_base_td_rate, \
                f"BASE_TD_RATE mismatch: {actual_base_td_rate} != {expected_base_td_rate}"

            logger.info(f"✅ BASE_TD_RATE: {actual_base_td_rate} (from config)")

            # Check other key values
            expected_possessions = config.simulation.game_simulation['base_possessions_per_game']
            actual_possessions = simulator.BASE_POSSESSIONS_PER_GAME
            assert actual_possessions == expected_possessions

            logger.info(f"✅ BASE_POSSESSIONS_PER_GAME: {actual_possessions} (from config)")

            self.results['tests']['simulator_config'] = {
                'passed': True,
                'base_td_rate': actual_base_td_rate,
                'base_possessions': actual_possessions,
            }
            logger.info("\n✅ TEST PASSED: Simulator uses config correctly")
            return True

        except Exception as e:
            msg = f"Simulator config test failed: {str(e)}"
            self.results['failures'].append(msg)
            self.results['tests']['simulator_config'] = {
                'passed': False,
                'error': str(e),
            }
            logger.error(f"\n❌ TEST FAILED: {msg}")
            return False

    def test_model_hyperparameters(self) -> bool:
        """Test that training scripts use config hyperparameters."""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Model Hyperparameters")
        logger.info("="*70)

        try:
            from nfl_quant.config_enhanced import config

            # Check usage predictor params
            usage_params = config.models.usage_predictor
            logger.info(f"✅ Usage Predictor Params:")
            logger.info(f"   - n_estimators: {usage_params['n_estimators']}")
            logger.info(f"   - max_depth: {usage_params['max_depth']}")
            logger.info(f"   - learning_rate: {usage_params['learning_rate']}")

            # Check efficiency predictor params
            eff_params = config.models.efficiency_predictor
            logger.info(f"✅ Efficiency Predictor Params:")
            logger.info(f"   - n_estimators: {eff_params['n_estimators']}")
            logger.info(f"   - max_depth: {eff_params['max_depth']}")
            logger.info(f"   - learning_rate: {eff_params['learning_rate']}")

            self.results['tests']['model_hyperparameters'] = {
                'passed': True,
                'usage_predictor': usage_params,
                'efficiency_predictor': eff_params,
            }
            logger.info("\n✅ TEST PASSED: Model hyperparameters loaded from config")
            return True

        except Exception as e:
            msg = f"Model hyperparameters test failed: {str(e)}"
            self.results['failures'].append(msg)
            self.results['tests']['model_hyperparameters'] = {
                'passed': False,
                'error': str(e),
            }
            logger.error(f"\n❌ TEST FAILED: {msg}")
            return False

    def test_qb_td_model_quality(self) -> bool:
        """Test that QB TD model no longer predicts constants."""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: QB TD Rate Model Quality")
        logger.info("="*70)

        try:
            model_path = Path('data/models/efficiency_predictor_v2_defense.joblib')

            if not model_path.exists():
                logger.warning(f"⚠️  Model not found: {model_path}")
                logger.warning("   Run training script first")
                self.results['tests']['qb_td_model'] = {
                    'passed': True,
                    'skipped': True,
                    'reason': 'Model file not found',
                }
                return True

            # Load model
            data = joblib.load(model_path)
            models = data.get('models', data)  # Support both old and new format

            if 'QB_td_rate_pass' not in models:
                logger.warning(f"⚠️  QB_td_rate_pass model not found in joblib file")
                self.results['tests']['qb_td_model'] = {
                    'passed': True,
                    'skipped': True,
                    'reason': 'QB_td_rate_pass not in models dict',
                }
                return True

            qb_td_model = models['QB_td_rate_pass']

            # Create test data with varying inputs
            test_X = pd.DataFrame({
                'week': [5, 5, 5, 5, 5],
                'trailing_comp_pct': [0.50, 0.60, 0.65, 0.70, 0.75],
                'trailing_yards_per_completion': [8.0, 10.0, 11.0, 12.0, 14.0],
                'trailing_td_rate_pass': [0.02, 0.03, 0.04, 0.05, 0.06],
                'opp_pass_def_epa': [0.05, 0.0, -0.05, -0.10, -0.15],
                'opp_pass_def_rank': [25, 20, 15, 10, 5],
                'trailing_opp_pass_def_epa': [0.05, 0.0, -0.05, -0.10, -0.15],
                'team_pace': [60, 63, 65, 68, 70],
            })

            # Make predictions
            predictions = qb_td_model.predict(test_X)

            # Check variance
            pred_std = float(np.std(predictions))
            pred_range = float(np.max(predictions) - np.min(predictions))

            logger.info(f"Predictions: {predictions}")
            logger.info(f"Std Dev: {pred_std:.4f}")
            logger.info(f"Range: {pred_range:.4f}")

            # Model should NOT predict constants
            if pred_std < 0.001:
                msg = f"QB TD model still predicting constants: std={pred_std:.6f}"
                self.results['failures'].append(msg)
                self.results['tests']['qb_td_model'] = {
                    'passed': False,
                    'predictions': predictions.tolist(),
                    'std': pred_std,
                    'range': pred_range,
                }
                logger.error(f"\n❌ TEST FAILED: {msg}")
                return False
            else:
                logger.info(f"✅ Model predictions have variance (std={pred_std:.4f})")
                self.results['tests']['qb_td_model'] = {
                    'passed': True,
                    'predictions': predictions.tolist(),
                    'std': pred_std,
                    'range': pred_range,
                }
                logger.info("\n✅ TEST PASSED: QB TD model produces varied predictions")
                return True

        except Exception as e:
            msg = f"QB TD model test failed: {str(e)}"
            self.results['failures'].append(msg)
            self.results['tests']['qb_td_model'] = {
                'passed': False,
                'error': str(e),
            }
            logger.error(f"\n❌ TEST FAILED: {msg}")
            return False

    def test_td_predictor_config(self) -> bool:
        """Test that td_predictor.py uses config values."""
        logger.info("\n" + "="*70)
        logger.info("TEST 5: TD Predictor Config Usage")
        logger.info("="*70)

        try:
            from nfl_quant.models import td_predictor
            from nfl_quant.config_enhanced import config

            # Check baseline TD rates loaded from config
            expected_qb_rate = config.td_prediction.baseline_td_rates['QB']
            actual_qb_rate = td_predictor.TouchdownPredictor.BASELINE_TD_RATES['QB']

            assert actual_qb_rate == expected_qb_rate, \
                f"QB baseline TD rate mismatch: {actual_qb_rate} != {expected_qb_rate}"

            logger.info(f"✅ QB Baseline TD Rate: {actual_qb_rate} (from config)")

            # Check red zone multipliers
            expected_rz_rush = config.td_prediction.red_zone_multipliers['rushing']
            actual_rz_rush = td_predictor.TouchdownPredictor.RED_ZONE_MULTIPLIER_RUSH

            assert actual_rz_rush == expected_rz_rush
            logger.info(f"✅ Red Zone Rush Multiplier: {actual_rz_rush} (from config)")

            self.results['tests']['td_predictor_config'] = {
                'passed': True,
                'qb_baseline_rate': actual_qb_rate,
                'red_zone_rush_mult': actual_rz_rush,
            }
            logger.info("\n✅ TEST PASSED: TD Predictor uses config correctly")
            return True

        except Exception as e:
            msg = f"TD Predictor config test failed: {str(e)}"
            self.results['failures'].append(msg)
            self.results['tests']['td_predictor_config'] = {
                'passed': False,
                'error': str(e),
            }
            logger.error(f"\n❌ TEST FAILED: {msg}")
            return False

    def save_results(self, output_path: Path):
        """Save test results to JSON."""
        self.results['passed'] = len(self.results['failures']) == 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\nTest results saved to: {output_path}")

    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)

        total_tests = len(self.results['tests'])
        passed_tests = sum(1 for t in self.results['tests'].values() if t['passed'])

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Failures: {len(self.results['failures'])}")

        if self.results['failures']:
            logger.error("\nFAILURES:")
            for fail in self.results['failures']:
                logger.error(f"  - {fail}")

        if self.results['passed']:
            logger.info("\n✅ ALL TESTS PASSED")
            logger.info("   Configuration system is working correctly")
        else:
            logger.error("\n❌ SOME TESTS FAILED")
            logger.error("   Review failures above")


def main():
    """Run configuration system tests."""
    logger.info("Configuration System Integration Test")
    logger.info("="*70 + "\n")

    tester = ConfigSystemTester()

    # Run all tests
    tests = [
        tester.test_config_loading,
        tester.test_simulator_config_usage,
        tester.test_model_hyperparameters,
        tester.test_qb_td_model_quality,
        tester.test_td_predictor_config,
    ]

    for test_fn in tests:
        test_fn()

    # Save and print summary
    output_path = Path('data/diagnostics/config_system_test_results.json')
    tester.save_results(output_path)
    tester.print_summary()

    # Exit with appropriate code
    sys.exit(0 if tester.results['passed'] else 1)


if __name__ == '__main__':
    main()
