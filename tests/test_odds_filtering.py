"""Unit tests for pre-game odds filtering system."""

import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.utils.odds import (
    load_game_status_map,
    is_valid_pregame_odds,
    filter_pregame_odds
)


class TestGameStatusMap(unittest.TestCase):
    """Test game status mapping from nflverse data."""

    def test_load_game_status_map_valid_week(self):
        """Test loading game status for a valid week."""
        # Week 8 should have pre_game games (current week in data)
        status_map = load_game_status_map(week=8)

        self.assertIsInstance(status_map, dict)
        # Should have some games for week 8
        self.assertGreater(len(status_map), 0)

        # Check format of game_ids and statuses
        for game_id, status in status_map.items():
            self.assertIsInstance(game_id, str)
            self.assertIn(status, ['pre_game', 'in_progress', 'complete'])

    def test_load_game_status_map_completed_week(self):
        """Test loading game status for a completed week."""
        # Week 1 should have all complete games
        status_map = load_game_status_map(week=1)

        self.assertGreater(len(status_map), 0)

        # All games should be complete
        statuses = set(status_map.values())
        self.assertEqual(statuses, {'complete'})

    def test_load_game_status_map_invalid_week(self):
        """Test loading game status for invalid week."""
        # Week 99 should return empty dict
        status_map = load_game_status_map(week=99)

        self.assertIsInstance(status_map, dict)
        self.assertEqual(len(status_map), 0)


class TestIsValidPregameOdds(unittest.TestCase):
    """Test individual odds validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.game_status_map = {
            '202510802': 'pre_game',
            '202510803': 'in_progress',
            '202510804': 'complete'
        }
        self.current_time = datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc)

    def test_valid_pregame_odds(self):
        """Test validation of valid pre-game odds."""
        row = pd.Series({
            'game_id': '202510802',
            'commence_time': '2025-10-26T17:00:00Z',  # 5 hours in future
            'last_update': '2025-10-26T11:00:00Z'  # 1 hour ago
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time
        )

        self.assertTrue(is_valid)
        self.assertEqual(reason, 'valid_pregame')

    def test_in_progress_game_rejected(self):
        """Test that in-progress games are rejected."""
        row = pd.Series({
            'game_id': '202510803',
            'commence_time': '2025-10-26T12:00:00Z',
            'last_update': '2025-10-26T11:00:00Z'
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time
        )

        self.assertFalse(is_valid)
        self.assertEqual(reason, 'game_status=in_progress')

    def test_completed_game_rejected(self):
        """Test that completed games are rejected."""
        row = pd.Series({
            'game_id': '202510804',
            'commence_time': '2025-10-25T17:00:00Z',  # Yesterday
            'last_update': '2025-10-25T16:00:00Z'
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time
        )

        self.assertFalse(is_valid)
        self.assertEqual(reason, 'game_status=complete')

    def test_game_already_started_rejected(self):
        """Test that odds for started games are rejected."""
        row = pd.Series({
            'game_id': '999999999',  # Unknown game_id
            'commence_time': '2025-10-26T11:00:00Z',  # 1 hour ago
            'last_update': '2025-10-26T10:00:00Z'
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time
        )

        self.assertFalse(is_valid)
        self.assertIn('game_already_started', reason)

    def test_too_close_to_kickoff_rejected(self):
        """Test that odds too close to kickoff are rejected."""
        row = pd.Series({
            'game_id': '999999999',
            'commence_time': '2025-10-26T12:03:00Z',  # 3 minutes in future
            'last_update': '2025-10-26T11:00:00Z'
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time,
            min_minutes_before_kickoff=5
        )

        self.assertFalse(is_valid)
        self.assertIn('too_close_to_kickoff', reason)

    def test_stale_odds_rejected(self):
        """Test that stale odds are rejected."""
        row = pd.Series({
            'game_id': '999999999',
            'commence_time': '2025-10-27T17:00:00Z',  # Tomorrow
            'last_update': '2025-10-24T12:00:00Z'  # 48 hours ago
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time,
            max_hours_stale=24
        )

        self.assertFalse(is_valid)
        self.assertIn('stale_odds', reason)

    def test_missing_commence_time(self):
        """Test handling of missing commence_time."""
        row = pd.Series({
            'game_id': '202510802',
            'last_update': '2025-10-26T11:00:00Z'
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            self.game_status_map,
            self.current_time
        )

        # Should pass if game status is pre_game
        self.assertTrue(is_valid)
        self.assertEqual(reason, 'valid_pregame')


class TestFilterPregameOdds(unittest.TestCase):
    """Test the main filtering function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test odds DataFrame
        self.test_odds = pd.DataFrame([
            {
                'game_id': '202510802',
                'player_name': 'Josh Allen',
                'market': 'player_pass_yds',
                'line': 262.5,
                'commence_time': '2025-10-26T17:00:00Z',
                'last_update': '2025-10-26T11:00:00Z',
                'odds': -110
            },
            {
                'game_id': '202510803',  # in_progress
                'player_name': 'Patrick Mahomes',
                'market': 'player_pass_yds',
                'line': 275.5,
                'commence_time': '2025-10-26T13:00:00Z',
                'last_update': '2025-10-26T12:00:00Z',
                'odds': -110
            },
            {
                'game_id': '202510804',  # complete
                'player_name': 'Lamar Jackson',
                'market': 'player_pass_yds',
                'line': 245.5,
                'commence_time': '2025-10-25T17:00:00Z',
                'last_update': '2025-10-25T16:00:00Z',
                'odds': -110
            }
        ])

    def test_filter_pregame_odds_basic(self):
        """Test basic filtering functionality."""
        current_time = datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc)

        filtered = filter_pregame_odds(
            self.test_odds,
            week=8,
            current_time=current_time
        )

        # Should filter out in_progress and complete games
        # May keep 1-2 valid depending on game_id matching
        self.assertGreater(len(filtered), 0)
        self.assertLess(len(filtered), len(self.test_odds))

        # Verify Josh Allen is in the filtered results
        self.assertIn('Josh Allen', filtered['player_name'].values)

    def test_filter_empty_dataframe(self):
        """Test filtering of empty DataFrame."""
        empty_df = pd.DataFrame()

        filtered = filter_pregame_odds(
            empty_df,
            week=8
        )

        self.assertTrue(filtered.empty)

    def test_filter_removes_temp_columns(self):
        """Test that temporary validation columns are removed."""
        current_time = datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc)

        filtered = filter_pregame_odds(
            self.test_odds,
            week=8,
            current_time=current_time
        )

        # Temporary columns should not be in result
        self.assertNotIn('is_valid_pregame', filtered.columns)
        self.assertNotIn('rejection_reason', filtered.columns)

    def test_filter_preserves_original_columns(self):
        """Test that original columns are preserved."""
        current_time = datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc)

        original_columns = set(self.test_odds.columns)

        filtered = filter_pregame_odds(
            self.test_odds,
            week=8,
            current_time=current_time
        )

        # All original columns should be preserved
        self.assertTrue(original_columns.issubset(set(filtered.columns)))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_malformed_commence_time(self):
        """Test handling of malformed commence_time."""
        game_status_map = {'999999999': 'pre_game'}
        current_time = datetime.now(timezone.utc)

        row = pd.Series({
            'game_id': '999999999',
            'commence_time': 'invalid-date-format',
            'last_update': current_time.isoformat()
        })

        is_valid, reason = is_valid_pregame_odds(
            row,
            game_status_map,
            current_time
        )

        # Should still validate based on game status
        self.assertTrue(is_valid)

    def test_timezone_handling(self):
        """Test correct handling of timezone-aware and naive datetimes."""
        game_status_map = {}
        current_time = datetime(2025, 10, 26, 12, 0, tzinfo=timezone.utc)

        # Test with Z suffix
        row1 = pd.Series({
            'game_id': '999999999',
            'commence_time': '2025-10-26T17:00:00Z',
        })

        is_valid1, _ = is_valid_pregame_odds(row1, game_status_map, current_time)
        self.assertTrue(is_valid1)

        # Test with +00:00 suffix
        row2 = pd.Series({
            'game_id': '999999999',
            'commence_time': '2025-10-26T17:00:00+00:00',
        })

        is_valid2, _ = is_valid_pregame_odds(row2, game_status_map, current_time)
        self.assertTrue(is_valid2)


if __name__ == '__main__':
    unittest.main()
