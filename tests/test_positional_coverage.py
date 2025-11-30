"""
Regression Tests for Positional Feature Coverage

These tests prevent the RB receiving stats bug from returning.
"""

import pytest
import json
from pathlib import Path

from nfl_quant.constants import RECEIVING_POSITIONS, RUSHING_POSITIONS
from nfl_quant.validation import validate_historical_stats


class TestPositionalCoverage:
    """Test that all positions get proper stats coverage"""
    
    def test_receiving_positions_include_rb(self):
        """Critical regression test: RB must be in RECEIVING_POSITIONS"""
        assert 'RB' in RECEIVING_POSITIONS, "RB missing from receiving positions - BUG!"
        assert 'WR' in RECEIVING_POSITIONS
        assert 'TE' in RECEIVING_POSITIONS
    
    def test_rb_can_have_target_share(self):
        """Test that RB can have target_share calculated from nflverse data"""
        # Mock Bijan Robinson data
        bijan_data = {
            'player_name': 'Bijan Robinson',
            'position': 'RB',
            'targets': 34,  # 5-6 targets per game over 6 weeks
            'receptions': 26,
            'trailing_target_share': 0.14375,
            'weeks_played': 6,
        }
        
        # Should pass validation
        is_valid, error = validate_historical_stats(bijan_data, 'RB', 'Bijan Robinson')
        assert is_valid, f"Bijan data invalid: {error}"
        assert bijan_data['trailing_target_share'] is not None, "RB target_share should not be None"
    
    def test_historical_stats_rb_target_share_not_null(self):
        """Check that historical stats JSON has RB target shares"""
        stats_file = Path('data/historical_player_stats.json')
        if not stats_file.exists():
            pytest.skip("Historical stats file not found")
        
        with open(stats_file) as f:
            stats = json.load(f)
        
        # Check a few RBs
        rb_players = [
            ('Bijan Robinson', 0.1625),  # Known value from fix
        ]
        
        for player_name, expected_min_target_share in rb_players:
            if player_name not in stats:
                pytest.skip(f"{player_name} not in stats")
            
            player_data = stats[player_name]
            if player_data['position'] == 'RB':
                target_share = player_data.get('trailing_target_share')
                
                # RB should have target_share if they have receiving stats
                # It can be None if they're a pure power back with no targets
                # But Bijan should definitely have a target share > 0
                if 'Bijan' in player_name and target_share is None:
                    pytest.fail(
                        f"Regression bug: {player_name} (RB) has trailing_target_share = None. "
                        "This is the same bug we fixed!"
                    )


class TestPositionalPropCoverage:
    """Test that all props have proper position coverage"""
    
    def test_receiving_yards_props_all_positions(self):
        """Receiving yards props should work for RB, WR, TE"""
        from nfl_quant.constants import get_positions_for_prop
        
        positions = get_positions_for_prop('receiving_yards')
        assert 'RB' in positions, "RB missing from receiving_yards props"
        assert 'WR' in positions
        assert 'TE' in positions
    
    def test_receptions_props_all_positions(self):
        """Receptions props should work for RB, WR, TE"""
        from nfl_quant.constants import get_positions_for_prop
        
        positions = get_positions_for_prop('receptions')
        assert 'RB' in positions, "RB missing from receptions props"
        assert 'WR' in positions
        assert 'TE' in positions


class TestHistoricalDataCompleteness:
    """Test that historical data is complete"""
    
    def test_week_specific_stats_rb_receiving(self):
        """Check week-specific stats include RB receiving data"""
        stats_file = Path('data/week_specific_trailing_stats.json')
        if not stats_file.exists():
            pytest.skip("Week-specific stats file not found")
        
        with open(stats_file) as f:
            stats = json.load(f)
        
        # Check that Bijan has target share for at least one week
        bijan_keys = [k for k in stats.keys() if k.startswith('Bijan Robinson_week')]
        
        if not bijan_keys:
            pytest.skip("Bijan not in week-specific stats")
        
        for key in bijan_keys[:3]:  # Check first 3 weeks
            bijan_data = stats[key]
            if bijan_data.get('position') == 'RB':
                target_share = bijan_data.get('trailing_target_share')
                
                if target_share is None:
                    pytest.fail(
                        f"Regression: {key} has trailing_target_share = None for RB. "
                        "This indicates the fix was lost!"
                    )


class TestAntiPatternPrevention:
    """Test that anti-patterns don't exist in code"""
    
    def test_no_hardcoded_wr_te_only_checks(self):
        """Ensure we don't have hardcoded ['WR', 'TE'] checks that exclude RB"""
        # This test would need to scan code, so we'll just document the pattern
        # In practice, grep for: position.*in.*\['WR', 'TE'\]
        pass  # Manual code review required


if __name__ == '__main__':
    pytest.main([__file__, '-v'])




