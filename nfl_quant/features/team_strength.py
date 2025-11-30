"""
Team Strength Features - Enhanced Elo from NFLverse
===================================================
Generalized Elo system that works for any season/week.

Features:
- Margin of victory multiplier (diminishing returns)
- Home field advantage
- Bye week bonus
- Season carryover with regression
- Self-validating checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import math
import logging

logger = logging.getLogger(__name__)


class EnhancedEloCalculator:
    """
    Calculate NFL Elo ratings using FiveThirtyEight methodology.
    Works for any season/week combination.
    """

    # FiveThirtyEight NFL Elo parameters
    K_FACTOR = 20
    HOME_ADVANTAGE = 48
    BYE_BONUS = 25
    MEAN_ELO = 1505
    SEASON_REVERSION = 1/3
    BYE_REST_THRESHOLD = 10  # Days of rest that indicates bye week

    def __init__(self, schedules_path: str = 'data/nflverse/schedules.parquet'):
        self.schedules = pd.read_parquet(schedules_path)
        self.elo_history: Dict[Tuple[int, int, str], float] = {}
        self._current_elo: Dict[str, float] = {}
        self._cache: Dict[Tuple[int, int], Dict[str, float]] = {}  # (season, week) -> elo dict

    def _get_available_seasons(self) -> List[int]:
        """Get list of seasons with data."""
        return sorted(self.schedules['season'].unique())

    def _get_completed_weeks(self, season: int) -> List[int]:
        """Get list of completed weeks for a season."""
        completed = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['home_score'].notna()) &
            (self.schedules['game_type'] == 'REG')
        ]['week'].unique()
        return sorted(completed)

    def _get_teams_for_season(self, season: int) -> set:
        """Get all teams that played in a season."""
        home = set(self.schedules[self.schedules['season'] == season]['home_team'].unique())
        away = set(self.schedules[self.schedules['season'] == season]['away_team'].unique())
        return home | away

    def _margin_of_victory_multiplier(self, point_diff: int,
                                       winner_elo: float, loser_elo: float) -> float:
        """
        FiveThirtyEight MOV multiplier with autocorrelation adjustment.
        """
        if point_diff == 0:
            return 1.0

        base = math.log(abs(point_diff) + 1)
        elo_diff = winner_elo - loser_elo
        autocorr = 2.2 / (elo_diff * 0.001 + 2.2)

        return base * autocorr

    def _get_pregame_adjustment(self, is_home: bool, team_rest: Optional[int],
                                 opp_rest: Optional[int], location: str) -> float:
        """Calculate pregame Elo adjustment for context factors."""
        adjustment = 0.0

        # Home field advantage (unless neutral site)
        if is_home and location != 'Neutral':
            adjustment += self.HOME_ADVANTAGE

        # Bye week bonus
        team_rest = team_rest or 7
        opp_rest = opp_rest or 7

        if team_rest >= self.BYE_REST_THRESHOLD:
            adjustment += self.BYE_BONUS
        if opp_rest >= self.BYE_REST_THRESHOLD:
            adjustment -= self.BYE_BONUS

        return adjustment

    def _initialize_season(self, season: int) -> Dict[str, float]:
        """Initialize Elo ratings for a new season with regression to mean."""
        teams = self._get_teams_for_season(season)
        available_seasons = self._get_available_seasons()

        if season == min(available_seasons):
            return {team: self.MEAN_ELO for team in teams}

        # Get end-of-previous-season ratings
        prev_season = season - 1
        prev_elos = {}

        for team in teams:
            prev_games = self.schedules[
                (self.schedules['season'] == prev_season) &
                ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team)) &
                (self.schedules['home_score'].notna())
            ].sort_values('week')

            if len(prev_games) > 0:
                last_week = prev_games['week'].max()
                prev_elo = self.elo_history.get((prev_season, last_week, team), self.MEAN_ELO)
            else:
                prev_elo = self.MEAN_ELO

            # Regress toward mean
            prev_elos[team] = prev_elo * (1 - self.SEASON_REVERSION) + self.MEAN_ELO * self.SEASON_REVERSION

        return prev_elos

    def _process_game(self, game: pd.Series, season: int):
        """Process a single game and update Elo ratings."""
        home = game['home_team']
        away = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']

        home_elo = self._current_elo.get(home, self.MEAN_ELO)
        away_elo = self._current_elo.get(away, self.MEAN_ELO)

        home_rest = game.get('home_rest', 7)
        away_rest = game.get('away_rest', 7)
        location = game.get('location', 'Home')

        home_adj = self._get_pregame_adjustment(True, home_rest, away_rest, location)
        away_adj = self._get_pregame_adjustment(False, away_rest, home_rest, location)

        home_elo_adj = home_elo + home_adj
        away_elo_adj = away_elo + away_adj

        home_expected = 1 / (1 + 10 ** ((away_elo_adj - home_elo_adj) / 400))
        away_expected = 1 - home_expected

        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
            winner_elo, loser_elo = home_elo_adj, away_elo_adj
        elif away_score > home_score:
            home_actual, away_actual = 0.0, 1.0
            winner_elo, loser_elo = away_elo_adj, home_elo_adj
        else:
            home_actual, away_actual = 0.5, 0.5
            winner_elo, loser_elo = home_elo_adj, away_elo_adj

        point_diff = abs(home_score - away_score)
        mov_mult = self._margin_of_victory_multiplier(point_diff, winner_elo, loser_elo)

        home_delta = self.K_FACTOR * mov_mult * (home_actual - home_expected)
        away_delta = self.K_FACTOR * mov_mult * (away_actual - away_expected)

        self._current_elo[home] = home_elo + home_delta
        self._current_elo[away] = away_elo + away_delta

        week = game['week']
        self.elo_history[(season, week, home)] = self._current_elo[home]
        self.elo_history[(season, week, away)] = self._current_elo[away]

    def calculate_elo_through_week(self, season: int, week: int,
                                    use_cache: bool = True) -> Dict[str, float]:
        """
        Calculate Elo ratings for all teams entering a specific week.
        Uses only data from BEFORE the specified week (no leakage).
        """
        cache_key = (season, week)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Need to process from beginning to build history
        available_seasons = self._get_available_seasons()

        # Process all prior seasons first (for carryover)
        for prev_season in available_seasons:
            if prev_season >= season:
                break
            self._current_elo = self._initialize_season(prev_season)
            games = self.schedules[
                (self.schedules['season'] == prev_season) &
                (self.schedules['game_type'] == 'REG') &
                (self.schedules['home_score'].notna())
            ].sort_values(['week', 'gameday'])

            for _, game in games.iterrows():
                self._process_game(game, prev_season)

        # Now process current season up to (but not including) target week
        self._current_elo = self._initialize_season(season)

        games = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] < week) &
            (self.schedules['game_type'] == 'REG') &
            (self.schedules['home_score'].notna())
        ].sort_values(['week', 'gameday'])

        for _, game in games.iterrows():
            self._process_game(game, season)

        self._cache[cache_key] = self._current_elo.copy()
        return self._current_elo.copy()

    def get_team_record(self, team: str, season: int, through_week: int) -> Dict:
        """Get team's W-L record through a specific week (exclusive)."""
        games = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] < through_week) &
            (self.schedules['game_type'] == 'REG') &
            ((self.schedules['home_team'] == team) | (self.schedules['away_team'] == team)) &
            (self.schedules['home_score'].notna())
        ]

        wins, losses, ties = 0, 0, 0
        points_for, points_against = 0, 0

        for _, game in games.iterrows():
            is_home = game['home_team'] == team
            team_score = game['home_score'] if is_home else game['away_score']
            opp_score = game['away_score'] if is_home else game['home_score']

            points_for += team_score
            points_against += opp_score

            if team_score > opp_score:
                wins += 1
            elif team_score < opp_score:
                losses += 1
            else:
                ties += 1

        games_played = wins + losses + ties

        return {
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'games_played': games_played,
            'win_pct': wins / games_played if games_played > 0 else 0.5,
            'points_for': points_for,
            'points_against': points_against,
            'point_diff': points_for - points_against,
            'point_diff_per_game': (points_for - points_against) / games_played if games_played > 0 else 0
        }

    def get_team_features(self, team: str, opponent: str, season: int, week: int,
                          is_home: bool = True) -> Dict:
        """Get all team strength features for a matchup."""
        elo_ratings = self.calculate_elo_through_week(season, week)

        team_elo = elo_ratings.get(team, self.MEAN_ELO)
        opp_elo = elo_ratings.get(opponent, self.MEAN_ELO)

        # Get game context from schedule
        game = self.schedules[
            (self.schedules['season'] == season) &
            (self.schedules['week'] == week) &
            (((self.schedules['home_team'] == team) & (self.schedules['away_team'] == opponent)) |
             ((self.schedules['home_team'] == opponent) & (self.schedules['away_team'] == team)))
        ]

        if len(game) > 0:
            game = game.iloc[0]
            actual_is_home = game['home_team'] == team
            team_rest = game['home_rest'] if actual_is_home else game['away_rest']
            opp_rest = game['away_rest'] if actual_is_home else game['home_rest']
            location = game.get('location', 'Home')
            is_home = actual_is_home  # Override with actual
        else:
            team_rest, opp_rest = 7, 7
            location = 'Home' if is_home else 'Away'

        team_adj = self._get_pregame_adjustment(is_home, team_rest, opp_rest, location)
        opp_adj = self._get_pregame_adjustment(not is_home, opp_rest, team_rest, location)

        team_elo_adj = team_elo + team_adj
        opp_elo_adj = opp_elo + opp_adj

        win_prob = 1 / (1 + 10 ** ((opp_elo_adj - team_elo_adj) / 400))
        expected_spread = (team_elo_adj - opp_elo_adj) / 25

        # Get records
        team_record = self.get_team_record(team, season, week)
        opp_record = self.get_team_record(opponent, season, week)

        return {
            'team_elo': team_elo,
            'opp_elo': opp_elo,
            'team_elo_adjusted': team_elo_adj,
            'opp_elo_adjusted': opp_elo_adj,
            'elo_diff': team_elo - opp_elo,
            'elo_diff_adjusted': team_elo_adj - opp_elo_adj,
            'win_probability': win_prob,
            'expected_spread': expected_spread,
            'team_win_pct': team_record['win_pct'],
            'opp_win_pct': opp_record['win_pct'],
            'team_point_diff_per_game': team_record['point_diff_per_game'],
            'opp_point_diff_per_game': opp_record['point_diff_per_game'],
            'team_rest': team_rest,
            'opp_rest': opp_rest,
            'is_home': is_home
        }

    def get_all_team_ratings(self, season: int, week: int) -> pd.DataFrame:
        """Get rankings table for all teams entering a week."""
        elo_ratings = self.calculate_elo_through_week(season, week)
        teams = self._get_teams_for_season(season)

        records = []
        for team in teams:
            elo = elo_ratings.get(team, self.MEAN_ELO)
            record = self.get_team_record(team, season, week)

            records.append({
                'team': team,
                'elo': elo,
                'wins': record['wins'],
                'losses': record['losses'],
                'record': f"{record['wins']}-{record['losses']}",
                'point_diff': record['point_diff'],
                'ppg_diff': record['point_diff_per_game']
            })

        df = pd.DataFrame(records)
        df = df.sort_values('elo', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        return df[['rank', 'team', 'elo', 'record', 'wins', 'losses', 'point_diff', 'ppg_diff']]

    def validate(self, season: int, week: int) -> Dict:
        """
        Self-validating checks for any season/week.
        Returns validation results without hardcoded expectations.
        """
        results = {
            'season': season,
            'week': week,
            'checks': [],
            'all_passed': True
        }

        rankings = self.get_all_team_ratings(season, week)

        # Check 1: Elo correlates with win percentage
        correlation = rankings['elo'].corr(rankings['wins'])
        check1_pass = correlation > 0.7
        results['checks'].append({
            'name': 'Elo correlates with wins',
            'passed': check1_pass,
            'value': f"r={correlation:.2f}",
            'expected': 'r > 0.70'
        })
        if not check1_pass:
            results['all_passed'] = False

        # Check 2: Top 5 Elo teams have winning records
        top5 = rankings.head(5)
        top5_winning = (top5['wins'] > top5['losses']).all()
        results['checks'].append({
            'name': 'Top 5 Elo teams have winning records',
            'passed': top5_winning,
            'value': top5[['team', 'record']].to_dict('records'),
            'expected': 'All W > L'
        })
        if not top5_winning:
            results['all_passed'] = False

        # Check 3: Bottom 5 Elo teams have losing records (if enough games played)
        bottom5 = rankings.tail(5)
        min_games = rankings['wins'].add(rankings['losses']).min()
        if min_games >= 3:
            bottom5_losing = (bottom5['wins'] < bottom5['losses']).all()
            results['checks'].append({
                'name': 'Bottom 5 Elo teams have losing records',
                'passed': bottom5_losing,
                'value': bottom5[['team', 'record']].to_dict('records'),
                'expected': 'All W < L'
            })
            if not bottom5_losing:
                results['all_passed'] = False

        # Check 4: Elo spread makes sense (average should be near MEAN_ELO)
        avg_elo = rankings['elo'].mean()
        elo_check = abs(avg_elo - self.MEAN_ELO) < 50
        results['checks'].append({
            'name': 'Average Elo near expected mean',
            'passed': elo_check,
            'value': f"{avg_elo:.0f}",
            'expected': f"{self.MEAN_ELO} ± 50"
        })
        if not elo_check:
            results['all_passed'] = False

        # Check 5: Elo range is reasonable (spread of ~400-600 points typical)
        elo_range = rankings['elo'].max() - rankings['elo'].min()
        range_check = 200 < elo_range < 800
        results['checks'].append({
            'name': 'Elo range is reasonable',
            'passed': range_check,
            'value': f"{elo_range:.0f}",
            'expected': '200-800'
        })
        if not range_check:
            results['all_passed'] = False

        # Check 6: Point differential correlates with Elo
        ppg_corr = rankings['elo'].corr(rankings['ppg_diff'])
        ppg_check = ppg_corr > 0.8
        results['checks'].append({
            'name': 'Elo correlates with point differential',
            'passed': ppg_check,
            'value': f"r={ppg_corr:.2f}",
            'expected': 'r > 0.80'
        })
        if not ppg_check:
            results['all_passed'] = False

        return results

    def print_validation_report(self, season: int, week: int):
        """Print a formatted validation report."""
        results = self.validate(season, week)

        print(f"\n{'='*60}")
        print(f"ELO VALIDATION REPORT - Season {season}, Week {week}")
        print(f"{'='*60}\n")

        for check in results['checks']:
            status = "✅" if check['passed'] else "❌"
            print(f"{status} {check['name']}")
            print(f"   Got: {check['value']}")
            print(f"   Expected: {check['expected']}\n")

        print(f"{'='*60}")
        print(f"OVERALL: {'✅ ALL PASSED' if results['all_passed'] else '❌ SOME FAILED'}")
        print(f"{'='*60}\n")

        return results['all_passed']
