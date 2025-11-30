"""
Regime Detection CLI Interface

Command-line tool for regime detection, analysis, and reporting.

Commands:
- detect: Detect regimes for teams
- analyze: Analyze specific player in regime context
- project: Generate regime-adjusted projections
- report: Create comprehensive reports
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import json

from .detector import RegimeDetector
from .metrics import RegimeMetricsCalculator
from .projections import RegimeAwareProjector
from .reports import RegimeReportGenerator
from ..data.fetcher import DataFetcher


class RegimeCLI:
    """Command-line interface for regime detection."""

    def __init__(self):
        self.detector = RegimeDetector()
        self.calculator = RegimeMetricsCalculator()
        self.projector = RegimeAwareProjector(
            detector=self.detector,
            calculator=self.calculator,
        )
        self.reporter = RegimeReportGenerator()
        self.fetcher = DataFetcher()

    def detect(
        self,
        team: Optional[str],
        week: int,
        season: int,
        output: Optional[str] = None,
    ) -> None:
        """
        Detect regimes for team(s).

        Args:
            team: Team abbreviation (or None for all teams)
            week: Current week
            season: Season year
            output: Output path for JSON results (optional)
        """
        print(f"\n🔍 Detecting regime changes for {team or 'all teams'} (Week {week}, {season})")

        # Load data
        print("Loading play-by-play data...")
        pbp_df = self.fetcher.load_pbp_parquet(season)

        print("Loading player stats...")
        player_stats_df = self._load_player_stats(season, week)

        teams_to_check = [team] if team else self._get_all_teams(pbp_df)

        results = {}

        for t in teams_to_check:
            print(f"\nAnalyzing {t}...")

            result = self.detector.detect_all_regimes(
                team=t,
                current_week=week,
                season=season,
                pbp_df=pbp_df,
                player_stats_df=player_stats_df,
            )

            results[t] = result

            # Print summary
            if result.has_active_regime:
                regime = result.active_regime
                print(f"  ✓ Regime detected: {regime.trigger.description}")
                print(f"    Type: {regime.trigger.type.value}")
                print(f"    Start: Week {regime.start_week}")
                print(f"    Games: {regime.games_in_regime}")
                print(f"    Confidence: {regime.trigger.confidence:.0%}")
                print(f"    Affected players: {len(regime.affected_players)}")
            else:
                print(f"  ○ No regime change detected (stable)")

        # Export if requested
        if output:
            self.reporter.export_to_json(results, output)
            print(f"\n✓ Results exported to {output}")

    def analyze(
        self,
        player: str,
        week: int,
        season: int,
        opponent: Optional[str] = None,
        output: Optional[str] = None,
    ) -> None:
        """
        Analyze specific player in regime context.

        Args:
            player: Player name
            week: Current week
            season: Season year
            opponent: Opponent team (optional)
            output: Output path for markdown report (optional)
        """
        print(f"\n📊 Analyzing {player} (Week {week}, {season})")

        # Load data
        pbp_df = self.fetcher.load_pbp_parquet(season)
        player_stats_df = self._load_player_stats(season, week)

        # Find player's team and position
        player_info = player_stats_df[
            player_stats_df["player_name"] == player
        ].iloc[0]

        team = player_info["team"]
        position = player_info["position"]
        player_id = player_info.get("player_id", "")

        print(f"Team: {team}, Position: {position}")

        # Get regime-adjusted stats
        stats = self.projector.get_regime_specific_stats(
            player_name=player,
            player_id=player_id,
            position=position,
            team=team,
            current_week=week,
            season=season,
            pbp_df=pbp_df,
            player_stats_df=player_stats_df,
        )

        # Print results
        print("\n=== Regime-Adjusted Trailing Stats ===")
        print(f"Snaps per game: {stats['snaps_per_game']:.1f}")
        print(f"Targets per game: {stats['targets_per_game']:.1f}")
        print(f"Carries per game: {stats['carries_per_game']:.1f}")
        print(f"Yards per target: {stats['yards_per_target']:.2f}")
        print(f"Yards per carry: {stats['yards_per_carry']:.2f}")
        print(f"Catch rate: {stats['catch_rate']:.1%}")
        print(f"Games in sample: {stats['games_played']}")

        if stats.get("regime_detected"):
            print(f"\n=== Regime Context ===")
            print(f"Regime type: {stats['regime_type']}")
            print(f"Regime start: Week {stats['regime_start_week']}")
            print(f"Weeks in regime: {stats['weeks_in_regime']}")
            print(f"Confidence: {stats['regime_confidence']:.0%}")

            if stats.get("blending_applied"):
                print(f"\nBlending: {stats['new_regime_weight']:.0%} new / {stats['old_regime_weight']:.0%} old")

        # Export if requested
        if output:
            with open(output, "w") as f:
                f.write(f"# {player} Regime Analysis\n\n")
                f.write(json.dumps(stats, indent=2))
            print(f"\n✓ Analysis exported to {output}")

    def project(
        self,
        week: int,
        season: int,
        players: Optional[str] = None,
        output: Optional[str] = None,
    ) -> None:
        """
        Generate regime-adjusted projections.

        Args:
            week: Current week
            season: Season year
            players: Path to CSV with player list (optional, processes all if None)
            output: Output path for CSV results
        """
        print(f"\n🎯 Generating regime-adjusted projections (Week {week}, {season})")

        # Load data
        pbp_df = self.fetcher.load_pbp_parquet(season)
        player_stats_df = self._load_player_stats(season, week)

        # Get player list
        if players:
            player_list_df = pd.read_csv(players)
            player_list = player_list_df.to_dict("records")
        else:
            # Use all players from stats
            player_list = player_stats_df[["player_name", "player_id", "position", "team"]].drop_duplicates().to_dict("records")

        print(f"Processing {len(player_list)} players...")

        # Batch process
        results_df = self.projector.batch_process_players(
            players=player_list,
            current_week=week,
            season=season,
            pbp_df=pbp_df,
            player_stats_df=player_stats_df,
        )

        print(f"\n✓ Processed {len(results_df)} players")

        # Print sample
        print("\nSample results:")
        print(results_df.head(10).to_string())

        # Export
        if output:
            results_df.to_csv(output, index=False)
            print(f"\n✓ Projections exported to {output}")

    def report(
        self,
        team: Optional[str],
        week: int,
        season: int,
        output_dir: str = "reports/regime",
    ) -> None:
        """
        Generate comprehensive regime reports.

        Args:
            team: Team abbreviation (or None for all teams)
            week: Current week
            season: Season year
            output_dir: Output directory for reports
        """
        print(f"\n📝 Generating regime reports (Week {week}, {season})")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        pbp_df = self.fetcher.load_pbp_parquet(season)
        player_stats_df = self._load_player_stats(season, week)

        teams_to_report = [team] if team else self._get_all_teams(pbp_df)

        for t in teams_to_report:
            print(f"\nGenerating report for {t}...")

            # Detect regimes
            result = self.detector.detect_all_regimes(
                team=t,
                current_week=week,
                season=season,
                pbp_df=pbp_df,
                player_stats_df=player_stats_df,
            )

            if not result.has_active_regime:
                print(f"  No regime detected for {t}, skipping")
                continue

            # Generate team summary
            team_summary = self.reporter.generate_team_summary(
                team=t,
                season=season,
                current_week=week,
                detection_result=result,
                player_metrics=[],  # Would populate with player metrics
                pbp_df=pbp_df,
            )

            # Generate markdown
            report_md = self.reporter.generate_team_summary_markdown(team_summary)

            # Save
            report_file = output_path / f"{t}_regime_report_week{week}.md"
            with open(report_file, "w") as f:
                f.write(report_md)

            print(f"  ✓ Report saved to {report_file}")

        print(f"\n✓ All reports generated in {output_dir}")

    def _load_player_stats(self, season: int, week: int) -> pd.DataFrame:
        """Load player stats up to specified week from NFLverse."""
        try:
            from nfl_quant.utils.nflverse_loader import load_player_stats

            # Load all stats for the season from R-fetched data
            df = load_player_stats(seasons=season)

            # Filter to weeks before current week
            df = df[df['week'] < week].copy()
            df["season"] = season

            return df
        except Exception as e:
            # Fallback: empty DataFrame
            print(f"Warning: Could not load player stats from NFLverse: {e}")
            return pd.DataFrame()

    def _get_all_teams(self, pbp_df: pd.DataFrame) -> list:
        """Get list of all teams from PBP data."""
        teams = set()
        teams.update(pbp_df["posteam"].dropna().unique())
        teams.update(pbp_df["defteam"].dropna().unique())
        return sorted(list(teams))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Regime Change Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect regime changes")
    detect_parser.add_argument("--team", "-t", help="Team abbreviation (or omit for all)")
    detect_parser.add_argument("--week", "-w", type=int, required=True, help="Current week")
    detect_parser.add_argument("--season", "-s", type=int, required=True, help="Season year")
    detect_parser.add_argument("--output", "-o", help="Output JSON path")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific player")
    analyze_parser.add_argument("--player", "-p", required=True, help="Player name")
    analyze_parser.add_argument("--week", "-w", type=int, required=True, help="Current week")
    analyze_parser.add_argument("--season", "-s", type=int, required=True, help="Season year")
    analyze_parser.add_argument("--opponent", help="Opponent team")
    analyze_parser.add_argument("--output", "-o", help="Output markdown path")

    # Project command
    project_parser = subparsers.add_parser("project", help="Generate projections")
    project_parser.add_argument("--week", "-w", type=int, required=True, help="Current week")
    project_parser.add_argument("--season", "-s", type=int, required=True, help="Season year")
    project_parser.add_argument("--players", help="CSV with player list")
    project_parser.add_argument("--output", "-o", required=True, help="Output CSV path")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--team", "-t", help="Team abbreviation (or omit for all)")
    report_parser.add_argument("--week", "-w", type=int, required=True, help="Current week")
    report_parser.add_argument("--season", "-s", type=int, required=True, help="Season year")
    report_parser.add_argument("--output-dir", "-o", default="reports/regime", help="Output directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cli = RegimeCLI()

    try:
        if args.command == "detect":
            cli.detect(
                team=args.team,
                week=args.week,
                season=args.season,
                output=args.output,
            )
        elif args.command == "analyze":
            cli.analyze(
                player=args.player,
                week=args.week,
                season=args.season,
                opponent=args.opponent,
                output=args.output,
            )
        elif args.command == "project":
            cli.project(
                week=args.week,
                season=args.season,
                players=args.players,
                output=args.output,
            )
        elif args.command == "report":
            cli.report(
                team=args.team,
                week=args.week,
                season=args.season,
                output_dir=args.output_dir,
            )

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
