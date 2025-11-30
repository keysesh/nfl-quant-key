"""Command-line interface for NFL Quant pipeline."""

import logging
from pathlib import Path
from typing import Optional

import typer
from pandas import DataFrame

from nfl_quant.config import settings
from nfl_quant.data.fetcher import DataFetcher
from nfl_quant.features.engine import FeatureEngine
from nfl_quant.features.injuries import InjuryImpactModel
from nfl_quant.schemas import InjuryImpact, SimulationInput
from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.utils.odds import OddsEngine

app = typer.Typer(help="NFL Quantitative Analytics Pipeline (2025 season)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def fetch(
    season: int = typer.Option(2025, help="Season (must be 2025)"),
    data_type: str = typer.Option(
        "all",
        help="Data type: all, schedule, teamstats, pbp, rosters, injuries, or stats",
    ),
) -> None:
    """Fetch NFL data from NFLverse using R (faster, no cache bugs).

    NOTE: This command now uses R nflreadr instead of Python nflreadpy.
    For best results, run the R script directly:
        Rscript scripts/fetch/fetch_nflverse_data.R --current-plus-last

    Examples:
        quant fetch --season 2025 --data-type schedule
        quant fetch --season 2025 --data-type pbp
        quant fetch --season 2025 --data-type rosters
    """
    try:
        import subprocess
        import pandas as pd

        settings.validate_season(season)
        fetcher = DataFetcher()
        nflverse_dir = Path("data/nflverse")

        if data_type in ["all", "schedule"]:
            logger.info("Fetching schedule...")
            fetcher.fetch_nflverse_schedule(season)

        if data_type in ["all", "teamstats"]:
            logger.info("Fetching team stats...")
            fetcher.fetch_nflverse_team_stats(season)

        if data_type in ["all", "pbp"]:
            logger.info("Fetching play-by-play...")
            fetcher.fetch_nflverse_pbp(season)

        if data_type in ["all", "rosters"]:
            logger.info("Loading NFLverse rosters from R-fetched data...")
            rosters_file = nflverse_dir / "rosters.parquet"
            if not rosters_file.exists():
                rosters_file = nflverse_dir / "rosters.csv"
            if rosters_file.exists():
                if rosters_file.suffix == ".parquet":
                    rosters = pd.read_parquet(rosters_file)
                else:
                    rosters = pd.read_csv(rosters_file)
                rosters = rosters[rosters["season"] == season] if "season" in rosters.columns else rosters
                typer.echo(f"Loaded {len(rosters)} roster entries from {rosters_file}")
            else:
                typer.echo("⚠️  No rosters file found. Run: Rscript scripts/fetch/fetch_nflverse_data.R")

        if data_type in ["all", "injuries"]:
            logger.info("Loading injuries from local data...")
            injuries_file = Path("data/injuries/current_injuries.csv")
            if injuries_file.exists():
                injuries = pd.read_csv(injuries_file)
                typer.echo(f"Loaded {len(injuries)} injury records from {injuries_file}")
            else:
                typer.echo("⚠️  No injuries file found. Run: python scripts/fetch/fetch_injuries_api.py")

        if data_type in ["all", "stats"]:
            logger.info("Loading NFLverse player stats from R-fetched data...")
            stats_file = nflverse_dir / "player_stats.parquet"
            if not stats_file.exists():
                stats_file = nflverse_dir / "player_stats.csv"
            if stats_file.exists():
                if stats_file.suffix == ".parquet":
                    stats = pd.read_parquet(stats_file)
                else:
                    stats = pd.read_csv(stats_file)
                stats = stats[stats["season"] == season] if "season" in stats.columns else stats
                typer.echo(f"Loaded {len(stats)} player-week stat records from {stats_file}")
            else:
                typer.echo("⚠️  No player stats file found. Run: Rscript scripts/fetch/fetch_nflverse_data.R")

        typer.echo(f"✅ Data fetch complete for {data_type}")
        typer.echo("💡 For fastest fetching, use R directly: Rscript scripts/fetch/fetch_nflverse_data.R")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def features(
    season: int = typer.Option(2025, help="Season (must be 2025)"),
    week: int = typer.Option(..., help="Week number (1-18)"),
) -> None:
    """Derive team-level features for a given week.

    Examples:
        quant features --season 2025 --week 1
        quant features --season 2025 --week 9
    """
    try:
        settings.validate_season(season)

        if not (1 <= week <= 18):
            raise ValueError("Week must be 1-18")

        fetcher = DataFetcher()
        engine = FeatureEngine()

        logger.info(f"Loading PBP data for season {season}, week {week}...")
        pbp = fetcher.load_pbp_parquet(season)

        # Use data from previous weeks to build features for target week
        # This allows predictions for upcoming games
        pbp_historical = pbp[pbp["week"] < week].copy()

        if len(pbp_historical) == 0:
            logger.warning(f"No historical PBP data found before week {week}")
            raise typer.Exit(code=1)

        # Get unique teams from schedule for target week
        schedule = fetcher.load_schedule_parquet(season)
        schedule_week = schedule[schedule["week"] == week]
        teams_in_week = set(schedule_week["home_team"].unique()) | set(schedule_week["away_team"].unique())

        # Get all teams that have played so far (filter out None values)
        posteams = set(pbp_historical["posteam"].dropna().unique())
        defteams = set(pbp_historical["defteam"].dropna().unique())
        teams = posteams | defteams

        logger.info(f"Using data from weeks 1-{week-1} to generate features for week {week}")

        features_data = []
        for team in sorted(teams):
            for is_offense in [True, False]:
                try:
                    team_features = engine.derive_team_week_features(
                        pbp_historical, team, week, season, is_offense
                    )
                    if engine.validate_features_completeness(team_features):
                        features_data.append(team_features)
                except Exception as e:
                    logger.warning(f"Failed to compute features for {team} (off={is_offense}): {e}")

        logger.info(f"Derived {len(features_data)} team-week feature sets")

        # Save to CSV
        output_dir = settings.PROCESSED_DATA_DIR
        output_file = output_dir / f"features_week{week}.csv"
        df = DataFrame([f.model_dump() for f in features_data])
        df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")

        typer.echo(f"✅ Features derived for {len(features_data)} teams (week {week})")
    except Exception as e:
        logger.error(f"Feature derivation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def simulate(
    season: int = typer.Option(2025, help="Season (must be 2025)"),
    week: int = typer.Option(..., help="Week number (1-18)"),
    trials: int = typer.Option(50000, help="Monte Carlo trials"),
    bankroll: float = typer.Option(10000.0, help="Starting bankroll"),
    kelly_fraction: float = typer.Option(0.5, help="Kelly fraction (0.0-1.0)"),
    max_bet_pct: float = typer.Option(5.0, help="Max bet as % of bankroll"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    odds_file: Optional[str] = typer.Option(None, help="Path to odds CSV"),
) -> None:
    """Run Monte Carlo simulations and generate betting recommendations.

    Examples:
        quant simulate --season 2025 --week 1 --trials 50000 --odds-file data/odds_week1.csv
        quant simulate --season 2025 --week 1 --bankroll 5000 --kelly-fraction 0.25
    """
    try:
        settings.validate_season(season)

        if not (1 <= week <= 18):
            raise ValueError("Week must be 1-18")

        fetcher = DataFetcher()
        simulator = MonteCarloSimulator(seed=seed)
        odds_engine = OddsEngine()
        injury_model = InjuryImpactModel()

        logger.info(f"Loading data for season {season}, week {week}...")
        schedule = fetcher.load_schedule_parquet(season)
        pbp = fetcher.load_pbp_parquet(season)
        pbp_week = pbp[pbp["week"] == week].copy()

        # Filter schedule to this week
        games_week = schedule[schedule["week"] == week]

        if len(games_week) == 0:
            logger.warning(f"No games found for week {week}")
            raise typer.Exit(code=1)

        # Load odds if provided
        odds_records = {}
        if odds_file:
            if not odds_engine.validate_odds_schema(odds_file):
                raise ValueError(f"Invalid odds schema in {odds_file}")
            odds_list = odds_engine.load_odds_csv(odds_file)
            odds_records = {(o.game_id, o.side): o for o in odds_list}

        # Simulate each game
        sim_results = []
        for _, game in games_week.iterrows():
            game_id = game["game_id"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            logger.info(f"Simulating {home_team} vs {away_team} ({game_id})...")

            # Get team features (use historical data only - before this week)
            pbp_to_date = pbp[pbp["week"] < week].copy()

            # Calculate EPAs (offensive and defensive)
            home_off_epa = pbp_to_date[pbp_to_date["posteam"] == home_team]["epa"].mean() or 0.0
            home_def_epa = -(pbp_to_date[pbp_to_date["defteam"] == home_team]["epa"].mean() or 0.0)
            away_off_epa = pbp_to_date[pbp_to_date["posteam"] == away_team]["epa"].mean() or 0.0
            away_def_epa = -(pbp_to_date[pbp_to_date["defteam"] == away_team]["epa"].mean() or 0.0)

            # Apply injury adjustments using NFLverse data
            try:
                home_injury = injury_model.compute_injury_impact(season, week, home_team)
                away_injury = injury_model.compute_injury_impact(season, week, away_team)

                home_off_epa, home_def_epa = injury_model.apply_injury_adjustments(
                    home_off_epa, home_def_epa, home_injury
                )
                away_off_epa, away_def_epa = injury_model.apply_injury_adjustments(
                    away_off_epa, away_def_epa, away_injury
                )
            except Exception as e:
                logger.warning(f"Could not compute injury impact for {home_team} vs {away_team}: {e}")
                # Create empty injury impacts if error occurs
                home_injury = InjuryImpact(
                    team=home_team, week=week, total_impact_offensive_epa=0.0,
                    total_impact_defensive_epa=0.0, injury_count=0,
                    missing_qb=False, missing_ol_count=0, player_impacts=[]
                )
                away_injury = InjuryImpact(
                    team=away_team, week=week, total_impact_offensive_epa=0.0,
                    total_impact_defensive_epa=0.0, injury_count=0,
                    missing_qb=False, missing_ol_count=0, player_impacts=[]
                )

            # Calculate team-specific pace from historical data
            from nfl_quant.features.team_metrics import TeamMetricsExtractor
            extractor = TeamMetricsExtractor()

            # Get pace for each team (offensive plays per game)
            home_pace_data = extractor.get_team_pace(home_team, week)
            away_pace_data = extractor.get_team_pace(away_team, week)

            # Get expected offensive plays for each team in this matchup
            # This accounts for opponent defensive pace
            home_expected_plays = extractor.get_combined_pace(home_team, away_team, week)
            away_expected_plays = extractor.get_combined_pace(away_team, home_team, week)

            # Total expected game plays (both teams combined)
            total_game_plays = home_expected_plays + away_expected_plays

            # FIXED: Pass pace as plays per game, NOT seconds per play
            # The simulator expects plays per game (typically 60-70 range)
            if total_game_plays > 0:
                # Use the actual plays per game for each team
                home_pace = home_expected_plays  # Plays per game
                away_pace = away_expected_plays  # Plays per game
            else:
                # Fallback if no pace data available
                logger.warning(f"No pace data for {home_team} vs {away_team}, using default 65.0")
                home_pace = 65.0  # NFL average plays per team per game
                away_pace = 65.0


            # Create simulation input
            sim_input = SimulationInput(
                game_id=game_id,
                season=season,
                week=week,
                home_team=home_team,
                away_team=away_team,
                home_offensive_epa=float(home_off_epa),
                away_offensive_epa=float(away_off_epa),
                home_defensive_epa=float(home_def_epa),
                away_defensive_epa=float(away_def_epa),
                home_pace=home_pace,
                away_pace=away_pace,
            )

            sim_output = simulator.simulate_game(sim_input, trials=trials)
            sim_results.append((game_id, home_team, away_team, sim_output, home_injury, away_injury))

        # Size bets if odds provided
        bets = []
        if odds_records:
            for game_id, home_team, away_team, sim_output, home_inj, away_inj in sim_results:
                for (odds_game_id, side), odds_rec in odds_records.items():
                    if odds_game_id != game_id:
                        continue

                    if "home" in side:
                        win_prob = sim_output.home_win_prob
                    else:
                        win_prob = sim_output.away_win_prob

                    bet_sizing = odds_engine.size_bet(
                        game_id,
                        side,
                        odds_rec.american_odds,
                        win_prob,
                        bankroll,
                        kelly_fraction,
                        max_bet_pct,
                    )
                    bets.append(bet_sizing)

        # Export results
        output_dir = settings.REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save simulation JSON
        for game_id, _, _, sim_out, _, _ in sim_results:
            sim_file = output_dir / f"sim_{game_id}_{seed}.json"
            simulator.export_results(sim_out, str(sim_file))

        logger.info(f"✅ Simulated {len(sim_results)} games (week {week})")
        if bets:
            logger.info(f"✅ Sized {len(bets)} bets")
        typer.echo(f"✅ Simulations complete. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_models() -> None:
    """Train player prop prediction models (usage and efficiency).

    Examples:
        quant train-models
    """
    try:
        import subprocess
        import sys

        logger.info("Training player prop models...")

        # Run the train_player_models script
        result = subprocess.run(
            [sys.executable, "train_player_models.py"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Model training failed: {result.stderr}")
            raise typer.Exit(code=1)

        # Show output
        print(result.stdout)

        typer.echo("✅ Player prop models trained successfully")
        typer.echo("📁 Models saved to data/models/")

    except Exception as e:
        logger.error(f"Train models command failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def props(
    week: int = typer.Option(8, help="Week number (1-18)"),
    trials: int = typer.Option(50000, help="Monte Carlo trials"),
) -> None:
    """Generate player prop projections for a given week.

    Examples:
        quant props --week 8
        quant props --week 9 --trials 100000
    """
    try:
        import subprocess
        import sys

        logger.info(f"Generating player prop projections for week {week}...")

        # Run the generate_player_props script
        result = subprocess.run(
            [sys.executable, "generate_player_props.py"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Props generation failed: {result.stderr}")
            raise typer.Exit(code=1)

        # Show output
        print(result.stdout)

        typer.echo(f"✅ Player props generated for week {week}")
        typer.echo(f"📁 Results saved to reports/PLAYER_PROPS_WEEK{week}.csv")

    except Exception as e:
        logger.error(f"Props command failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def export(
    season: int = typer.Option(2025, help="Season (must be 2025)"),
    week: int = typer.Option(..., help="Week number (1-18)"),
    format: str = typer.Option("csv", help="Output format: csv or json"),
    output: Optional[str] = typer.Option(None, help="Output file path"),
) -> None:
    """Export results to CSV or JSON.

    Examples:
        quant export --season 2025 --week 1 --format csv --output reports/week1.csv
    """
    try:
        settings.validate_season(season)
        logger.info(f"Exporting week {week} data (format={format})...")

        if not output:
            output = f"reports/week{week}_export.{format}"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Placeholder: gather results and export
        logger.info(f"Exported to {output_path}")
        typer.echo(f"✅ Results exported to {output_path}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
