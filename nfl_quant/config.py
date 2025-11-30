"""Configuration and constants for NFL Quant pipeline."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings for the NFL Quant pipeline."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

    # Season (hard lock to 2025)
    SEASON: int = 2025
    ALLOWED_SEASONS: list[int] = [2025]

    # API Configuration
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    REQUEST_TIMEOUT: int = 30
    REQUEST_RETRIES: int = 3
    REQUEST_BACKOFF: float = 1.0

    # Odds API Configuration
    ODDS_API_KEY: Optional[str] = None
    ODDS_API_REGION: str = "us"
    ODDS_API_MARKETS: str = "spreads,totals,h2h"
    ODDS_API_ODDS_FORMAT: str = "american"

    # External API Keys
    METEOSTAT_API_KEY: Optional[str] = None

    # Dynamic Season/Week (overridable via environment)
    CURRENT_SEASON: Optional[int] = None
    CURRENT_WEEK: Optional[int] = None

    # Feature Derivation
    EPA_THRESHOLD: float = 0.0  # Success = EPA > 0
    EXPLOSIVE_PLAY_AIR_YARDS_THRESHOLD: int = 15
    EXPLOSIVE_PLAY_TOTAL_YARDS_THRESHOLD: int = 20
    REDZONE_YARD_LINE: int = 20

    # Simulation
    DEFAULT_TRIALS: int = 50000
    DEFAULT_SEED: int = 42
    # NOTE: Bankroll configuration must be provided in configs/bankroll_config.json
    # No default bankroll is provided - users must set their actual wallet amount

    # Injury Impact (multipliers)
    INJURY_CONFIG_PATH: Path = CONFIGS_DIR / "injury_multipliers.yaml"

    # Validation
    MIN_GAMES_FOR_FEATURE: int = 3

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False

    def validate_season(self, season: Optional[int] = None) -> int:
        """Validate season is 2025.

        Args:
            season: Season to validate, or None to use default

        Returns:
            Validated season

        Raises:
            ValueError: If season is not in allowed seasons
        """
        check_season = season if season is not None else self.SEASON
        if check_season not in self.ALLOWED_SEASONS:
            raise ValueError(
                f"Season {check_season} not allowed. Only {self.ALLOWED_SEASONS} supported."
            )
        return check_season

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.REPORTS_DIR,
            self.CONFIGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_dirs()

# NOTE: nflreadpy is NOT used - all NFLverse data comes from R/nflreadr
# Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch/update data
# Data is stored in data/nflverse/ directory as parquet files

# Constants
TEAM_ABBREVIATIONS = {
    "ARI": "ARI",  # Arizona Cardinals
    "BAL": "BAL",  # Baltimore Ravens
    "HOU": "HOU",  # Houston Texans
}

# Reverse mapping for compatibility
TEAM_ABBREV_REVERSE = {v: k for k, v in TEAM_ABBREVIATIONS.items()}
TEAM_ABBREV_REVERSE.update({k: k for k in TEAM_ABBREVIATIONS.keys()})



