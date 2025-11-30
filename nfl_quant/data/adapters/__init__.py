"""Stats data adapters for transforming source-specific formats to canonical format."""

from .base_adapter import StatsAdapter
from .nflverse_adapter import NFLVerseAdapter

__all__ = ["StatsAdapter", "NFLVerseAdapter"]
