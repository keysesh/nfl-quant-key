"""
NFL QUANT Betting Strategies

Contains strategy modules for generating betting picks.
"""

from .contrarian import (
    generate_contrarian_picks,
    get_contrarian_signal,
    BetDirection,
    ContrarianPick
)

__all__ = [
    'generate_contrarian_picks',
    'get_contrarian_signal',
    'BetDirection',
    'ContrarianPick'
]
