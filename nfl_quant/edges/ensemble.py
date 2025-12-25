"""
Edge Ensemble Orchestrator

Combines LVT and Player Bias edges into a unified betting system.

Betting Rules:
- BOTH AGREE: Highest conviction (2 units)
- LVT ONLY: High conviction (1.5 units)
- PLAYER BIAS ONLY: Moderate conviction (1 unit)
- CONFLICT: No bet (edges disagree on direction)
- NEITHER: No bet (no edge detected)
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .lvt_edge import LVTEdge
from .player_bias_edge import PlayerBiasEdge
from .direction_edge import get_direction_edge
from configs.ensemble_config import (
    EdgeSource,
    EnsembleDecision,
    MARKET_ENSEMBLE_CONFIG,
    GLOBAL_SETTINGS,
    get_units_for_source,
    get_combined_confidence,
    get_market_config,
)
from configs.edge_config import (
    get_lvt_threshold,
    get_player_bias_threshold,
    EDGE_MARKETS,
    check_data_quality,
    MARKET_TRAILING_REQUIREMENTS,
    MIN_PLAYER_GAMES,
)
from configs.model_config import MARKET_DIRECTION_CONSTRAINTS


# Use direction-specific models for validation
USE_DIRECTION_MODELS = True


def is_direction_allowed(
    market: str,
    direction: str,
    row: pd.Series = None,
) -> tuple[bool, str]:
    """
    Check if a direction is allowed for a market using direction-specific models.

    Args:
        market: Market name (e.g., 'player_rush_yds')
        direction: Direction ('OVER' or 'UNDER')
        row: Optional row data for direction model validation

    Returns:
        Tuple of (allowed, reason)
    """
    # First check static constraints
    constraint = MARKET_DIRECTION_CONSTRAINTS.get(market)
    if constraint == 'UNDER_ONLY' and direction == 'OVER':
        return (False, f"OVER disabled for {market} (static constraint)")
    if constraint == 'OVER_ONLY' and direction == 'UNDER':
        return (False, f"UNDER disabled for {market} (static constraint)")

    # If direction models enabled and we have row data, validate with model
    if USE_DIRECTION_MODELS and row is not None:
        direction_edge = get_direction_edge()
        if direction_edge.loaded:
            allowed, conf, reason = direction_edge.should_allow_direction(row, market, direction)
            return (allowed, reason)

    # Default: allow
    return (True, "Direction allowed")


class EdgeEnsemble:
    """
    Orchestrates LVT and Player Bias edges for betting decisions.

    This class:
    1. Evaluates bets against both edges
    2. Determines edge source (BOTH, LVT_ONLY, PLAYER_BIAS_ONLY, CONFLICT, NEITHER)
    3. Applies betting rules and unit sizing
    4. Enforces direction constraints and risk limits
    """

    def __init__(
        self,
        lvt_edge: LVTEdge,
        player_bias_edge: PlayerBiasEdge,
    ):
        """
        Initialize ensemble with trained edges.

        Args:
            lvt_edge: Trained LVT edge
            player_bias_edge: Trained Player Bias edge
        """
        self.lvt_edge = lvt_edge
        self.player_bias_edge = player_bias_edge
        self.settings = GLOBAL_SETTINGS

    def evaluate_bet(
        self,
        row: pd.Series,
        market: str,
    ) -> EnsembleDecision:
        """
        Evaluate a potential bet using both edges.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            EnsembleDecision with betting recommendation
        """
        if market not in EDGE_MARKETS:
            return EnsembleDecision(
                should_bet=False,
                direction=None,
                units=0.0,
                source=EdgeSource.NEITHER,
                lvt_confidence=0.0,
                player_bias_confidence=0.0,
                combined_confidence=0.0,
                reasoning=f"Market {market} not supported",
            )

        # DATA QUALITY CHECK: No data = No bet
        # Check if we have the required data for at least one edge
        lvt_quality = check_data_quality(row, market, edge_type='lvt')
        pb_quality = check_data_quality(row, market, edge_type='player_bias')

        # If BOTH edges lack required data, skip the bet entirely
        if not lvt_quality.has_required_data and not pb_quality.has_required_data:
            return EnsembleDecision(
                should_bet=False,
                direction=None,
                units=0.0,
                source=EdgeSource.NO_DATA,
                lvt_confidence=0.0,
                player_bias_confidence=0.0,
                combined_confidence=0.0,
                reasoning=f"Insufficient data: LVT missing {lvt_quality.missing_fields}, PB missing {pb_quality.missing_fields}",
            )

        # Evaluate LVT edge (only if has required data)
        lvt_triggers = False
        lvt_conf = 0.0
        lvt_direction = None

        if lvt_quality.has_required_data:
            lvt_triggers = self.lvt_edge.should_trigger(row, market)
            lvt_conf = self.lvt_edge.get_confidence(row, market) if lvt_triggers else 0.0
            lvt_direction = self.lvt_edge.get_direction(row, market) if lvt_triggers else None

            # Check LVT confidence threshold
            lvt_threshold = get_lvt_threshold(market)
            if lvt_triggers and lvt_conf < lvt_threshold.confidence:
                lvt_triggers = False  # Below confidence threshold

        # Evaluate Player Bias edge (only if has required data)
        pb_triggers = False
        pb_conf = 0.0
        pb_direction = None

        if pb_quality.has_required_data:
            pb_triggers = self.player_bias_edge.should_trigger(row, market)
            pb_conf = self.player_bias_edge.get_confidence(row, market) if pb_triggers else 0.0
            pb_direction = self.player_bias_edge.get_direction(row, market) if pb_triggers else None

            # Check Player Bias confidence threshold
            pb_threshold = get_player_bias_threshold(market)
            if pb_triggers and pb_conf < pb_threshold.confidence:
                pb_triggers = False  # Below confidence threshold

        # Determine edge source and decision
        return self._make_decision(
            row=row,
            market=market,
            lvt_triggers=lvt_triggers,
            lvt_conf=lvt_conf,
            lvt_direction=lvt_direction,
            pb_triggers=pb_triggers,
            pb_conf=pb_conf,
            pb_direction=pb_direction,
        )

    def _make_decision(
        self,
        row: pd.Series,
        market: str,
        lvt_triggers: bool,
        lvt_conf: float,
        lvt_direction: Optional[str],
        pb_triggers: bool,
        pb_conf: float,
        pb_direction: Optional[str],
    ) -> EnsembleDecision:
        """
        Make betting decision based on edge evaluations.

        Args:
            market: Market being evaluated
            lvt_triggers: Whether LVT edge triggers
            lvt_conf: LVT confidence (P(UNDER))
            lvt_direction: LVT recommended direction
            pb_triggers: Whether Player Bias edge triggers
            pb_conf: Player Bias confidence (P(UNDER))
            pb_direction: Player Bias recommended direction

        Returns:
            EnsembleDecision with final recommendation
        """
        # Both edges trigger
        if lvt_triggers and pb_triggers:
            if lvt_direction == pb_direction:
                # BOTH AGREE
                source = EdgeSource.BOTH
                direction = lvt_direction
                units = get_units_for_source(market, source)
                combined_conf = get_combined_confidence(market, lvt_conf, pb_conf, source)

                # Enforce market-specific direction constraint using direction models
                allowed, reason = is_direction_allowed(market, direction, row)
                if not allowed:
                    return EnsembleDecision(
                        should_bet=False,
                        direction=None,
                        units=0.0,
                        source=EdgeSource.NEITHER,
                        lvt_confidence=lvt_conf,
                        player_bias_confidence=pb_conf,
                        combined_confidence=0.0,
                        reasoning=reason,
                    )

                return EnsembleDecision(
                    should_bet=True,
                    direction=direction,
                    units=units,
                    source=source,
                    lvt_confidence=lvt_conf,
                    player_bias_confidence=pb_conf,
                    combined_confidence=combined_conf,
                    reasoning=f"BOTH edges agree on {direction}: LVT={lvt_conf:.1%}, PB={pb_conf:.1%}",
                )
            else:
                # CONFLICT
                return EnsembleDecision(
                    should_bet=False,
                    direction=None,
                    units=0.0,
                    source=EdgeSource.CONFLICT,
                    lvt_confidence=lvt_conf,
                    player_bias_confidence=pb_conf,
                    combined_confidence=0.0,
                    reasoning=f"CONFLICT: LVT says {lvt_direction}, PB says {pb_direction}",
                )

        # Only LVT triggers
        elif lvt_triggers:
            source = EdgeSource.LVT_ONLY
            direction = lvt_direction
            units = get_units_for_source(market, source)
            combined_conf = get_combined_confidence(market, lvt_conf, pb_conf, source)

            # Enforce market-specific direction constraint using direction models
            allowed, reason = is_direction_allowed(market, direction, row)
            if not allowed:
                return EnsembleDecision(
                    should_bet=False,
                    direction=None,
                    units=0.0,
                    source=EdgeSource.NEITHER,
                    lvt_confidence=lvt_conf,
                    player_bias_confidence=pb_conf,
                    combined_confidence=0.0,
                    reasoning=reason,
                )

            return EnsembleDecision(
                should_bet=True,
                direction=direction,
                units=units,
                source=source,
                lvt_confidence=lvt_conf,
                player_bias_confidence=pb_conf,
                combined_confidence=combined_conf,
                reasoning=f"LVT edge triggers {direction}: {lvt_conf:.1%}",
            )

        # Only Player Bias triggers
        elif pb_triggers:
            source = EdgeSource.PLAYER_BIAS_ONLY
            direction = pb_direction
            units = get_units_for_source(market, source)
            combined_conf = get_combined_confidence(market, lvt_conf, pb_conf, source)

            # Enforce market-specific direction constraint using direction models
            allowed, reason = is_direction_allowed(market, direction, row)
            if not allowed:
                return EnsembleDecision(
                    should_bet=False,
                    direction=None,
                    units=0.0,
                    source=EdgeSource.NEITHER,
                    lvt_confidence=lvt_conf,
                    player_bias_confidence=pb_conf,
                    combined_confidence=0.0,
                    reasoning=reason,
                )

            return EnsembleDecision(
                should_bet=True,
                direction=direction,
                units=units,
                source=source,
                lvt_confidence=lvt_conf,
                player_bias_confidence=pb_conf,
                combined_confidence=combined_conf,
                reasoning=f"Player Bias edge triggers {direction}: {pb_conf:.1%}",
            )

        # Neither triggers
        else:
            return EnsembleDecision(
                should_bet=False,
                direction=None,
                units=0.0,
                source=EdgeSource.NEITHER,
                lvt_confidence=lvt_conf,
                player_bias_confidence=pb_conf,
                combined_confidence=0.0,
                reasoning="No edge detected",
            )

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        market: str,
    ) -> pd.DataFrame:
        """
        Evaluate multiple bets at once.

        Args:
            df: DataFrame with features for multiple bets
            market: Market being evaluated

        Returns:
            DataFrame with betting decisions for each row
        """
        results = []
        for idx, row in df.iterrows():
            decision = self.evaluate_bet(row, market)
            results.append({
                'index': idx,
                'should_bet': decision.should_bet,
                'direction': decision.direction,
                'units': decision.units,
                'source': decision.source.value,
                'lvt_confidence': decision.lvt_confidence,
                'player_bias_confidence': decision.player_bias_confidence,
                'combined_confidence': decision.combined_confidence,
                'reasoning': decision.reasoning,
            })

        return pd.DataFrame(results).set_index('index')

    def get_recommendations(
        self,
        df: pd.DataFrame,
        market: str,
        max_bets: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get filtered betting recommendations.

        Args:
            df: DataFrame with features for multiple bets
            market: Market being evaluated
            max_bets: Maximum bets to return (sorted by confidence)

        Returns:
            DataFrame with recommended bets only
        """
        decisions = self.evaluate_batch(df, market)

        # Filter to bets only
        bets = decisions[decisions['should_bet']].copy()

        if len(bets) == 0:
            return bets

        # Sort by combined confidence
        bets = bets.sort_values('combined_confidence', ascending=False)

        # Apply max_bets limit
        if max_bets is not None:
            config = get_market_config(market)
            max_bets = min(max_bets, config.max_daily_bets)
            bets = bets.head(max_bets)

        return bets

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of ensemble state.

        Returns:
            Dict with ensemble info and metrics
        """
        return {
            'lvt_edge': {
                'version': self.lvt_edge.version,
                'trained_date': str(self.lvt_edge.trained_date),
                'markets': list(self.lvt_edge.models.keys()),
                'metrics': self.lvt_edge.metrics,
            },
            'player_bias_edge': {
                'version': self.player_bias_edge.version,
                'trained_date': str(self.player_bias_edge.trained_date),
                'markets': list(self.player_bias_edge.models.keys()),
                'metrics': self.player_bias_edge.metrics,
            },
            'settings': {
                'enforce_under_only': self.settings.enforce_under_only,
                'max_total_daily_bets': self.settings.max_total_daily_bets,
            },
        }

    @classmethod
    def load(cls, lvt_path: Path = None, pb_path: Path = None) -> 'EdgeEnsemble':
        """
        Load ensemble from disk.

        Args:
            lvt_path: Path to LVT edge model
            pb_path: Path to Player Bias edge model

        Returns:
            Loaded EdgeEnsemble instance
        """
        lvt_edge = LVTEdge.load(lvt_path)
        player_bias_edge = PlayerBiasEdge.load(pb_path)

        return cls(lvt_edge, player_bias_edge)

    def save(self, lvt_path: Path = None, pb_path: Path = None) -> None:
        """
        Save ensemble to disk.

        Args:
            lvt_path: Path to save LVT edge model
            pb_path: Path to save Player Bias edge model
        """
        self.lvt_edge.save(lvt_path)
        self.player_bias_edge.save(pb_path)

    def __repr__(self) -> str:
        return (
            f"EdgeEnsemble(\n"
            f"  lvt_edge={self.lvt_edge},\n"
            f"  player_bias_edge={self.player_bias_edge}\n"
            f")"
        )
