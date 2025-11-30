"""
Alt-line generation for player props and game totals.

Generates alternative lines (±2.5, ±5.0, ±7.5, etc.) and calculates
probabilities and expected value for each alt line.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AltLine:
    """Represents an alternative line."""
    line: float
    prob_over: float
    prob_under: float
    odds_over: Optional[int] = None
    odds_under: Optional[int] = None
    ev_over: Optional[float] = None
    ev_under: Optional[float] = None
    delta_from_main: float = 0.0


class AltLineGenerator:
    """Generate alternative lines for props."""

    def __init__(self):
        """Initialize alt-line generator."""
        self.default_offsets = [-7.5, -5.0, -2.5, 2.5, 5.0, 7.5]

    def generate_alt_lines(
        self,
        main_line: float,
        mean_projection: float,
        std_projection: float,
        main_odds_over: Optional[int] = None,
        main_odds_under: Optional[int] = None,
        custom_offsets: Optional[List[float]] = None
    ) -> List[AltLine]:
        """
        Generate alternative lines based on projection distribution.

        Args:
            main_line: Main market line
            mean_projection: Mean of model projection
            std_projection: Standard deviation of model projection
            main_odds_over: Odds for Over on main line (if available)
            main_odds_under: Odds for Under on main line (if available)
            custom_offsets: Custom line offsets (uses default if None)

        Returns:
            List of AltLine objects
        """
        offsets = custom_offsets or self.default_offsets
        alt_lines = []

        for offset in offsets:
            alt_line_value = main_line + offset

            # Skip if line is negative (invalid for most props)
            if alt_line_value < 0:
                continue

            # Calculate probability using normal CDF
            # P(X > line) = 1 - CDF(line)
            z_score = (alt_line_value - mean_projection) / std_projection
            prob_under = self._normal_cdf(z_score)
            prob_over = 1.0 - prob_under

            # Estimate alt-line odds if main odds available
            odds_over, odds_under = self._estimate_alt_odds(
                main_line=main_line,
                alt_line=alt_line_value,
                prob_over=prob_over,
                main_odds_over=main_odds_over,
                main_odds_under=main_odds_under
            )

            # Calculate EV if odds available
            ev_over = self._calculate_ev(
                prob=prob_over,
                odds=odds_over
            ) if odds_over else None

            ev_under = self._calculate_ev(
                prob=prob_under,
                odds=odds_under
            ) if odds_under else None

            alt_lines.append(AltLine(
                line=alt_line_value,
                prob_over=prob_over,
                prob_under=prob_under,
                odds_over=odds_over,
                odds_under=odds_under,
                ev_over=ev_over,
                ev_under=ev_under,
                delta_from_main=offset
            ))

        # Sort by best EV (over or under)
        alt_lines.sort(
            key=lambda x: max(
                x.ev_over or -999,
                x.ev_under or -999
            ),
            reverse=True
        )

        return alt_lines

    def get_best_alt_lines(
        self,
        alt_lines: List[AltLine],
        top_n: int = 3,
        min_ev: float = 0.0
    ) -> List[AltLine]:
        """
        Get the best N alternative lines by EV.

        Args:
            alt_lines: List of alternative lines
            top_n: Number of top lines to return
            min_ev: Minimum EV threshold

        Returns:
            Top N alt lines by EV
        """
        # Filter by minimum EV
        valid_lines = [
            line for line in alt_lines
            if (line.ev_over and line.ev_over >= min_ev) or
               (line.ev_under and line.ev_under >= min_ev)
        ]

        return valid_lines[:top_n]

    def format_alt_line_summary(
        self,
        alt_lines: List[AltLine],
        max_display: int = 4
    ) -> str:
        """
        Format alt-lines for dashboard display.

        Args:
            alt_lines: List of alt lines
            max_display: Maximum lines to display

        Returns:
            Formatted string for dashboard cell
        """
        if not alt_lines:
            return "N/A"

        summaries = []
        for line in alt_lines[:max_display]:
            # Determine best direction
            if line.ev_over and line.ev_under:
                if line.ev_over > line.ev_under:
                    direction = "O"
                    prob = line.prob_over
                    ev = line.ev_over
                else:
                    direction = "U"
                    prob = line.prob_under
                    ev = line.ev_under
            elif line.ev_over:
                direction = "O"
                prob = line.prob_over
                ev = line.ev_over
            elif line.ev_under:
                direction = "U"
                prob = line.prob_under
                ev = line.ev_under
            else:
                # No odds, show prob only
                direction = "O" if line.prob_over > 0.5 else "U"
                prob = max(line.prob_over, line.prob_under)
                ev = None

            if ev is not None:
                summaries.append(
                    f"{direction}{line.line:.1f} "
                    f"({prob:.1%}, EV:{ev:+.2f})"
                )
            else:
                summaries.append(
                    f"{direction}{line.line:.1f} ({prob:.1%})"
                )

        return " | ".join(summaries)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """
        Calculate cumulative distribution function for standard normal.

        Args:
            z: Z-score

        Returns:
            Probability P(X <= z)
        """
        # Using scipy for erf function (more reliable than numpy)
        try:
            from scipy.special import erf
            return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        except ImportError:
            # Fallback to approximation if scipy not available
            # Abramowitz and Stegun approximation
            a1 =  0.254829592
            a2 = -0.284496736
            a3 =  1.421413741
            a4 = -1.453152027
            a5 =  1.061405429
            p  =  0.3275911

            sign = 1 if z >= 0 else -1
            z = abs(z)

            t = 1.0 / (1.0 + p * z)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)

            return 0.5 * (1.0 + sign * y)

    @staticmethod
    def _estimate_alt_odds(
        main_line: float,
        alt_line: float,
        prob_over: float,
        main_odds_over: Optional[int],
        main_odds_under: Optional[int]
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Estimate alt-line odds based on main line odds and probability shift.

        Args:
            main_line: Main market line
            alt_line: Alternative line
            prob_over: Probability of over on alt line
            main_odds_over: Main line over odds
            main_odds_under: Main line under odds

        Returns:
            Tuple of (odds_over, odds_under) or (None, None)
        """
        # If no main odds, cannot estimate
        if main_odds_over is None and main_odds_under is None:
            return None, None

        # Convert probability to fair odds
        def prob_to_american(p: float) -> int:
            if p >= 0.99:
                return -10000
            if p <= 0.01:
                return 10000

            decimal = 1.0 / p
            if decimal >= 2.0:
                return int((decimal - 1) * 100)
            else:
                return int(-100 / (decimal - 1))

        odds_over = prob_to_american(prob_over)
        odds_under = prob_to_american(1.0 - prob_over)

        return odds_over, odds_under

    @staticmethod
    def _calculate_ev(prob: float, odds: Optional[int]) -> Optional[float]:
        """
        Calculate expected value.

        Args:
            prob: Win probability
            odds: American odds

        Returns:
            Expected value per $1 wagered
        """
        if odds is None:
            return None

        # Convert American to decimal
        if odds > 0:
            decimal = (odds / 100) + 1
        else:
            decimal = (100 / abs(odds)) + 1

        # EV = (prob * profit) - (1 - prob) * stake
        # For $1 bet: EV = prob * (decimal - 1) - (1 - prob) * 1
        ev = prob * (decimal - 1) - (1 - prob)

        return ev


def generate_alt_lines_for_prop(
    player: str,
    prop_type: str,
    main_line: float,
    mean_projection: float,
    std_projection: float,
    main_odds_over: Optional[int] = None,
    main_odds_under: Optional[int] = None
) -> Dict:
    """
    Generate alt-lines for a specific prop.

    Args:
        player: Player name
        prop_type: Type of prop (e.g., 'receiving_yards')
        main_line: Main market line
        mean_projection: Model mean projection
        std_projection: Model standard deviation
        main_odds_over: Main odds for Over
        main_odds_under: Main odds for Under

    Returns:
        Dictionary with alt-line data
    """
    generator = AltLineGenerator()

    alt_lines = generator.generate_alt_lines(
        main_line=main_line,
        mean_projection=mean_projection,
        std_projection=std_projection,
        main_odds_over=main_odds_over,
        main_odds_under=main_odds_under
    )

    best_lines = generator.get_best_alt_lines(alt_lines, top_n=4)

    return {
        'player': player,
        'prop_type': prop_type,
        'main_line': main_line,
        'all_alt_lines': alt_lines,
        'best_alt_lines': best_lines,
        'summary': generator.format_alt_line_summary(best_lines)
    }
