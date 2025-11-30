#!/usr/bin/env python3
"""
Test if efficiency model was trained with different trailing_yards_per_target values.

Hypothesis: Model was trained when Bug #3 existed, so it saw generic
yards_per_opportunity (~7.0) instead of position-specific yards_per_target (~9.4).

Now that Bug #3 is fixed, model sees 9.4 and thinks it's "too high" so reduces it.
