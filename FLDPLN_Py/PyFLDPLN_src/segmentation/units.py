"""Unit conversion helpers for translating user-facing config values (square
miles, miles) into pixel-count thresholds, matching the original MATLAB
wrapper's mi2px/sqmi2px conversions (which assume horizontal raster units are
meters).
"""

from __future__ import annotations

import math

FEET_PER_MILE = 5280
METERS_PER_FOOT = 0.3048


def _round_half_away_from_zero(x: float) -> int:
    """Matches MATLAB's round() (half away from zero), unlike Python's
    built-in round() (half to even / banker's rounding)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def miles_to_pixels(miles: float, pixel_size_m: float) -> int:
    mi_to_px = FEET_PER_MILE * METERS_PER_FOOT / pixel_size_m
    return _round_half_away_from_zero(miles * mi_to_px)


def sqmiles_to_pixels(sqmiles: float, pixel_size_m: float) -> int:
    sqmi_to_px = (FEET_PER_MILE**2) * (METERS_PER_FOOT**2) / (pixel_size_m**2)
    return _round_half_away_from_zero(sqmiles * sqmi_to_px)
