"""First-level empirical power-calibration API."""

from .calibration import (
    DEFAULT_EFFECT_SIZES,
    PowerCalibration,
    inject_activation,
    power_analysis,
)

__all__ = [
    'DEFAULT_EFFECT_SIZES',
    'PowerCalibration',
    'inject_activation',
    'power_analysis',
]
