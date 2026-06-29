"""Performance metrics for orbit correction."""
from __future__ import annotations

import numpy as np


def rms_error(values: np.ndarray) -> float:
    """Root-mean-square of an array of values."""
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def max_abs_error(values: np.ndarray) -> float:
    """Maximum absolute deviation."""
    if len(values) == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def improvement(before: float, after: float) -> float:
    """Percentage improvement: 100 * (before - after) / before.

    Returns 0 if before is zero or negative.
    """
    if before <= 0:
        return 0.0
    return 100.0 * (before - after) / before


__all__ = ["rms_error", "max_abs_error", "improvement"]
