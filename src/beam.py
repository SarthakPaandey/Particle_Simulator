"""Beam state container."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BeamState:
    """Represents a single-particle horizontal state X = [x, x']^T.

    Attributes
    ----------
    x : float
        Transverse position in meters.
    xp : float
        Divergence / angle in radians (x').
    """

    x: float = 0.0
    xp: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array([[self.x], [self.xp]])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "BeamState":
        return cls(x=float(vec[0, 0]), xp=float(vec[1, 0]))


__all__ = ["BeamState"]
