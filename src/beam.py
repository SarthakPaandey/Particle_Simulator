"""Beam state container."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BeamState:
    """Represents a single-particle 4D transverse state X = [x, x', y, y']^T.

    Attributes
    ----------
    x : float
        Horizontal transverse position in meters.
    xp : float
        Horizontal divergence / angle in radians (x').
    y : float
        Vertical transverse position in meters.
    yp : float
        Vertical divergence / angle in radians (y').
    """

    x: float = 0.0
    xp: float = 0.0
    y: float = 0.0
    yp: float = 0.0

    def as_vector(self) -> np.ndarray:
        return np.array([[self.x], [self.xp], [self.y], [self.yp]])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "BeamState":
        return cls(
            x=float(vec[0, 0]),
            xp=float(vec[1, 0]),
            y=float(vec[2, 0]),
            yp=float(vec[3, 0]),
        )


__all__ = ["BeamState"]
