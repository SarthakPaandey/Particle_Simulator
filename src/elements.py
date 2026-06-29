"""Accelerator element definitions.

Each element provides a 2x2 transfer matrix operating on the horizontal
state vector X = [x, x']^T (position in meters, angle in radians).

Corrector and error elements also expose a ``kick`` angle (radians).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Element:
    """Base accelerator element."""

    name: str
    length: float = 0.0
    kind: str = "element"
    # Cumulative longitudinal position (m) assigned by the lattice builder.
    s: float = 0.0

    def matrix(self) -> np.ndarray:
        return np.eye(2)

    def apply_kick(self, state: np.ndarray, kick_angle: float = 0.0) -> np.ndarray:
        """Apply an angular kick to the state (default: no-op).

        Subclasses that model correctors or errors override this method.
        """
        return state


@dataclass
class Drift(Element):
    """Field-free drift space of length L (meters)."""

    kind: str = "drift"

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("Drift length must be non-negative.")

    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.length], [0.0, 1.0]])


@dataclass
class Quadrupole(Element):
    """Thin-lens quadrupole.

    Parameters
    ----------
    focal_length : float
        Focal length magnitude (meters); sign handled by ``focusing``.
    focusing : bool
        If True, the magnet focuses in the horizontal plane.
    """

    focal_length: float = 1.0
    focusing: bool = True
    kind: str = "quadrupole"

    def __post_init__(self) -> None:
        if self.focal_length == 0:
            raise ValueError("Quadrupole focal length must be non-zero.")

    def matrix(self) -> np.ndarray:
        inv_f = -1.0 / self.focal_length if self.focusing else 1.0 / self.focal_length
        return np.array([[1.0, 0.0], [inv_f, 1.0]])


@dataclass
class BPM(Element):
    """Beam Position Monitor: records beam x-position, no optics."""

    kind: str = "bpm"
    index: int = -1  # Filled by the lattice builder

    def matrix(self) -> np.ndarray:
        return np.eye(2)


@dataclass
class Corrector(Element):
    """Corrector magnet: imparts an angular kick theta (radians)."""

    kind: str = "corrector"
    index: int = -1  # Filled by the lattice builder
    strength: float = 0.0  # Current kick angle in radians

    def apply_kick(self, state: np.ndarray, kick_angle: Optional[float] = None) -> np.ndarray:
        theta = self.strength if kick_angle is None else kick_angle
        new_state = state.copy()
        new_state[1, 0] += theta
        return new_state


@dataclass
class ErrorKick(Element):
    """Alignment / field error modelled as a random angular kick."""

    kind: str = "error"
    kick: float = 0.0

    def apply_kick(self, state: np.ndarray, kick_angle: Optional[float] = None) -> np.ndarray:
        k = self.kick if kick_angle is None else kick_angle
        new_state = state.copy()
        new_state[1, 0] += k
        return new_state


__all__ = [
    "Element",
    "Drift",
    "Quadrupole",
    "BPM",
    "Corrector",
    "ErrorKick",
]
