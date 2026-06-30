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
        return np.eye(4)

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
        M = np.eye(4)
        M[0, 1] = self.length
        M[2, 3] = self.length
        return M


@dataclass
class Quadrupole(Element):
    """Thin-lens quadrupole.

    Parameters
    ----------
    focal_length : float
        Focal length magnitude (meters); sign handled by ``focusing``.
    focusing : bool
        If True, the magnet focuses in the horizontal plane (and defocuses in the vertical plane).
    dx : float
        Horizontal alignment offset (meters).
    dy : float
        Vertical alignment offset (meters).
    tilt : float
        Roll (skew) angle (radians).
    """

    focal_length: float = 1.0
    focusing: bool = True
    kind: str = "quadrupole"
    dx: float = 0.0
    dy: float = 0.0
    tilt: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length == 0:
            raise ValueError("Quadrupole focal length must be non-zero.")

    def matrix(self) -> np.ndarray:
        inv_f_x = -1.0 / self.focal_length if self.focusing else 1.0 / self.focal_length
        inv_f_y = 1.0 / self.focal_length if self.focusing else -1.0 / self.focal_length
        M = np.eye(4)
        M[1, 0] = inv_f_x
        M[3, 2] = inv_f_y

        if self.tilt != 0.0:
            c = np.cos(self.tilt)
            s = np.sin(self.tilt)
            R = np.zeros((4, 4))
            R[0, 0] = c;  R[0, 2] = s
            R[1, 1] = c;  R[1, 3] = s
            R[2, 0] = -s; R[2, 2] = c
            R[3, 1] = -s; R[3, 3] = c
            M = R.T @ M @ R

        return M


@dataclass
class BPM(Element):
    """Beam Position Monitor: records beam x & y-position with gains and offsets."""

    kind: str = "bpm"
    index: int = -1  # Filled by the lattice builder
    dx: float = 0.0
    dy: float = 0.0
    gain_x: float = 1.0
    gain_y: float = 1.0

    def matrix(self) -> np.ndarray:
        return np.eye(4)


@dataclass
class Corrector(Element):
    """Corrector magnet: imparts an angular kick (radians) in the designated plane."""

    kind: str = "corrector"
    index: int = -1  # Filled by the lattice builder
    strength: float = 0.0  # Current kick angle in radians
    plane: str = "x"  # "x" for horizontal, "y" for vertical

    def apply_kick(self, state: np.ndarray, kick_angle: Optional[float] = None) -> np.ndarray:
        theta = self.strength if kick_angle is None else kick_angle
        new_state = state.copy()
        if self.plane == "x":
            new_state[1, 0] += theta
        elif self.plane == "y":
            new_state[3, 0] += theta
        return new_state


@dataclass
class ErrorKick(Element):
    """Alignment / field error modelled as random angular kicks in x and y."""

    kind: str = "error"
    kick_x: float = 0.0
    kick_y: float = 0.0

    def apply_kick(self, state: np.ndarray, kick_angles: Optional[tuple[float, float]] = None) -> np.ndarray:
        kx, ky = (self.kick_x, self.kick_y) if kick_angles is None else kick_angles
        new_state = state.copy()
        new_state[1, 0] += kx
        new_state[3, 0] += ky
        return new_state


__all__ = [
    "Element",
    "Drift",
    "Quadrupole",
    "BPM",
    "Corrector",
    "ErrorKick",
]
