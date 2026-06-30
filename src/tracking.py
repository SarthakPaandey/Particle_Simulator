"""Beam orbit tracking through a lattice."""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .beam import BeamState
from .elements import BPM, Corrector, ErrorKick, Element, Quadrupole
from .lattice import Lattice


class Trajectory:
    """Stores the beam state at every lattice position in 4D transverse phase space."""

    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        self.s: np.ndarray = lattice.s_positions.copy()
        n = len(lattice)
        self.x: np.ndarray = np.zeros(n)
        self.xp: np.ndarray = np.zeros(n)
        self.y: np.ndarray = np.zeros(n)
        self.yp: np.ndarray = np.zeros(n)
        self.bpm_readings_x: List[float] = []
        self.bpm_readings_y: List[float] = []
        self._bpm_elements: List[BPM] = []

    @property
    def n_bpms(self) -> int:
        return len(self._bpm_elements) if self._bpm_elements else len(self.lattice.bpms)

    @property
    def bpm_s_positions(self) -> np.ndarray:
        return np.array([bpm.s for bpm in self.lattice.bpms])

    @property
    def bpm_x_positions(self) -> np.ndarray:
        if self._bpm_elements:
            return np.array(self.bpm_readings_x)
        return np.array([0.0] * len(self.lattice.bpms))

    @property
    def bpm_y_positions(self) -> np.ndarray:
        if self._bpm_elements:
            return np.array(self.bpm_readings_y)
        return np.array([0.0] * len(self.lattice.bpms))


def track_beam(
    lattice: Lattice,
    initial_state: BeamState,
    corrector_strengths: Optional[np.ndarray] = None,
    error_kicks: Optional[np.ndarray] = None,  # Shape (2, len(lattice)) for x and y
    add_noise: bool = False,
    bpm_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Trajectory:
    """Propagate a single particle through the lattice in 4D.

    The beam state X = [x, x', y, y']^T is advanced element-by-element:
      X_{k+1} = M_k @ X_k + [0, C_x + E_x, 0, C_y + E_y]^T

    Parameters
    ----------
    lattice : Lattice
    initial_state : BeamState
    corrector_strengths : np.ndarray, optional
        Array of kick angles (radians) for each corrector.
    error_kicks : np.ndarray, optional
        Array of error kicks shape (2, len(lattice)), row 0: x, row 1: y.
    add_noise : bool
    bpm_noise_sigma : float
    rng : Generator, optional

    Returns
    -------
    traj : Trajectory
    """
    if corrector_strengths is not None:
        for corr, strength in zip(lattice.correctors, corrector_strengths):
            corr.strength = float(strength)

    if error_kicks is not None and error_kicks.shape != (2, len(lattice)):
        raise ValueError(
            f"error_kicks shape {error_kicks.shape} != (2, {len(lattice)})"
        )

    if rng is None:
        rng = np.random.default_rng()

    traj = Trajectory(lattice)
    state = initial_state.as_vector().copy()

    for idx, element in enumerate(lattice.elements):
        traj.x[idx] = float(state[0, 0])
        traj.xp[idx] = float(state[1, 0])
        traj.y[idx] = float(state[2, 0])
        traj.yp[idx] = float(state[3, 0])

        M = element.matrix()
        if isinstance(element, Quadrupole) and (element.dx != 0.0 or element.dy != 0.0):
            d = np.array([[element.dx], [0.0], [element.dy], [0.0]])
            state = M @ state + (np.eye(4) - M) @ d
        else:
            state = M @ state

        kx = 0.0
        ky = 0.0

        if isinstance(element, Corrector):
            if element.plane == "x":
                kx += element.strength
            elif element.plane == "y":
                ky += element.strength

        if isinstance(element, ErrorKick):
            kx += element.kick_x
            ky += element.kick_y

        # Generic error kicks applied element-by-element
        if error_kicks is not None:
            kx += error_kicks[0, idx]
            ky += error_kicks[1, idx]

        if kx != 0.0:
            state[1, 0] += kx
        if ky != 0.0:
            state[3, 0] += ky

        if isinstance(element, BPM):
            x_true = float(state[0, 0])
            y_true = float(state[2, 0])
            noise_x = 0.0
            noise_y = 0.0
            if add_noise and bpm_noise_sigma > 0:
                noise_x = float(rng.normal(0.0, bpm_noise_sigma))
                noise_y = float(rng.normal(0.0, bpm_noise_sigma))
            
            # Apply BPM gain and offset calibration
            x_meas = element.gain_x * (x_true - element.dx) + noise_x
            y_meas = element.gain_y * (y_true - element.dy) + noise_y

            traj._bpm_elements.append(element)
            traj.bpm_readings_x.append(x_meas)
            traj.bpm_readings_y.append(y_meas)

    return traj


def compute_response_matrix(
    lattice: Lattice,
    delta_theta: float = 1e-4,
    additive_noise: bool = False,
    bpm_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute the (2*N_BPM) x N_CORR response matrix via finite differences.

    R[i, j] = ΔBPM_i / Δθ_j   approximately  ∂BPM_i / ∂θ_j

    The BPM error vector is stacked: [x_1, ..., x_N, y_1, ..., y_N]^T.

    Parameters
    ----------
    lattice : Lattice
    delta_theta : float
    additive_noise, bpm_noise_sigma : passed to track_beam.
    rng : Generator, optional

    Returns
    -------
    R : np.ndarray, shape (2 * n_bpm, n_corr)
    """
    bpms = lattice.bpms
    correctors = lattice.correctors
    n_bpm = len(bpms)
    n_corr = len(correctors)

    if n_bpm == 0 or n_corr == 0:
        return np.zeros((2 * n_bpm, n_corr))

    if rng is None:
        rng = np.random.default_rng()

    # Reference: all correctors zero
    ref_traj = track_beam(lattice, BeamState(), None, None, additive_noise, bpm_noise_sigma, rng)
    ref_bpm = np.concatenate([ref_traj.bpm_x_positions, ref_traj.bpm_y_positions])

    R = np.zeros((2 * n_bpm, n_corr))
    for j in range(n_corr):
        strengths = np.zeros(n_corr)
        strengths[j] = delta_theta
        traj = track_beam(lattice, BeamState(), strengths, None, additive_noise, bpm_noise_sigma, rng)
        bpm_readings = np.concatenate([traj.bpm_x_positions, traj.bpm_y_positions])
        R[:, j] = (bpm_readings - ref_bpm) / delta_theta

    return R


__all__ = ["Trajectory", "track_beam", "compute_response_matrix"]
