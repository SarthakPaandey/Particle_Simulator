"""Beam orbit tracking through a lattice."""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from .beam import BeamState
from .elements import BPM, Corrector, ErrorKick, Element
from .lattice import Lattice


class Trajectory:
    """Stores the beam state at every lattice position."""

    def __init__(self, lattice: Lattice):
        self.lattice = lattice
        self.s: np.ndarray = lattice.s_positions.copy()
        n = len(lattice)
        self.x: np.ndarray = np.zeros(n)
        self.xp: np.ndarray = np.zeros(n)
        self.bpm_readings: List[float] = []
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
            return np.array(self.bpm_readings)
        return np.array([0.0] * len(self.lattice.bpms))


def track_beam(
    lattice: Lattice,
    initial_state: BeamState,
    corrector_strengths: Optional[np.ndarray] = None,
    error_kicks: Optional[np.ndarray] = None,
    add_noise: bool = False,
    bpm_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Trajectory:
    """Propagate a single particle through the lattice.

    The beam state X = [x, x']^T is advanced element-by-element:
      X_{k+1} = M_k @ X_k + [0, C_k + E_k]^T
    where C_k is the corrector kick and E_k is the error kick at element k.

    Parameters
    ----------
    lattice : Lattice
        The accelerator lattice.
    initial_state : BeamState
        Starting state [x, x'].
    corrector_strengths : np.ndarray, optional
        Array of kick angles (radians) for each corrector.
    error_kicks : np.ndarray, optional
        Array of error kick angles per lattice element (length = len(lattice)).
    add_noise : bool
        If True, add Gaussian noise to BPM readings.
    bpm_noise_sigma : float
        Standard deviation of BPM noise in meters.
    rng : Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    traj : Trajectory
    """
    if corrector_strengths is not None:
        for corr, strength in zip(lattice.correctors, corrector_strengths):
            corr.strength = float(strength)

    if error_kicks is not None and len(error_kicks) != len(lattice):
        raise ValueError(
            f"error_kicks length {len(error_kicks)} != lattice length {len(lattice)}"
        )

    if rng is None:
        rng = np.random.default_rng()

    traj = Trajectory(lattice)
    state = initial_state.as_vector().copy()
    corr_idx = 0

    for idx, element in enumerate(lattice.elements):
        traj.x[idx] = float(state[0, 0])
        traj.xp[idx] = float(state[1, 0])

        state = element.matrix() @ state

        kick = 0.0

        if isinstance(element, Corrector):
            kick += element.strength

        if isinstance(element, ErrorKick):
            kick += element.kick

        # Generic error kicks applied element-by-element
        if error_kicks is not None:
            kick += error_kicks[idx]

        if kick != 0.0:
            state[1, 0] += kick

        if isinstance(element, BPM):
            x_true = float(state[0, 0])
            noise = 0.0
            if add_noise and bpm_noise_sigma > 0:
                noise = float(rng.normal(0.0, bpm_noise_sigma, 1)[0])
            traj._bpm_elements.append(element)
            traj.bpm_readings.append(x_true + noise)

    return traj


def compute_response_matrix(
    lattice: Lattice,
    delta_theta: float = 1e-4,
    additive_noise: bool = False,
    bpm_noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute the N_BPM x N_CORR response matrix via finite differences.

    R[i, j] = ΔBPM_i / Δθ_j   approximately  ∂x_i / ∂θ_j

    Parameters
    ----------
    lattice : Lattice
    delta_theta : float
        Small kick (radians) used for the finite-difference estimate.
    additive_noise, bpm_noise_sigma : passed to track_beam.
    rng : Generator, optional

    Returns
    -------
    R : np.ndarray, shape (n_bpm, n_corr)
    """
    bpms = lattice.bpms
    correctors = lattice.correctors
    n_bpm = len(bpms)
    n_corr = len(correctors)

    if n_bpm == 0 or n_corr == 0:
        return np.zeros((n_bpm, n_corr))

    if rng is None:
        rng = np.random.default_rng()

    # Reference: all correctors zero, with optional noise
    ref_traj = track_beam(lattice, BeamState(), None, None, additive_noise, bpm_noise_sigma, rng)
    ref_bpm = ref_traj.bpm_x_positions.reshape(-1, 1)

    R = np.zeros((n_bpm, n_corr))
    for j in range(n_corr):
        strengths = np.zeros(n_corr)
        strengths[j] = delta_theta
        traj = track_beam(lattice, BeamState(), strengths, None, additive_noise, bpm_noise_sigma, rng)
        R[:, j] = (traj.bpm_x_positions - ref_bpm[:, 0]) / delta_theta

    return R


__all__ = ["Trajectory", "track_beam", "compute_response_matrix"]
