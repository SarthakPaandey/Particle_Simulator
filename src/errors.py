"""Error / misalignment simulation for orbit distortion."""
from __future__ import annotations

import numpy as np
from typing import Optional


def generate_error_kicks(
    n_kicks: int,
    error_sigma: float = 5e-5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Gaussian random angular kicks (radians).

    Parameters
    ----------
    n_kicks : int
        Number of kicks to generate.
    error_sigma : float
        Standard deviation of each kick in radians.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    kicks : np.ndarray, shape (n_kicks,)
    """
    if n_kicks <= 0 or error_sigma <= 0:
        return np.zeros(max(n_kicks, 0))
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, error_sigma, size=n_kicks)


def generate_lattice_error_kicks(
    lattice: "Lattice",  # noqa: F821
    error_sigma: float = 5e-5,
    kick_type: str = "quads",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Gaussian random error kicks mapped to specific lattice elements in 4D.

    Parameters
    ----------
    lattice : Lattice
        The accelerator lattice.
    error_sigma : float
        Standard deviation of each kick in radians.
    kick_type : str
        "quads" (only quadrupoles get kicks) or "all" (all elements get kicks).
    seed : int, optional

    Returns
    -------
    kicks : np.ndarray, shape (2, len(lattice))
        Row 0: horizontal kicks, Row 1: vertical kicks.
    """
    kicks = np.zeros((2, len(lattice)))
    if error_sigma <= 0:
        return kicks

    rng = np.random.default_rng(seed)
    from .elements import Quadrupole

    for idx, el in enumerate(lattice.elements):
        if kick_type == "all" or (kick_type == "quads" and isinstance(el, Quadrupole)):
            kicks[0, idx] = rng.normal(0.0, error_sigma)
            kicks[1, idx] = rng.normal(0.0, error_sigma)
    return kicks


def apply_lattice_element_errors(
    lattice: "Lattice",  # noqa: F821
    sigma_quad_dx: float = 0.0,
    sigma_quad_dy: float = 0.0,
    sigma_quad_tilt: float = 0.0,
    sigma_bpm_offset: float = 0.0,
    sigma_bpm_gain: float = 0.0,
    seed: Optional[int] = None,
) -> None:
    """Generate and apply physical alignment & calibration errors directly to elements.

    Modifies the lattice in-place.
    """
    rng = np.random.default_rng(seed)
    from .elements import Quadrupole, BPM
    
    for el in lattice.elements:
        if isinstance(el, Quadrupole):
            el.dx = rng.normal(0.0, sigma_quad_dx) if sigma_quad_dx > 0 else 0.0
            el.dy = rng.normal(0.0, sigma_quad_dy) if sigma_quad_dy > 0 else 0.0
            el.tilt = rng.normal(0.0, sigma_quad_tilt) if sigma_quad_tilt > 0 else 0.0
        elif isinstance(el, BPM):
            el.dx = rng.normal(0.0, sigma_bpm_offset) if sigma_bpm_offset > 0 else 0.0
            el.dy = rng.normal(0.0, sigma_bpm_offset) if sigma_bpm_offset > 0 else 0.0
            el.gain_x = 1.0 + rng.normal(0.0, sigma_bpm_gain) if sigma_bpm_gain > 0 else 1.0
            el.gain_y = 1.0 + rng.normal(0.0, sigma_bpm_gain) if sigma_bpm_gain > 0 else 1.0
        else:
            if hasattr(el, 'dx'):
                el.dx = 0.0
            if hasattr(el, 'dy'):
                el.dy = 0.0
            if hasattr(el, 'tilt'):
                el.tilt = 0.0


__all__ = ["generate_error_kicks", "generate_lattice_error_kicks", "apply_lattice_element_errors"]

