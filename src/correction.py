"""Orbit correction algorithms: least-squares, SVD, and iterative feedback."""
from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, svd
from typing import Optional, Tuple


def least_squares_correction(
    R: np.ndarray,
    bpm_error: np.ndarray,
    limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Solve ``R c ≈ -b`` by ordinary least squares.

    Parameters
    ----------
    R : np.ndarray, shape (n_bpm, n_corr)
        Response matrix.
    bpm_error : np.ndarray, shape (n_bpm,)
        BPM orbit error vector (measured positions).
    limit : float, optional
        If given, clip the corrector strengths to [-limit, limit] in radians.

    Returns
    -------
    c : np.ndarray, shape (n_corr,)
        Optimal corrector strengths (radians).
    residuals : np.ndarray
        Sums of squared residuals.
    rank : int
        Effective rank of R.
    singular_values : np.ndarray
        Singular values of R.
    """
    c, residuals, rank, singular_values = lstsq(R, -bpm_error, rcond=None)
    if limit is not None and limit > 0:
        c = np.clip(c, -limit, limit)
    return c, residuals, rank, singular_values


def svd_correction(
    R: np.ndarray,
    bpm_error: np.ndarray,
    cutoff: float = 1e-4,
    n_singular: Optional[int] = None,
    limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``R c ≈ -b`` using truncated SVD pseudo-inverse.

    Parameters
    ----------
    R : np.ndarray, shape (n_bpm, n_corr)
    bpm_error : np.ndarray, shape (n_bpm,)
    cutoff : float
        Singular values below this threshold are suppressed.
    n_singular : int, optional
        If given, keep only the ``n_singular`` largest singular values.
    limit : float, optional
        If given, clip the corrector strengths to [-limit, limit] in radians.

    Returns
    -------
    c : np.ndarray, shape (n_corr,)
    U : np.ndarray
    S : np.ndarray
    Vt : np.ndarray
    """
    U, s, Vt = svd(R, full_matrices=False)
    inv_s = np.zeros_like(s)

    if n_singular is not None and 0 < n_singular <= len(s):
        safe = s[:n_singular] > 0
        inv_s[:n_singular][safe] = 1.0 / s[:n_singular][safe]
    else:
        mask = s > cutoff
        inv_s[mask] = 1.0 / s[mask]

    # Pseudo-inverse: c = -V @ diag(inv_s) @ U^T @ b
    c = -Vt.T @ (inv_s * (U.T @ bpm_error))

    if limit is not None and limit > 0:
        c = np.clip(c, -limit, limit)

    return c, U, s, Vt


def iterative_correction(
    lattice: "Lattice",  # noqa: F821
    initial_state: "BeamState",  # noqa: F821
    response_matrix: np.ndarray,
    method: str = "svd",
    gain: float = 0.8,
    max_iterations: int = 5,
    tolerance_mm: float = 0.05,
    corrector_limit: float = 1e-3,
    apply_limits: bool = False,
    svd_cutoff: float = 1e-4,
    n_singular: Optional[int] = None,
    add_noise: bool = False,
    bpm_noise_sigma: float = 0.0,
    error_kicks: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Iterative feedback orbit correction in 4D.

    At each iteration:
      1. Track beam with current corrector strengths.
      2. Measure BPM orbit error b (stacked x and y).
      3. Compute correction delta c via chosen method.
      4. Apply c += gain * delta c.
      5. Check convergence via RMS BPM error.
      6. Stop if RMS < tolerance or max iterations reached.
    """
    from .beam import BeamState
    from .tracking import track_beam
    from .metrics import rms_error

    if error_kicks is None:
        error_kicks = np.zeros((2, len(lattice)))

    c = np.zeros(len(lattice.correctors))
    rms_history: list = []

    for iteration in range(max_iterations + 1):
        traj = track_beam(
            lattice,
            initial_state,
            corrector_strengths=c,
            error_kicks=error_kicks,
            add_noise=add_noise,
            bpm_noise_sigma=bpm_noise_sigma,
            rng=rng,
        )

        # Concatenate horizontal and vertical readings
        bpm_error = np.concatenate([traj.bpm_x_positions, traj.bpm_y_positions])
        rms_current = rms_error(bpm_error * 1000.0)
        rms_history.append(rms_current)

        if iteration == max_iterations or rms_current < tolerance_mm:
            break

        if method == "lsq":
            delta_c, *_ = least_squares_correction(response_matrix, bpm_error, limit=corrector_limit if apply_limits else None)
        elif method == "micado":
            # Retain a default subset (e.g., 6 correctors) or use full
            delta_c, _ = micado_correction(response_matrix, bpm_error, n_correctors=6, limit=corrector_limit if apply_limits else None)
        else:
            delta_c, *_ = svd_correction(response_matrix, bpm_error, cutoff=svd_cutoff, n_singular=n_singular, limit=corrector_limit if apply_limits else None)

        c += gain * delta_c.ravel()

        if apply_limits:
            c = np.clip(c, -corrector_limit, corrector_limit)

    return c, np.array(rms_history), len(rms_history) - 1


def micado_correction(
    R: np.ndarray,
    bpm_error: np.ndarray,
    n_correctors: int = 6,
    limit: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Solve ``R c ≈ -b`` using the MICADO algorithm.

    This algorithm iteratively selects the single most effective corrector
    at each step to minimize the residual sum of squares.

    Parameters
    ----------
    R : np.ndarray, shape (2 * n_bpm, n_corr)
    bpm_error : np.ndarray, shape (2 * n_bpm,)
    n_correctors : int
        Number of correctors to activate.
    limit : float, optional
        If given, clip individual corrector strengths.

    Returns
    -------
    c : np.ndarray, shape (n_corr,)
        Corrector strengths.
    residual : float
        Sum of squared residuals.
    """
    n_bpm, n_corr = R.shape
    b = -bpm_error
    selected_indices = []
    
    c = np.zeros(n_corr)
    
    n_use = min(n_correctors, n_corr)
    if n_use <= 0:
        return c, float(np.sum(b**2))
        
    for step in range(n_use):
        best_j = -1
        best_residual = float("inf")
        best_c_sub = None
        
        for j in range(n_corr):
            if j in selected_indices:
                continue
            
            test_indices = selected_indices + [j]
            R_sub = R[:, test_indices]
            
            # Solve subset system
            c_sub, _, _, _ = lstsq(R_sub, b, rcond=None)
            
            if limit is not None and limit > 0:
                c_sub = np.clip(c_sub, -limit, limit)
                
            pred = R_sub @ c_sub
            res = float(np.sum((pred - b) ** 2))
            
            if res < best_residual:
                best_residual = res
                best_j = j
                best_c_sub = c_sub
                
        if best_j == -1:
            break
            
        selected_indices.append(best_j)
        c = np.zeros(n_corr)
        for idx_sub, j_idx in enumerate(selected_indices):
            c[j_idx] = best_c_sub[idx_sub]
            
    residual_final = float(np.sum((R @ c - b) ** 2))
    return c, residual_final


__all__ = ["least_squares_correction", "svd_correction", "iterative_correction", "micado_correction"]

