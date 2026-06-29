"""Plotting utilities for the orbit correction simulator.

Two interfaces are provided:

* Matplotlib ``Figure`` objects saved to PNG for reports and the README.
* Functions returning ``plotly.graph_objects.Figure`` for the Streamlit
  dashboard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Matplotlib helpers (saved PNGs for reports)
# ----------------------------------------------------------------------
def _save(fig, path: Optional[str]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_orbit(
    s: np.ndarray,
    ideal_x: np.ndarray,
    distorted_x: np.ndarray,
    corrected_x: np.ndarray,
    bpm_s: np.ndarray,
    bpm_distorted: np.ndarray,
    bpm_corrected: np.ndarray,
    corrector_s: Optional[Sequence[float]] = None,
    title: str = "Beam Orbit",
    ylabel: str = "x [mm]",
    xlabel: str = "s [m]",
    path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, ideal_x * 1000, "k--", lw=1.2, label="Ideal orbit", alpha=0.7)
    ax.plot(s, distorted_x * 1000, "r-", lw=1.5, label="Before correction")
    ax.plot(s, corrected_x * 1000, "b-", lw=1.5, label="After correction")

    if len(bpm_s) > 0:
        ax.scatter(bpm_s, bpm_distorted * 1000, c="red", s=40, marker="o", label="BPM (before)", zorder=5)
        ax.scatter(bpm_s, bpm_corrected * 1000, c="blue", s=40, marker="s", label="BPM (after)", zorder=5)

    if corrector_s is not None and len(corrector_s) > 0:
        for cs in corrector_s:
            ax.axvline(cs, color="green", alpha=0.2, lw=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, path)
    return fig


def plot_response_matrix(R: np.ndarray, title: str = "Response Matrix", path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.max(np.abs(R)) if R.size else 1.0
    im = ax.imshow(R, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Corrector index")
    ax.set_ylabel("BPM index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="d(BPM)/dθ [m/rad]")
    _save(fig, path)
    return fig


def plot_singular_values(s: np.ndarray, title: str = "Singular Value Spectrum", path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(s) + 1), s, "o-", lw=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value [m/rad]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, path)
    return fig


def plot_convergence(rms_history: np.ndarray, title: str = "Convergence", path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(rms_history)), rms_history, "o-", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS BPM error [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save(fig, path)
    return fig


def plot_corrector_strengths(
    strengths: np.ndarray,
    corrector_s: Sequence[float],
    title: str = "Corrector Strengths",
    path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(strengths))
    ax.bar(x, strengths * 1000.0)
    ax.set_xlabel("Corrector index")
    ax.set_ylabel("Strength [mrad]")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, path)
    return fig


__all__ = [
    "plot_orbit",
    "plot_response_matrix",
    "plot_singular_values",
    "plot_convergence",
    "plot_corrector_strengths",
]
