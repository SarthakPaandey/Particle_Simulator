"""Orbit Correction Visualisation Helpers (Plotly-based, Streamlit-friendly).

These functions build interactive Plotly charts that match the dashboard
look-and-feel and don't require a browser roundtrip.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"


def orbit_figure(
    s: np.ndarray,
    ideal_x: np.ndarray,
    distorted_x: np.ndarray,
    corrected_x: np.ndarray | dict[str, np.ndarray],
    bpm_s: np.ndarray,
    bpm_distorted: np.ndarray,
    bpm_corrected: np.ndarray | dict[str, np.ndarray],
    corrector_s: Optional[Sequence[float]] = None,
) -> "go.Figure":
    fig = go.Figure()
    
    # Plot ideal orbit
    fig.add_trace(go.Scatter(
        x=s, y=ideal_x * 1000, mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="Ideal Orbit",
    ))
    
    # Plot distorted orbit
    fig.add_trace(go.Scatter(
        x=s, y=distorted_x * 1000, mode="lines",
        line=dict(color="red", width=2),
        name="Before Correction",
    ))

    # Plot distorted BPMs
    if len(bpm_s) > 0:
        fig.add_trace(go.Scatter(
            x=bpm_s, y=bpm_distorted * 1000,
            mode="markers", marker=dict(color="red", size=8, symbol="circle"),
            name="BPM (before)",
        ))

    # Standard colors for correction methods
    method_styles = {
        "least-squares": dict(color="#1f77b4", symbol="square"),  # Blue
        "SVD": dict(color="#9467bd", symbol="diamond"),          # Purple
        "iterative-SVD": dict(color="#2ca02c", symbol="triangle-up") # Green
    }

    # Plot corrected orbits
    if isinstance(corrected_x, dict):
        for name, x_corr in corrected_x.items():
            style = method_styles.get(name, dict(color="blue", symbol="square"))
            fig.add_trace(go.Scatter(
                x=s, y=x_corr * 1000, mode="lines",
                line=dict(color=style["color"], width=2),
                name=f"After {name}",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=s, y=corrected_x * 1000, mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="After Correction",
        ))

    # Plot corrected BPMs
    if len(bpm_s) > 0:
        if isinstance(bpm_corrected, dict):
            for name, x_bpm in bpm_corrected.items():
                style = method_styles.get(name, dict(color="blue", symbol="square"))
                fig.add_trace(go.Scatter(
                    x=bpm_s, y=x_bpm * 1000,
                    mode="markers", marker=dict(color=style["color"], size=7, symbol=style["symbol"]),
                    name=f"BPM ({name})",
                ))
        else:
            fig.add_trace(go.Scatter(
                x=bpm_s, y=bpm_corrected * 1000,
                mode="markers", marker=dict(color="#1f77b4", size=8, symbol="square"),
                name="BPM (after)",
            ))

    if corrector_s is not None:
        for cs in corrector_s:
            fig.add_vline(x=cs, line=dict(color="gray", width=0.5, dash="dot"))

    fig.update_layout(
        title="Beam Orbit (Horizontal Plane)",
        xaxis_title="s [m]",
        yaxis_title="x [mm]",
        height=480,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def response_figure(R: np.ndarray) -> "go.Figure":
    vmax = float(np.max(np.abs(R))) if R.size else 1.0
    fig = go.Figure(data=go.Heatmap(
        z=R, colorscale="RdBu", zmin=-vmax, zmax=vmax,
        x=[f"C{j}" for j in range(R.shape[1])],
        y=[f"BPM{i}" for i in range(R.shape[0])],
        colorbar=dict(title="dBPM/dθ<br>[m/rad]"),
    ))
    fig.update_layout(
        title="Response Matrix",
        xaxis_title="Corrector",
        yaxis_title="BPM",
        height=420,
    )
    return fig


def singular_values_figure(s: np.ndarray, cutoff: Optional[float] = None, n_keep: Optional[int] = None) -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(1, len(s) + 1), y=s,
        mode="markers+lines",
        name="singular values",
    ))
    if cutoff is not None:
        fig.add_hline(y=cutoff, line=dict(color="red", dash="dash"),
                      annotation_text=f"cutoff={cutoff:.1e}", annotation_position="top right")
    if n_keep is not None and n_keep > 0:
        fig.add_vline(x=n_keep + 0.5, line=dict(color="blue", dash="dot"),
                      annotation_text=f"keep={n_keep}", annotation_position="top right")
    fig.update_layout(
        title="Singular Value Spectrum",
        xaxis_title="Index",
        yaxis_title="Singular value [m/rad] (log)",
        yaxis_type="log",
        height=380,
    )
    return fig


def convergence_figure(rms_history: np.ndarray) -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(rms_history)), y=rms_history,
        mode="markers+lines", name="RMS BPM error",
    ))
    fig.update_layout(
        title="Iterative Correction Convergence",
        xaxis_title="Iteration",
        yaxis_title="RMS BPM error [mm]",
        height=380,
    )
    return fig


def corrector_bar_figure(strengths: np.ndarray) -> "go.Figure":
    fig = go.Figure(data=go.Bar(
        x=np.arange(len(strengths)),
        y=strengths * 1000.0,
    ))
    fig.update_layout(
        title="Corrector Strengths",
        xaxis_title="Corrector index",
        yaxis_title="θ [mrad]",
        height=350,
    )
    return fig


def lattice_layout_figure(s_positions: np.ndarray, bpms: np.ndarray, correctors: np.ndarray) -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_positions, y=[0] * len(s_positions),
        mode="lines+markers",
        line=dict(color="lightgray", width=3),
        marker=dict(size=2, color="gray"),
        name="Lattice",
        hoverinfo="x",
    ))
    if len(bpms) > 0:
        fig.add_trace(go.Scatter(
            x=bpms, y=[0] * len(bpms),
            mode="markers",
            marker=dict(symbol="diamond", color="blue", size=10),
            name="BPM",
        ))
    if len(correctors) > 0:
        fig.add_trace(go.Scatter(
            x=correctors, y=[0] * len(correctors),
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="Corrector",
        ))
    fig.update_layout(
        title="Lattice Layout",
        xaxis_title="s [m]",
        yaxis=dict(visible=False),
        height=200,
    )
    return fig


__all__ = [
    "orbit_figure",
    "response_figure",
    "singular_values_figure",
    "convergence_figure",
    "corrector_bar_figure",
    "lattice_layout_figure",
]
