from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"


def orbit_figure(
    s: np.ndarray,
    ideal_x: np.ndarray,
    distorted_x: np.ndarray,
    corrected_x: np.ndarray | dict[str, np.ndarray],
    bpm_s: np.ndarray,
    bpm_distorted_x: np.ndarray,
    bpm_corrected_x: np.ndarray | dict[str, np.ndarray],
    ideal_y: np.ndarray,
    distorted_y: np.ndarray,
    corrected_y: np.ndarray | dict[str, np.ndarray],
    bpm_distorted_y: np.ndarray,
    bpm_corrected_y: np.ndarray | dict[str, np.ndarray],
    hcor_s: Optional[Sequence[float]] = None,
    vcor_s: Optional[Sequence[float]] = None,
) -> "go.Figure":
    # Build a stacked layout with shared X axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Horizontal Plane (x)", "Vertical Plane (y)")
    )

    # Standard colors and symbols for correction methods
    method_styles = {
        "least-squares": dict(color="#1f77b4", symbol="square"),
        "SVD": dict(color="#9467bd", symbol="diamond"),
        "iterative-SVD": dict(color="#2ca02c", symbol="triangle-up"),
        "micado": dict(color="#ff7f0e", symbol="x")
    }

    # =========================================================================
    # ROW 1: Horizontal Plane (x)
    # =========================================================================
    fig.add_trace(go.Scatter(
        x=s, y=ideal_x * 1000, mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        legendgroup="Ideal", name="Ideal Orbit",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=s, y=distorted_x * 1000, mode="lines",
        line=dict(color="red", width=2),
        legendgroup="Distorted", name="Before Correction",
    ), row=1, col=1)

    if len(bpm_s) > 0:
        fig.add_trace(go.Scatter(
            x=bpm_s, y=bpm_distorted_x * 1000,
            mode="markers", marker=dict(color="red", size=8, symbol="circle"),
            legendgroup="BPM Distorted", name="BPM (before)",
        ), row=1, col=1)

    # Corrected x orbits
    if isinstance(corrected_x, dict):
        for name, x_corr in corrected_x.items():
            style = method_styles.get(name, dict(color="blue", symbol="square"))
            fig.add_trace(go.Scatter(
                x=s, y=x_corr * 1000, mode="lines",
                line=dict(color=style["color"], width=2),
                legendgroup=name, name=f"After {name}",
            ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=s, y=corrected_x * 1000, mode="lines",
            line=dict(color="#1f77b4", width=2),
            legendgroup="After Correction", name="After Correction",
        ), row=1, col=1)

    # Corrected x BPMs
    if len(bpm_s) > 0:
        if isinstance(bpm_corrected_x, dict):
            for name, x_bpm in bpm_corrected_x.items():
                style = method_styles.get(name, dict(color="blue", symbol="square"))
                fig.add_trace(go.Scatter(
                    x=bpm_s, y=x_bpm * 1000,
                    mode="markers", marker=dict(color=style["color"], size=7, symbol=style["symbol"]),
                    legendgroup=name, name=f"BPM ({name})",
                ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=bpm_s, y=bpm_corrected_x * 1000,
                mode="markers", marker=dict(color="#1f77b4", size=8, symbol="square"),
                legendgroup="After Correction", name="BPM (after)",
            ), row=1, col=1)

    # =========================================================================
    # ROW 2: Vertical Plane (y)
    # =========================================================================
    fig.add_trace(go.Scatter(
        x=s, y=ideal_y * 1000, mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        legendgroup="Ideal", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=s, y=distorted_y * 1000, mode="lines",
        line=dict(color="red", width=2),
        legendgroup="Distorted", showlegend=False,
    ), row=2, col=1)

    if len(bpm_s) > 0:
        fig.add_trace(go.Scatter(
            x=bpm_s, y=bpm_distorted_y * 1000,
            mode="markers", marker=dict(color="red", size=8, symbol="circle"),
            legendgroup="BPM Distorted", showlegend=False,
        ), row=2, col=1)

    # Corrected y orbits
    if isinstance(corrected_y, dict):
        for name, y_corr in corrected_y.items():
            style = method_styles.get(name, dict(color="blue", symbol="square"))
            fig.add_trace(go.Scatter(
                x=s, y=y_corr * 1000, mode="lines",
                line=dict(color=style["color"], width=2),
                legendgroup=name, showlegend=False,
            ), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=s, y=corrected_y * 1000, mode="lines",
            line=dict(color="#1f77b4", width=2),
            legendgroup="After Correction", showlegend=False,
        ), row=2, col=1)

    # Corrected y BPMs
    if len(bpm_s) > 0:
        if isinstance(bpm_corrected_y, dict):
            for name, y_bpm in bpm_corrected_y.items():
                style = method_styles.get(name, dict(color="blue", symbol="square"))
                fig.add_trace(go.Scatter(
                    x=bpm_s, y=y_bpm * 1000,
                    mode="markers", marker=dict(color=style["color"], size=7, symbol=style["symbol"]),
                    legendgroup=name, showlegend=False,
                ), row=2, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=bpm_s, y=bpm_corrected_y * 1000,
                mode="markers", marker=dict(color="#1f77b4", size=8, symbol="square"),
                legendgroup="After Correction", showlegend=False,
            ), row=2, col=1)

    # =========================================================================
    # Corrector markers (vertical lines)
    # =========================================================================
    if hcor_s is not None:
        for cs in hcor_s:
            fig.add_shape(type="line", x0=cs, x1=cs, y0=-1, y1=1, xref="x", yref="paper",
                          line=dict(color="blue", width=0.5, dash="dot"), row=1, col=1)
    if vcor_s is not None:
        for cs in vcor_s:
            fig.add_shape(type="line", x0=cs, x1=cs, y0=0, y1=2, xref="x", yref="paper",
                          line=dict(color="green", width=0.5, dash="dot"), row=2, col=1)

    fig.update_yaxes(title_text="x [mm]", row=1, col=1)
    fig.update_yaxes(title_text="y [mm]", row=2, col=1)
    fig.update_xaxes(title_text="s [m]", row=2, col=1)

    fig.update_layout(
        title="Beam Orbits (Transverse Planes)",
        height=680,
        legend=dict(orientation="h", y=-0.12),
        margin=dict(l=50, r=40, t=55, b=40)
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
