"""Particle Beam Orbit Correction Simulator — Interactive Dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src`` is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.beam import BeamState
from src.lattice import Lattice, build_fodo_lattice
from src.tracking import track_beam, compute_response_matrix
from src.errors import generate_lattice_error_kicks
from src.correction import least_squares_correction, svd_correction, iterative_correction
from src.metrics import rms_error, max_abs_error, improvement
from src.plotly_viz import (
    orbit_figure,
    response_figure,
    singular_values_figure,
    convergence_figure,
    corrector_bar_figure,
    lattice_layout_figure,
)
from src.io_utils import (
    export_bpm_readings,
    export_response_matrix,
    export_response_matrix_npy,
    export_simulation_params,
    load_results_dir,
)

load_results_dir("results")

# ============================================================================
# Page config & Custom Theme Elements
# ============================================================================
st.set_page_config(
    page_title="Beam Orbit Correction Simulator",
    layout="wide",
    page_icon="⚛",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished, premium aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar — all user-configurable parameters
# ============================================================================
with st.sidebar:
    st.title("⚛ Configuration")
    
    st.header("1. Lattice Configuration")
    n_cells = st.number_input("Number of FODO cells", min_value=1, max_value=100, value=12, step=1)
    drift_length = st.number_input("Drift length [m]", min_value=0.1, value=1.0, step=0.1)
    quad_focal_length = st.number_input("Quad focal length [m]", min_value=0.5, value=8.0, step=0.5)
    bpms_per_cell = st.number_input("BPMs per cell", min_value=1, max_value=10, value=1, step=1)
    correctors_per_cell = st.number_input("Correctors per cell", min_value=1, max_value=10, value=1, step=1)

    st.header("2. Beam & Errors")
    initial_offset_mm = st.number_input("Initial offset [mm]", value=2.0, step=0.5)
    initial_angle_mrad = st.number_input("Initial angle [mrad]", value=0.1, step=0.05)
    
    # Simple randomize button that updates a seed in session state
    if "random_seed" not in st.session_state:
        st.session_state["random_seed"] = 42
        
    col_seed1, col_seed2 = st.columns([2, 1])
    with col_seed1:
        seed_input = st.number_input("Random seed", value=st.session_state["random_seed"], step=1)
        st.session_state["random_seed"] = int(seed_input)
    with col_seed2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🎲 New"):
            st.session_state["random_seed"] = int(np.random.default_rng().integers(1, 100000))
            st.rerun()

    add_random_errors = st.checkbox("Add random lattice errors", value=True)
    error_kick_sigma_mrad = st.number_input("Error kick σ [mrad]", value=0.05, step=0.01, min_value=0.0)
    error_placement = st.selectbox("Error kick placement", ["Quadrupoles Only", "All Elements"])
    
    add_bpm_noise = st.checkbox("Add BPM noise", value=True)
    bpm_noise_sigma_mm = st.number_input("BPM noise σ [mm]", value=0.02, step=0.01, min_value=0.0)

    st.header("3. Response Matrix Settings")
    delta_theta_mrad = st.number_input("Response matrix Δθ [mrad]", value=0.1, step=0.01, min_value=0.001)
    noise_in_response_matrix = st.checkbox("Measure RM with noise", value=False)

    st.header("4. Correction Settings")
    selected_method = st.selectbox("Detailed view method", ["least-squares", "SVD", "iterative-SVD"])
    svd_cutoff = st.number_input("SVD cutoff", value=1e-4, step=1e-5, format="%e", min_value=0.0)
    n_singular = st.number_input("SVD singular values kept (0=all)", min_value=0, value=0, step=1)
    gain = st.slider("Correction gain α", min_value=0.01, max_value=1.0, value=0.8, step=0.05)
    max_iterations = st.number_input("Max iterations", min_value=1, max_value=50, value=5, step=1)
    tolerance_mm = st.number_input("Convergence tolerance [mm]", value=0.05, step=0.01, min_value=0.001)
    apply_limits = st.checkbox("Apply corrector limits", value=True)
    corrector_limit_mrad = st.number_input("Corrector limit [mrad]", value=5.0, step=0.5, min_value=0.1)

# ============================================================================
# Core Simulator Run (Reactive)
# ============================================================================
# Setup RNG
rng = np.random.default_rng(st.session_state["random_seed"])

# ── Build lattice ─────────────────────────────────────
lattice = build_fodo_lattice(
    n_cells=int(n_cells),
    drift_length=float(drift_length),
    quad_focal_length=float(quad_focal_length),
    bpms_per_cell=int(bpms_per_cell),
    correctors_per_cell=int(correctors_per_cell),
)

# ── Initial beam state ─────────────────────────────────
x0 = float(initial_offset_mm) * 1e-3
xp0 = float(initial_angle_mrad) * 1e-3
initial_state = BeamState(x=x0, xp=xp0)

# ── Error kicks ────────────────────────────────────────
error_sigma_rad = float(error_kick_sigma_mrad) * 1e-3
kick_type = "quads" if error_placement == "Quadrupoles Only" else "all"

kicks = np.zeros(len(lattice))
if add_random_errors and error_sigma_rad > 0:
    kicks = generate_lattice_error_kicks(
        lattice, error_sigma=error_sigma_rad, kick_type=kick_type, seed=st.session_state["random_seed"]
    )

# ── Distorted (uncorrected) trajectory ─────────────────
noise_sigma_m = float(bpm_noise_sigma_mm) * 1e-3
traj_distorted = track_beam(
    lattice, initial_state,
    corrector_strengths=None,
    error_kicks=kicks,
    add_noise=add_bpm_noise,
    bpm_noise_sigma=noise_sigma_m,
    rng=rng,
)

# ── Ideal (zero-error) trajectory ───────────────────────
traj_ideal = track_beam(
    lattice, BeamState(),
    corrector_strengths=None,
    error_kicks=np.zeros(len(lattice)),
    add_noise=False,
    bpm_noise_sigma=0.0,
    rng=rng,
)

# ── Response matrix ────────────────────────────────────
delta_theta_rad = float(delta_theta_mrad) * 1e-3
R = compute_response_matrix(
    lattice, delta_theta=delta_theta_rad,
    additive_noise=noise_in_response_matrix,
    bpm_noise_sigma=noise_sigma_m,
    rng=rng,
)
_, s_svd, _ = np.linalg.svd(R, full_matrices=False)
cond_number = float(s_svd[0] / s_svd[-1]) if len(s_svd) > 0 and s_svd[-1] > 0 else float("inf")

# ── Run Corrections in Parallel ─────────────────────────
bpm_error = traj_distorted.bpm_x_positions
n_singular_val = int(n_singular) if int(n_singular) > 0 else None
corr_limit_rad = float(corrector_limit_mrad) * 1e-3

# 1. Least Squares
c_lsq, _res_lsq, _rank_lsq, _sv_lsq = least_squares_correction(
    R, bpm_error, limit=corr_limit_rad if apply_limits else None
)
traj_lsq = track_beam(
    lattice, initial_state,
    corrector_strengths=c_lsq,
    error_kicks=kicks,
    add_noise=add_bpm_noise,
    bpm_noise_sigma=noise_sigma_m,
    rng=rng,
)

# 2. SVD
c_svd, _U_svd, _s_svd, _Vt_svd = svd_correction(
    R, bpm_error, cutoff=float(svd_cutoff), n_singular=n_singular_val,
    limit=corr_limit_rad if apply_limits else None
)
traj_svd = track_beam(
    lattice, initial_state,
    corrector_strengths=c_svd,
    error_kicks=kicks,
    add_noise=add_bpm_noise,
    bpm_noise_sigma=noise_sigma_m,
    rng=rng,
)

# 3. Iterative SVD
c_iter, rms_history, niters = iterative_correction(
    lattice, initial_state, R,
    method="svd",
    gain=float(gain),
    max_iterations=int(max_iterations),
    tolerance_mm=float(tolerance_mm),
    corrector_limit=corr_limit_rad,
    apply_limits=apply_limits,
    svd_cutoff=float(svd_cutoff),
    n_singular=n_singular_val,
    add_noise=add_bpm_noise,
    bpm_noise_sigma=noise_sigma_m,
    error_kicks=kicks,
    rng=rng,
)
traj_iter = track_beam(
    lattice, initial_state,
    corrector_strengths=c_iter,
    error_kicks=kicks,
    add_noise=add_bpm_noise,
    bpm_noise_sigma=noise_sigma_m,
    rng=rng,
)

# Map methods for easy display and lookups
all_methods = {
    "least-squares": {
        "c_star": c_lsq,
        "traj": traj_lsq,
        "rms_history": None,
        "niters": 1,
    },
    "SVD": {
        "c_star": c_svd,
        "traj": traj_svd,
        "rms_history": None,
        "niters": 1,
    },
    "iterative-SVD": {
        "c_star": c_iter,
        "traj": traj_iter,
        "rms_history": rms_history,
        "niters": niters,
    }
}

# Selected method details
selected_d = all_methods[selected_method]
c_star_selected = selected_d["c_star"].ravel()
traj_corrected_selected = selected_d["traj"]

# ── Metrics Calculations ────────────────────────────────
bpm_before = traj_distorted.bpm_x_positions
rms_before_mm = rms_error(bpm_before * 1000)
max_before_mm = max_abs_error(bpm_before * 1000)

metrics_compare = []
for name, m_data in all_methods.items():
    bpm_after = m_data["traj"].bpm_x_positions
    rms_after_mm = rms_error(bpm_after * 1000)
    max_after_mm = max_abs_error(bpm_after * 1000)
    pct_imp = improvement(rms_before_mm, rms_after_mm)
    c_norm_mrad = float(np.linalg.norm(m_data["c_star"]) * 1000)
    max_kick_mrad = float(np.max(np.abs(m_data["c_star"])) * 1000) if len(m_data["c_star"]) > 0 else 0.0
    
    # Assess status
    status = "Success"
    if name == "iterative-SVD":
        if rms_after_mm < float(tolerance_mm):
            status = f"Converged ({m_data['niters']} iters)"
        else:
            status = "Max Iters Reached"
    if apply_limits and np.any(np.isclose(np.abs(m_data["c_star"]), corr_limit_rad)):
        status += " [Clipped]"
        
    metrics_compare.append({
        "Method": name,
        "RMS After [mm]": f"{rms_after_mm:.3f}",
        "Max |x| After [mm]": f"{max_after_mm:.3f}",
        "Improvement [%]": f"{pct_imp:.1f}%",
        "‖c‖ [mrad]": f"{c_norm_mrad:.3f}",
        "Max Kick [mrad]": f"{max_kick_mrad:.3f}",
        "Status": status
    })

df_compare = pd.DataFrame(metrics_compare)

# ============================================================================
# Main Page Render
# ============================================================================
st.title("⚛  Particle Beam Orbit Correction Simulator")
st.markdown(
    """
    This dashboard simulates **beam orbit distortion** in a simplified
    FODO accelerator lattice and applies **least-squares / SVD-based orbit
    correction** using Beam Position Monitors (BPMs) and corrector magnets.
    
    Measurements and kicks are computed reactively as you change sidebar parameters.
    """
)

# ============================================================================
# Section 1 — Project Overview
# ============================================================================
with st.expander("📖  Project Overview & Mathematical Model", expanded=False):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            """
            ### Physics Framework
            In a particle accelerator, charged beams propagate through magnetic channels. 
            The beam's transverse state vector is modeled in 2D phase space:
            $$X = [x, x']^T$$
            where $x$ is position (m) and $x'$ is the angle/divergence (rad).
            
            We propagate the beam using transfer matrices $M$:
            - **Drift space** of length $L$:
              $$M_{\\text{drift}} = \\begin{pmatrix} 1 & L \\\\ 0 & 1 \\end{pmatrix}$$
            - **Thin Quadrupole** magnet with focal length $f$:
              $$M_{\\text{quad}} = \\begin{pmatrix} 1 & 0 \\\\ \\mp 1/f & 1 \\end{pmatrix}$$
              *(focusing: $-1/f$; defocusing: $+1/f$)*
            """
        )
    with col_info2:
        st.markdown(
            """
            ### Orbit Correction Problem
            The relation between corrector kicks $c$ and BPM deviations $b$ is described by the **Response Matrix** $R$:
            $$R c \\approx -b$$
            where $R_{ij} = \\frac{\\partial x_i}{\\partial \\theta_j}$. We solve this using:
            1. **Least-Squares:** Minimized via ordinary least squares: 
               $$c = -R^{\\dagger}b$$
            2. **SVD Truncation:** $R = U \\Sigma V^T$, where the pseudo-inverse $R^+$ is constructed by truncating small singular values to suppress noise amplification:
               $$c = -V \\Sigma^+ U^T b$$
            3. **Iterative Feedback:** Scaled corrections applied in loop:
               $$c^{(k+1)} = c^{(k)} - \\alpha R^+ b^{(k)}$$
            """
        )

# ============================================================================
# Section 2 — Lattice layout
# ============================================================================
st.subheader("Accelerator Lattice Layout")
bpm_s = np.array([bpm.s for bpm in lattice.bpms])
corr_s = np.array([c.s for c in lattice.correctors])
fig_lattice = lattice_layout_figure(lattice.s_positions, bpm_s, corr_s)
st.plotly_chart(fig_lattice, use_container_width=True)

st.caption(
    f"Lattice Circumference: **{lattice.total_length():.2f} m**  |  "
    f"Elements: **{len(lattice)}**  |  "
    f"BPMs: **{len(lattice.bpms)}**  |  "
    f"Correctors: **{len(lattice.correctors)}**"
)

# ============================================================================
# Section 3 — Orbit visualization
# ============================================================================
st.subheader("Beam Trajectory & Orbit Correction")

# Gather all corrected orbits for side-by-side plotting
corrected_orbits_dict = {
    "least-squares": traj_lsq.x,
    "SVD": traj_svd.x,
    "iterative-SVD": traj_iter.x,
}
corrected_bpms_dict = {
    "least-squares": traj_lsq.bpm_x_positions,
    "SVD": traj_svd.bpm_x_positions,
    "iterative-SVD": traj_iter.bpm_x_positions,
}

fig_orbit = orbit_figure(
    s=traj_ideal.s,
    ideal_x=traj_ideal.x,
    distorted_x=traj_distorted.x,
    corrected_x=corrected_orbits_dict,
    bpm_s=traj_distorted.bpm_s_positions,
    bpm_distorted=traj_distorted.bpm_x_positions,
    bpm_corrected=corrected_bpms_dict,
    corrector_s=corr_s.tolist(),
)
st.plotly_chart(fig_orbit, use_container_width=True)

# ============================================================================
# Section 4 — Method Comparison Table & Selected Iterative Convergence
# ============================================================================
col_table, col_status = st.columns([5, 3])

with col_table:
    st.subheader("Performance Comparison Table")
    # Display comparison metrics
    st.dataframe(
        df_compare.set_index("Method"), 
        use_container_width=True,
    )
    
with col_status:
    st.subheader("Selected Method Status")
    
    # 1. Base details card
    sel_rms = rms_error(traj_corrected_selected.bpm_x_positions * 1000)
    sel_max = max_abs_error(traj_corrected_selected.bpm_x_positions * 1000)
    sel_improve = improvement(rms_before_mm, sel_rms)
    
    st.markdown(
        f"""
        **Selected Method:** `{selected_method}`
        
        - RMS BPM Orbit: **{sel_rms:.3f} mm** *(Before: {rms_before_mm:.3f} mm)*
        - Peak Orbit Deviation: **{sel_max:.3f} mm** *(Before: {max_before_mm:.3f} mm)*
        - Correction Efficiency: **{sel_improve:.1f}%**
        """
    )
    
    # 2. Iterative SVD convergence status alerts
    if selected_method == "iterative-SVD" and rms_history is not None:
        if sel_rms < float(tolerance_mm):
            st.success(f"✅ Converged to {sel_rms:.3f} mm (< {tolerance_mm} mm) in {niters} iterations!")
        else:
            st.warning(f"⚠️ Failed to converge within tolerance ({tolerance_mm} mm) after {max_iterations} iterations. Residual RMS: {sel_rms:.3f} mm")
    else:
        st.info("ℹ️ Least-Squares and SVD are direct matrix inverse methods and resolve in a single step.")

# ============================================================================
# Section 5 — Response matrix & singular values
# ============================================================================
col_rm, col_sv = st.columns(2)

with col_rm:
    st.subheader("Response Matrix & Numerical Stability")
    fig_rm = response_figure(R)
    st.plotly_chart(fig_rm, use_container_width=True)
    
    # Add numerical stability analysis
    st.markdown(
        f"""
        - **Condition Number ($\kappa$):** `{cond_number:.2e}`
        """
    )
    if cond_number > 1e4:
        st.warning("⚠️ High Condition Number: System is ill-conditioned! Least-squares corrections may amplify noise excessively. SVD truncation is highly recommended.")
    else:
        st.success("✅ System is well-conditioned. Numerical correction is stable.")

with col_sv:
    st.subheader("Singular Value Spectrum")
    fig_sv = singular_values_figure(s_svd, cutoff=float(svd_cutoff), n_keep=n_singular_val)
    st.plotly_chart(fig_sv, use_container_width=True)

# ============================================================================
# Section 6 — Corrector strengths & Convergence details
# ============================================================================
col_corr, col_conv = st.columns(2)

with col_corr:
    st.subheader("Corrector Strengths (Selected Method)")
    fig_corr = corrector_bar_figure(c_star_selected)
    st.plotly_chart(fig_corr, use_container_width=True)

with col_conv:
    if selected_method == "iterative-SVD" and rms_history is not None:
        st.subheader("Iterative Convergence History")
        fig_conv = convergence_figure(rms_history)
        st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.subheader("Correction Info")
        st.info("Select `iterative-SVD` under Configuration -> Correction Settings to view convergence history.")

# ============================================================================
# Section 7 — Export Results
# ============================================================================
st.subheader("Export & Download Data")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)

with col_e1:
    bpm_df = pd.DataFrame({
        "s_bpm_m": traj_distorted.bpm_s_positions,
        "x_before_mm": traj_distorted.bpm_x_positions * 1000,
        "x_lsq_mm": traj_lsq.bpm_x_positions * 1000,
        "x_svd_mm": traj_svd.bpm_x_positions * 1000,
        "x_iter_mm": traj_iter.bpm_x_positions * 1000,
    })
    st.download_button(
        "📊  BPM Readings (CSV)",
        bpm_df.to_csv(index=False),
        file_name="bpm_readings.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_e2:
    R_df = pd.DataFrame(
        R,
        columns=[f"corrector_{j}" for j in range(R.shape[1])],
        index=[f"bpm_{i}" for i in range(R.shape[0])],
    )
    st.download_button(
        "📊  Response Matrix (CSV)",
        R_df.to_csv(index=True),
        file_name="response_matrix.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_e3:
    # Binary export to memory buffer
    npy_buffer = io.BytesIO()
    np.save(npy_buffer, R)
    st.download_button(
        "💾  Response Matrix (NumPy .npy)",
        data=npy_buffer.getvalue(),
        file_name="response_matrix.npy",
        mime="application/octet-stream",
        use_container_width=True,
    )

with col_e4:
    params = {
        "n_cells": int(n_cells),
        "drift_length_m": float(drift_length),
        "quad_focal_length_m": float(quad_focal_length),
        "initial_offset_mm": float(initial_offset_mm),
        "initial_angle_mrad": float(initial_angle_mrad),
        "random_seed": st.session_state["random_seed"],
        "error_kick_sigma_mrad": float(error_kick_sigma_mrad),
        "error_placement": error_placement,
        "bpm_noise_sigma_mm": float(bpm_noise_sigma_mm),
        "response_matrix_delta_theta_mrad": float(delta_theta_mrad),
        "response_matrix_measured_with_noise": noise_in_response_matrix,
        "svd_cutoff": float(svd_cutoff),
        "n_singular_values_retained": n_singular_val,
        "gain": float(gain),
        "max_iterations": int(max_iterations),
        "tolerance_mm": float(tolerance_mm),
        "apply_limits": apply_limits,
        "corrector_limit_mrad": float(corrector_limit_mrad),
    }
    st.download_button(
        "📋  Parameters (JSON)",
        json.dumps(params, indent=2),
        file_name="simulation_params.json",
        mime="application/json",
        use_container_width=True,
    )

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.caption(
    "⚛ Particle Beam Orbit Correction Simulator  •  "
    "Built with Python, NumPy, SciPy & Streamlit"
)
