"""Particle Beam Orbit Correction Simulator — Interactive Dashboard (4D & MICADO).

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
from src.correction import least_squares_correction, svd_correction, iterative_correction, micado_correction
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
    page_title="Beam Orbit Correction Simulator (4D)",
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
    st.title("⚛ 4D Configuration")
    
    st.header("1. Lattice Configuration")
    n_cells = st.number_input("Number of FODO cells", min_value=1, max_value=100, value=12, step=1)
    drift_length = st.number_input("Drift length [m]", min_value=0.1, value=1.0, step=0.1)
    quad_focal_length = st.number_input("Quad focal length [m]", min_value=0.5, value=8.0, step=0.5)
    bpms_per_cell = st.number_input("BPMs per cell", min_value=1, max_value=10, value=1, step=1)
    correctors_per_cell = st.number_input("Correctors per cell (each H/V)", min_value=1, max_value=10, value=1, step=1)

    st.header("2. Beam & Errors")
    st.subheader("Horizontal Initials")
    initial_offset_mm = st.number_input("Initial x offset [mm]", value=2.0, step=0.5)
    initial_angle_mrad = st.number_input("Initial x' angle [mrad]", value=0.1, step=0.05)
    
    st.subheader("Vertical Initials")
    initial_offset_y_mm = st.number_input("Initial y offset [mm]", value=1.5, step=0.5)
    initial_angle_y_mrad = st.number_input("Initial y' angle [mrad]", value=-0.05, step=0.05)
    
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
    selected_method = st.selectbox("Detailed view method", ["least-squares", "SVD", "iterative-SVD", "MICADO"])
    
    st.subheader("MICADO Config")
    n_micado_keep = st.number_input("MICADO correctors kept", min_value=1, max_value=100, value=6, step=1)
    
    st.subheader("SVD & Iterative Config")
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
y0 = float(initial_offset_y_mm) * 1e-3
yp0 = float(initial_angle_y_mrad) * 1e-3
initial_state = BeamState(x=x0, xp=xp0, y=y0, yp=yp0)

# ── Error kicks ────────────────────────────────────────
error_sigma_rad = float(error_kick_sigma_mrad) * 1e-3
kick_type = "quads" if error_placement == "Quadrupoles Only" else "all"

kicks = np.zeros((2, len(lattice)))
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
    error_kicks=np.zeros((2, len(lattice))),
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
bpm_error = np.concatenate([traj_distorted.bpm_x_positions, traj_distorted.bpm_y_positions])
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
c_iter, rms_history, traj_history, c_history, niters = iterative_correction(
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

# 4. MICADO
c_mic, res_mic = micado_correction(
    R, bpm_error, n_correctors=int(n_micado_keep),
    limit=corr_limit_rad if apply_limits else None
)
traj_mic = track_beam(
    lattice, initial_state,
    corrector_strengths=c_mic,
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
        "traj_history": traj_history,
        "c_history": c_history,
        "niters": niters,
    },
    "MICADO": {
        "c_star": c_mic,
        "traj": traj_mic,
        "rms_history": None,
        "niters": 1,
    }
}

# Selected method details
selected_d = all_methods[selected_method]
c_star_selected = selected_d["c_star"].ravel()
traj_corrected_selected = selected_d["traj"]

# ── Metrics Calculations ────────────────────────────────
rms_before_mm = rms_error(bpm_error * 1000)
max_before_mm = max_abs_error(bpm_error * 1000)

metrics_compare = []
for name, m_data in all_methods.items():
    bpm_after = np.concatenate([m_data["traj"].bpm_x_positions, m_data["traj"].bpm_y_positions])
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
    elif name == "MICADO":
        n_act = np.sum(~np.isclose(m_data["c_star"], 0.0))
        status = f"Active: {n_act}"
        
    if apply_limits and np.any(np.isclose(np.abs(m_data["c_star"]), corr_limit_rad)):
        status += " [Clipped]"
        
    metrics_compare.append({
        "Method": name,
        "RMS After [mm]": f"{rms_after_mm:.3f}",
        "Max |x,y| After [mm]": f"{max_after_mm:.3f}",
        "Improvement [%]": f"{pct_imp:.1f}%",
        "‖c‖ [mrad]": f"{c_norm_mrad:.3f}",
        "Max Kick [mrad]": f"{max_kick_mrad:.3f}",
        "Status": status
    })

df_compare = pd.DataFrame(metrics_compare)

# ============================================================================
# Main Page Render
# ============================================================================
st.title("⚛  4D Particle Beam Orbit Correction Simulator")
st.markdown(
    """
    Explore transverse beam dynamics and steering algorithms on FODO lattices. Adjust parameters in the sidebar 
    and navigate through the tabs below to step through the accelerator physics process.
    """
)

# Set up tabs
tabs = st.tabs([
    "🔬 1. Design & Lattice Layout",
    "⚠️ 2. Orbit Distortion & Errors",
    "🗺️ 3. Response Matrix Explorer",
    "🎯 4. Correction & Comparison",
    "🔁 5. Iterative Feedback Player"
])

bpm_s = np.array([bpm.s for bpm in lattice.bpms])
hcor_s = [c.s for c in lattice.correctors if c.plane == "x"]
vcor_s = [c.s for c in lattice.correctors if c.plane == "y"]

# ----------------------------------------------------------------------------
# TAB 1: Baseline Design & Lattice
# ----------------------------------------------------------------------------
with tabs[0]:
    st.header("Lattice Design & Baseline Configuration")
    st.markdown(
        """
        The baseline configuration displays the perfect nominal design structure of the accelerator. 
        Before errors are introduced, the beam follows the perfect design **Reference Orbit** ($x = 0, y = 0$ everywhere).
        """
    )
    
    col_l1, col_l2 = st.columns([5, 3])
    with col_l1:
        # Plot ideal orbit
        fig_ideal = orbit_figure(
            s=traj_ideal.s,
            ideal_x=traj_ideal.x,
            distorted_x=traj_ideal.x,
            corrected_x={},
            bpm_s=bpm_s,
            bpm_distorted_x=np.zeros_like(bpm_s),
            bpm_corrected_x={},
            ideal_y=traj_ideal.y,
            distorted_y=traj_ideal.y,
            corrected_y={},
            bpm_distorted_y=np.zeros_like(bpm_s),
            bpm_corrected_y={},
            hcor_s=hcor_s,
            vcor_s=vcor_s,
        )
        st.plotly_chart(fig_ideal, use_container_width=True)
        
    with col_l2:
        st.markdown("### 📋 Lattice Characteristics")
        st.markdown(
            f"""
            - **Total Circumference:** `{lattice.total_length():.2f} m`
            - **Lattice Elements:** `{len(lattice)}` total
            - **FODO Cells:** `{n_cells}` cells
            - **Quadrupoles:** `{2 * n_cells}` (focusing QF & defocusing QD)
            - **BPM Diagnostics:** `{len(lattice.bpms)}` dual-plane monitors
            - **Steering Correctors:** `{len(hcor_s)}` Horizontal (HCOR) + `{len(vcor_s)}` Vertical (VCOR)
            """
        )
        st.markdown("### 🔬 Baseline Lattice Elements Map")
        fig_lattice = lattice_layout_figure(lattice.s_positions, bpm_s, np.array(hcor_s + vcor_s))
        st.plotly_chart(fig_lattice, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: Orbit Distortion & Errors
# ----------------------------------------------------------------------------
with tabs[1]:
    st.header("Orbit Distortion & Error Representation")
    st.markdown(
        """
        Small alignment or magnetic field errors at the quadrupole magnets impart transverse angular kicks. 
        Coupled with the initial beam offset and angle, this causes the beam to deviate from the reference path.
        """
    )
    
    col_dist1, col_dist2 = st.columns([5, 2])
    with col_dist1:
        # Distorted orbit plot
        fig_distorted = orbit_figure(
            s=traj_ideal.s,
            ideal_x=traj_ideal.x,
            distorted_x=traj_distorted.x,
            corrected_x={},
            bpm_s=bpm_s,
            bpm_distorted_x=traj_distorted.bpm_x_positions,
            bpm_corrected_x={},
            ideal_y=traj_ideal.y,
            distorted_y=traj_distorted.y,
            corrected_y={},
            bpm_distorted_y=traj_distorted.bpm_y_positions,
            bpm_corrected_y={},
            hcor_s=hcor_s,
            vcor_s=vcor_s,
        )
        st.plotly_chart(fig_distorted, use_container_width=True)
        
    with col_dist2:
        st.markdown("### ⚠️ Orbit Deviation Metrics")
        
        # Display Metrics Cards
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Stacked RMS Orbit Error</div>
                <div class="metric-value">{rms_before_mm:.3f} mm</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">Peak Horiz. Offset (Max |x|)</div>
                <div class="metric-value">{max_abs_error(traj_distorted.x * 1000):.3f} mm</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">Peak Vert. Offset (Max |y|)</div>
                <div class="metric-value">{max_abs_error(traj_distorted.y * 1000):.3f} mm</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------------------------------------------------------------------
# TAB 3: Response Matrix Explorer
# ----------------------------------------------------------------------------
with tabs[2]:
    st.header("Response Matrix & Corrector Physics")
    st.markdown(
        """
        The **Response Matrix** $R$ represents the linear mapping from corrector strengths $c$ to BPM readings $b$. 
        Each column represents how the beam orbit responds to a single corrector magnet.
        """
    )
    
    col_rm1, col_rm2 = st.columns([1, 1])
    with col_rm1:
        st.subheader("Response Matrix Heatmap")
        fig_rm = response_figure(R)
        st.plotly_chart(fig_rm, use_container_width=True)
        
        st.markdown(f"- **Condition Number ($\\kappa$):** `{cond_number:.2e}`")
        if cond_number > 1e4:
            st.warning("⚠️ High Condition Number: System is ill-conditioned! Least-squares corrections may amplify noise excessively. SVD truncation is highly recommended.")
        else:
            st.success("✅ System is well-conditioned. Numerical correction is stable.")

    with col_rm2:
        st.subheader("🔍 Interactive Corrector Response Explorer")
        st.markdown(
            """
            Select any corrector magnet and set a test-kick. 
            The plot will display the optical **response function** of the beam to *only* this kick (this represents a single column of $R$).
            """
        )
        selected_corr = st.selectbox(
            "Select corrector to test-kick:", 
            lattice.correctors, 
            format_func=lambda c: f"{c.name} ({c.plane.upper()}-plane corrector at s={c.s:.2f} m)"
        )
        test_kick_val_mrad = st.number_input("Test-kick magnitude [mrad]", value=0.5, step=0.1)
        
        # Compute single-kick response
        test_kick_rad = test_kick_val_mrad * 1e-3
        test_strengths = np.zeros(len(lattice.correctors))
        test_strengths[selected_corr.index] = test_kick_rad
        
        traj_test_kick = track_beam(
            lattice, BeamState(),
            corrector_strengths=test_strengths,
            error_kicks=np.zeros((2, len(lattice))),
            add_noise=False,
            bpm_noise_sigma=0.0,
            rng=rng
        )
        
        # Plot test kick
        fig_test_kick = orbit_figure(
            s=traj_ideal.s,
            ideal_x=traj_ideal.x,
            distorted_x=traj_test_kick.x,
            corrected_x={},
            bpm_s=bpm_s,
            bpm_distorted_x=traj_test_kick.bpm_x_positions,
            bpm_corrected_x={},
            ideal_y=traj_ideal.y,
            distorted_y=traj_test_kick.y,
            corrected_y={},
            bpm_distorted_y=traj_test_kick.bpm_y_positions,
            bpm_corrected_y={},
            hcor_s=[selected_corr.s] if selected_corr.plane == "x" else [],
            vcor_s=[selected_corr.s] if selected_corr.plane == "y" else [],
        )
        st.plotly_chart(fig_test_kick, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4: Correction & Comparison
# ----------------------------------------------------------------------------
with tabs[3]:
    st.header("Correction Comparison & Solver Results")
    st.markdown(
        """
        Compare the final corrected orbits resulting from **Least-Squares**, **SVD truncation**, 
        **Iterative Feedback**, and **MICADO** algorithms under identical error conditions.
        """
    )
    
    col_comp1, col_comp2 = st.columns([5, 3])
    with col_comp1:
        # Gather corrected orbits for all methods
        corrected_x_dict = {name: m["traj"].x for name, m in all_methods.items()}
        corrected_y_dict = {name: m["traj"].y for name, m in all_methods.items()}
        corrected_bpm_x_dict = {name: m["traj"].bpm_x_positions for name, m in all_methods.items()}
        corrected_bpm_y_dict = {name: m["traj"].bpm_y_positions for name, m in all_methods.items()}

        fig_compare_orbit = orbit_figure(
            s=traj_ideal.s,
            ideal_x=traj_ideal.x,
            distorted_x=traj_distorted.x,
            corrected_x=corrected_x_dict,
            bpm_s=bpm_s,
            bpm_distorted_x=traj_distorted.bpm_x_positions,
            bpm_corrected_x=corrected_bpm_x_dict,
            ideal_y=traj_ideal.y,
            distorted_y=traj_distorted.y,
            corrected_y=corrected_y_dict,
            bpm_distorted_y=traj_distorted.bpm_y_positions,
            bpm_corrected_y=corrected_bpm_y_dict,
            hcor_s=hcor_s,
            vcor_s=vcor_s,
        )
        st.plotly_chart(fig_compare_orbit, use_container_width=True)
        
    with col_comp2:
        st.subheader("Performance Comparison Table")
        st.dataframe(df_compare.set_index("Method"), use_container_width=True)
        
        st.subheader("Singular Value Spectrum")
        fig_sv = singular_values_figure(s_svd, cutoff=float(svd_cutoff), n_keep=n_singular_val)
        st.plotly_chart(fig_sv, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 5: Iterative Feedback Player
# ----------------------------------------------------------------------------
with tabs[4]:
    st.header("Step-by-Step Iterative Feedback Player")
    st.markdown(
        """
        Iterative feedback corrections apply corrections scaled by a gain factor $\\alpha$. 
        Drag the slider below to step through each iteration and watch the closed orbit converge to the reference path.
        """
    )
    
    # Retrieve iterative history data
    rms_history_iter = all_methods["iterative-SVD"]["rms_history"]
    traj_history_iter = all_methods["iterative-SVD"]["traj_history"]
    c_history_iter = all_methods["iterative-SVD"]["c_history"]
    
    if rms_history_iter is not None and len(rms_history_iter) > 0:
        iter_step = st.slider(
            "Select Iteration Step:", 
            0, len(rms_history_iter) - 1, 0
        )
        st.markdown(f"**Step Details:** Iteration `{iter_step}` / `{len(rms_history_iter) - 1}` (RMS Orbit: `{rms_history_iter[iter_step]:.3f}` mm)")
        
        # Display alert status for the selected step
        step_rms = rms_history_iter[iter_step]
        if step_rms < float(tolerance_mm):
            st.success(f"✅ Step {iter_step} has converged! RMS: {step_rms:.3f} mm (< {tolerance_mm} mm)")
        else:
            st.warning(f"⚠️ Step {iter_step} is not yet converged. RMS: {step_rms:.3f} mm (>= {tolerance_mm} mm)")
            
        col_play1, col_play2 = st.columns([5, 3])
        with col_play1:
            traj_step = traj_history_iter[iter_step]
            
            fig_iter_orbit = orbit_figure(
                s=traj_ideal.s,
                ideal_x=traj_ideal.x,
                distorted_x=traj_distorted.x, # Keep distorted as reference
                corrected_x=traj_step.x,
                bpm_s=bpm_s,
                bpm_distorted_x=traj_distorted.bpm_x_positions,
                bpm_corrected_x=traj_step.bpm_x_positions,
                ideal_y=traj_ideal.y,
                distorted_y=traj_distorted.y,
                corrected_y=traj_step.y,
                bpm_distorted_y=traj_distorted.bpm_y_positions,
                bpm_corrected_y=traj_step.bpm_y_positions,
                hcor_s=hcor_s,
                vcor_s=vcor_s,
            )
            st.plotly_chart(fig_iter_orbit, use_container_width=True)
            
        with col_play2:
            st.markdown(f"**Corrector Strengths at Step {iter_step}**")
            c_step = c_history_iter[iter_step]
            fig_iter_corr = corrector_bar_figure(c_step)
            st.plotly_chart(fig_iter_corr, use_container_width=True)
            
            # Simple metrics at this step
            c_norm_step_mrad = float(np.linalg.norm(c_step) * 1000)
            st.metric("Corrector Norm ‖c‖", f"{c_norm_step_mrad:.3f} mrad")
    else:
        st.info("No iterative correction history found.")

# ============================================================================
# Section 7 — Export Results
# ============================================================================
st.subheader("Export & Download Data")

col_e1, col_e2, col_e3, col_e4 = st.columns(4)

with col_e1:
    bpm_df = pd.DataFrame({
        "s_bpm_m": traj_distorted.bpm_s_positions,
        "x_before_mm": traj_distorted.bpm_x_positions * 1000,
        "y_before_mm": traj_distorted.bpm_y_positions * 1000,
        "x_lsq_mm": traj_lsq.bpm_x_positions * 1000,
        "y_lsq_mm": traj_lsq.bpm_y_positions * 1000,
        "x_svd_mm": traj_svd.bpm_x_positions * 1000,
        "y_svd_mm": traj_svd.bpm_y_positions * 1000,
        "x_iter_mm": traj_iter.bpm_x_positions * 1000,
        "y_iter_mm": traj_iter.bpm_y_positions * 1000,
        "x_mic_mm": traj_mic.bpm_x_positions * 1000,
        "y_mic_mm": traj_mic.bpm_y_positions * 1000,
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
        index=[f"bpm_x_{i}" for i in range(len(lattice.bpms))] + [f"bpm_y_{i}" for i in range(len(lattice.bpms))],
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
        "initial_offset_x_mm": float(initial_offset_mm),
        "initial_angle_x_mrad": float(initial_angle_mrad),
        "initial_offset_y_mm": float(initial_offset_y_mm),
        "initial_angle_y_mrad": float(initial_angle_y_mrad),
        "random_seed": st.session_state["random_seed"],
        "error_kick_sigma_mrad": float(error_kick_sigma_mrad),
        "error_placement": error_placement,
        "bpm_noise_sigma_mm": float(bpm_noise_sigma_mm),
        "response_matrix_delta_theta_mrad": float(delta_theta_mrad),
        "response_matrix_measured_with_noise": noise_in_response_matrix,
        "micado_correctors_kept": int(n_micado_keep),
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
    "⚛ Particle Beam Orbit Correction Simulator (4D Dynamics)  •  "
    "Built with Python, NumPy, SciPy & Streamlit"
)
