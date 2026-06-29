# Particle Beam Orbit Correction Simulator

A Python-based scientific simulation and interactive dashboard that models a simplified particle accelerator beamline, simulates beam orbit errors caused by magnet misalignments, and applies least-squares / SVD-based correction using Beam Position Monitors (BPMs) and corrector magnets.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Does

1. **Builds a FODO lattice** — alternating focusing/defocusing quadrupoles separated by drift spaces, with BPMs and corrector magnets.
2. **Tracks the beam** — propagates the state vector `[x, x']` through each element using 2×2 transfer matrices.
3. **Introduces orbit errors** — random angular kicks simulate magnet misalignments; Gaussian noise models BPM uncertainty.
4. **Computes the response matrix** — `R[i,j] = ∂BPM_i / ∂θ_j` relates every corrector to every BPM.
5. **Corrects the orbit** — solves `R c = -b` via least-squares, SVD pseudo-inverse, or iterative feedback.
6. **Visualises everything** — interactive Streamlit dashboard with Plotly charts.

## Mathematical Model

### Beam State

```
X = [x, x']ᵀ   where x = position [m], x' = angle [rad]
```

### Transfer Matrices

**Drift** (length L):

```
M_drift = [[1, L],
           [0, 1]]
```

**Focusing quadrupole** (focal length f):

```
M_QF = [[1,    0],
        [-1/f, 1]]
```

**Defocusing quadrupole**:

```
M_QD = [[1,   0],
        [1/f, 1]]
```

**Corrector kick**: `x'_new = x' + θ`

### Correction Problem

Given `b` = BPM error vector and `R` = response matrix, find `c` minimizing:

```
‖R c + b‖²
```

- **Least squares**: `c = -(RᵀR)⁻¹ Rᵀ b`
- **SVD**: `R = U Σ Vᵀ`, then `c = -V Σ⁺ Uᵀ b` (with optional singular-value truncation)
- **Iterative**: `c_{k+1} = c_k + α Δc_k`, where `0 < α ≤ 1` is the gain

### RMS Orbit Error

```
RMS = √( (1/N) Σ x_i² )
```

## Project Structure

```
particle-beam-orbit-simulator/
├── app.py                    Streamlit dashboard
├── requirements.txt
├── config/
│   └── default_config.yaml
├── src/
│   ├── __init__.py
│   ├── elements.py          Drift, Quadrupole, BPM, Corrector, ErrorKick
│   ├── beam.py              BeamState dataclass
│   ├── lattice.py           FODO lattice builder
│   ├── tracking.py          Beam tracking & response matrix
│   ├── errors.py            Random error-kick generation
│   ├── correction.py        LSQ, SVD, iterative correction
│   ├── metrics.py           RMS, max, improvement
│   ├── visualization.py     Matplotlib plots (PNG export)
│   ├── plotly_viz.py        Plotly figures (dashboard)
│   └── io_utils.py          CSV/JSON/YAML I/O
├── tests/
│   ├── test_elements.py
│   ├── test_tracking.py
│   └── test_correction.py
├── notebooks/
├── results/
└── docs/
    ├── mathematical_model.md
    └── cern_relevance.md
```

## Dashboard Sections

| Section | Content |
|---|---|
| **Lattice Layout** | Visual schematic of element positions |
| **Beam Orbit** | Plot of ideal, distorted, and corrected orbits |
| **Response Matrix** | Heatmap of `R[i,j]` |
| **Singular Values** | Log-scale spectrum with cutoff markers |
| **Corrector Strengths** | Bar chart of computed kick angles |
| **Convergence** | RMS vs iteration (iterative mode) |
| **Performance Metrics** | RMS before/after, improvement %, condition number |
| **Export** | Download BPM data, response matrix, parameters |

## Running Tests

```bash
pytest tests/ -v
```

## CERN Relevance

CERN operates accelerators (LHC, SPS, PS, LINACs) that rely on beam diagnostics, orbit correction, magnet control, feedback systems, and control-room dashboards. This project is a simplified educational version of those real-world systems. See [`docs/cern_relevance.md`](docs/cern_relevance.md) for details.

## License

MIT
